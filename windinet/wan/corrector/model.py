"""The corrector: a small U-Net that adds a residual to each frame of Wan's decoded rollout, and its rollout."""

from __future__ import annotations

import torch
from diffusers import UNet2DModel
from torch import nn

from windinet.data.frames import FIELDS
from windinet.wan.corrector.config import CorrectorModelConfig
from windinet.wan.dit.model import autocast


class Corrector(nn.Module):
    """C(t) = W(t) + std * U-Net([true frame 0,] C(t-k..t-1), W(t-1), W(t), W(t+1)), inputs z-scored with the dataset
    stats, the frame index as the U-Net's timestep and gamma as its class embedding; physical units in and out."""

    def __init__(self, cfg: CorrectorModelConfig, stats: dict):
        super().__init__()
        self.history, self.first_frame, self.precision = cfg.history, cfg.first_frame, cfg.precision
        levels = len(cfg.channels)
        self.unet = UNet2DModel(
            in_channels=4 * (3 + cfg.history + cfg.first_frame),
            out_channels=4,
            block_out_channels=cfg.channels,
            layers_per_block=cfg.layers_per_block,
            down_block_types=("DownBlock2D",) * levels,
            up_block_types=("UpBlock2D",) * levels,
            add_attention=False,
            act_fn="gelu",
            resnet_time_scale_shift="scale_shift",
            class_embed_type="timestep",  # gamma: the timestep's sinusoids, its own MLP, added to the frame embedding
        )
        for m in self.unet.modules():
            if isinstance(m, nn.Conv2d) and m.padding != (0, 0):
                m.padding_mode = "replicate"  # outflow on every side: zero padding would read as a wall
        nn.init.zeros_(self.unet.conv_out.weight)  # starts as Wan alone
        nn.init.zeros_(self.unet.conv_out.bias)
        self.register_buffer("mean", torch.tensor([stats[f]["mean"] for f in FIELDS]).view(4, 1, 1))
        self.register_buffer("std", torch.tensor([stats[f]["std"] for f in FIELDS]).view(4, 1, 1))

    def forward(self, past: torch.Tensor, wan: torch.Tensor, t: int, gamma: torch.Tensor) -> torch.Tensor:
        """C(t) [B, 4, H, W] from `past` [B, n, 4, H, W] (the true frame 0, then up to the last `history` of C(1..t-1))
        and Wan's frames t-1, t, t+1 of `wan` [B, F, 4, H, W] (t+1 clamped to F-1)."""
        frames = wan[:, [t - 1, t, min(t + 1, wan.shape[1] - 1)]]
        if self.history:
            own = past[:, -self.history :]
            own = torch.cat([own[:, :1].expand(-1, self.history - own.shape[1], -1, -1, -1), own], 1)
            frames = torch.cat([own, frames], 1)
        if self.first_frame:
            frames = torch.cat([past[:, :1], frames], 1)
        x = ((frames - self.mean) / self.std).flatten(1, 2)
        frame = torch.full((len(x),), t, device=x.device)
        with autocast(self.precision, x.device):  # gamma in (1, 2) spread over the timestep range
            out = self.unet(x, frame, class_labels=(gamma - 1) * 1000).sample
        return wan[:, t] + self.std * out.float()


def rollout(model: Corrector, wan: torch.Tensor, past: torch.Tensor, gamma, steps: range) -> torch.Tensor:
    """`past` = C(0..steps.start-1) [B, n, 4, H, W] with C(0) the true frame 0, then C(t) for t in `steps`, each from
    the frames before; returns all of them."""
    out = list(past.unbind(1))
    for t in steps:
        own = out[1:][-model.history :] if model.history else []
        out.append(model(torch.stack(out[:1] + own, 1), wan, t, gamma))
    return torch.stack(out, 1)


def teacher_forced(model: Corrector, wan: torch.Tensor, truth: torch.Tensor, gamma, steps: range, noise: float):
    """C(t) for t in `steps`, each from the true frames before it, the history ones plus `noise` field stds of noise."""
    out = []
    for t in steps:
        own = truth[:, max(1, t - model.history) : t]
        own = own + noise * model.std * torch.randn_like(own)
        out.append(model(torch.cat([truth[:, :1], own], 1), wan, t, gamma))
    return torch.stack(out, 1)
