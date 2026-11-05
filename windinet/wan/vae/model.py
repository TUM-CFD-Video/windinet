"""Wan 2.1 VAE with normalised latents, frozen unless `train` says otherwise.

    rgb [B, 3, F, H, W] in (-1, 1), F = 4k + 1   --encode-->   z [B, 16, (F-1)/4 + 1, H/8, W/8]

`encode` standardises with the checkpoint's own latents_mean / latents_std (what the Wan pipeline feeds its
transformer, so natural-video latents look like N(0, 1) per channel); `decode` undoes that.
"""

from __future__ import annotations

import math

import torch
from diffusers import AutoencoderKLWan
from torch import nn
from torch.nn.functional import pad
from torch.utils.checkpoint import checkpoint

from windinet.wan import REPO
from windinet.wan.dit.model import autocast

TEMPORAL = 4  # frames per latent frame


class WanVAE:
    """`train`: 'none' (frozen), 'io' (encoder.conv_in and decoder.conv_out, the field interface), 'decoder'
    (decoder + post_quant_conv, latents unchanged) or 'all'. `channels` > 3: the VAE reads and writes that many
    channels (`widen_io`). `checkpointing`: recompute the blocks in backward (memory for time). `precision` 'mixed':
    bf16 autocast inside encode and decode when gradients flow, fp32 weights and outputs."""

    def __init__(
        self,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.float32,
        repo: str = REPO,
        train: str = "none",
        channels: int = 3,
        checkpointing: bool = False,
        precision: str = "fp32",
    ):
        # fp32 on purpose: the Wan pipeline keeps its VAE in fp32
        vae = AutoencoderKLWan.from_pretrained(repo, subfolder="vae", torch_dtype=dtype)
        if channels > 3:
            widen_io(vae, channels)
        self.vae = vae.to(device).eval().requires_grad_(False)
        io = (vae.encoder.conv_in, vae.decoder.conv_out)  # the field interface: its own lr in the VAE stage
        trainable = {"none": [], "io": io, "decoder": [vae.decoder, vae.post_quant_conv], "all": [vae]}[train]
        for module in trainable:
            module.requires_grad_(True)
        self.trainable_params = [p for p in vae.parameters() if p.requires_grad]
        self.io_params = [p for m in io for p in m.parameters() if p.requires_grad]
        if precision == "mixed":  # bf16 rounding of the output, amplified by the adapter's atanh, wrecks conv_out
            for conv in io:
                conv.forward = _fp32(conv.forward)
        if checkpointing:  # recompute each block in backward: a clip's graph is several times the memory
            for block in [*vae.encoder.down_blocks, *vae.decoder.up_blocks]:
                block.forward = _checkpointed(block.forward)
        self.dtype = dtype
        self.precision, self.device = precision, torch.device(device)
        shape = (1, -1, 1, 1, 1)
        self.latents_mean = torch.tensor(vae.config.latents_mean, device=device, dtype=dtype).view(shape)
        self.latents_std = torch.tensor(vae.config.latents_std, device=device, dtype=dtype).view(shape)

    def _autocast(self) -> torch.autocast:
        """bf16 only where gradients flow: evaluation stays fp32, comparable across runs."""
        return autocast(self.precision if torch.is_grad_enabled() else "fp32", self.device)

    @staticmethod
    def pad_frames(x: torch.Tensor) -> torch.Tensor:
        """Repeat the last frame until F = 4k + 1, the frame counts the causal encoder reads (1 + 4 + 4 ...)."""
        f = x.shape[2]
        target = math.ceil((f - 1) / TEMPORAL) * TEMPORAL + 1
        return x if target == f else torch.cat([x, x[:, :, -1:].expand(-1, -1, target - f, -1, -1)], dim=2)

    def encode(self, rgb: torch.Tensor) -> torch.Tensor:
        """Deterministic (posterior mean), normalised latents. Differentiable w.r.t. `rgb`."""
        with self._autocast():
            z = self.vae.encode(self.pad_frames(rgb).to(self.dtype)).latent_dist.mode()
        return (z.to(self.dtype) - self.latents_mean) / self.latents_std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        with self._autocast():
            rgb = self.vae.decode(z * self.latents_std + self.latents_mean).sample
        return rgb.to(self.dtype)  # fp32 for the adapter's atanh near +-1


def widen_io(vae: AutoencoderKLWan, channels: int) -> None:
    """Zero-initialised extra input slots of encoder.conv_in and output slots of decoder.conv_out: on the first three
    channels the widened VAE is the pretrained one, the new channels start ignored and decoded as zero."""
    for conv, dim in ((vae.encoder.conv_in, 1), (vae.decoder.conv_out, 0)):
        extra = channels - conv.weight.shape[dim]
        conv.weight = nn.Parameter(pad(conv.weight, [0, 0] * (conv.weight.ndim - 1 - dim) + [0, extra]))
        if dim == 0:
            conv.bias = nn.Parameter(pad(conv.bias, (0, extra)))
        conv.in_channels, conv.out_channels = conv.weight.shape[1], conv.weight.shape[0]


def _fp32(forward):
    """Run a module's forward in fp32 under an active autocast."""

    def run(*args, **kw):
        args = [a.float() if torch.is_tensor(a) else a for a in args]
        kw = {k: v.float() if torch.is_tensor(v) else v for k, v in kw.items()}
        with torch.autocast(args[0].device.type, enabled=False):
            return forward(*args, **kw)

    return run


def _checkpointed(block_forward):
    """Checkpoint a Wan block that reads and advances a per-pass frame cache: the forward pass runs on the real
    cache, the recompute in backward on a scratch copy of the entries this block consumed, so the caches of
    the whole pass are not kept alive until backward."""

    def forward(x, feat_cache=None, feat_idx=[0], **kw):  # noqa: B006 (the diffusers block signature)
        if feat_cache is None or not torch.is_grad_enabled():
            return block_forward(x, feat_cache, feat_idx, **kw)
        ctx = {"cache": feat_cache, "start": feat_idx[0]}

        def run(x):
            if "consumed" not in ctx:  # forward pass
                return block_forward(x, ctx["cache"], feat_idx, **kw)
            scratch = [None] * ctx["start"] + ctx["consumed"]  # recompute: the same entries at the same indices
            return block_forward(x, scratch, [ctx["start"]], **kw)

        before = list(feat_cache)
        out = checkpoint(run, x, use_reentrant=False)
        ctx["consumed"], ctx["cache"] = before[ctx["start"] : feat_idx[0]], None  # drop the live cache reference
        return out

    return forward
