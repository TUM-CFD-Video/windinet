"""Wan 2.1 VAE with normalised latents, frozen unless `train` says otherwise.

    rgb [B, 3, F, H, W] in (-1, 1), F = 4k + 1   --encode-->   z [B, 16, (F-1)/4 + 1, H/8, W/8]

`encode` standardises with the checkpoint's own latents_mean / latents_std (what the Wan pipeline feeds its
transformer, so natural-video latents look like N(0, 1) per channel); `decode` undoes that.
"""

from __future__ import annotations

import math

import torch
from diffusers import AutoencoderKLWan
from torch.utils.checkpoint import checkpoint

REPO = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"  # same VAE as every Wan 2.1 variant
TEMPORAL = 4  # frames per latent frame


class WanVAE:
    """`train`: 'none' (frozen), 'decoder' (decoder + post_quant_conv, latents unchanged) or 'all'."""

    def __init__(
        self,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.float32,
        repo: str = REPO,
        train: str = "none",
    ):
        # fp32 on purpose: the Wan pipeline keeps its VAE in fp32
        vae = AutoencoderKLWan.from_pretrained(repo, subfolder="vae", torch_dtype=dtype)
        self.vae = vae.to(device).eval().requires_grad_(False)
        trainable = {"none": [], "decoder": [vae.decoder, vae.post_quant_conv], "all": [vae]}[train]
        for module in trainable:
            module.requires_grad_(True)
        self.trainable_params = [p for p in vae.parameters() if p.requires_grad]
        if trainable:  # recompute each block in backward: the full graph of a clip needs several times the memory
            for block in [*vae.encoder.down_blocks, *vae.decoder.up_blocks]:
                block.forward = _checkpointed(block.forward)
        self.dtype = dtype
        shape = (1, -1, 1, 1, 1)
        self.latents_mean = torch.tensor(vae.config.latents_mean, device=device, dtype=dtype).view(shape)
        self.latents_std = torch.tensor(vae.config.latents_std, device=device, dtype=dtype).view(shape)

    @staticmethod
    def pad_frames(x: torch.Tensor) -> torch.Tensor:
        """Repeat the last frame until F = 4k + 1, the frame counts the causal encoder reads (1 + 4 + 4 ...)."""
        f = x.shape[2]
        target = math.ceil((f - 1) / TEMPORAL) * TEMPORAL + 1
        return x if target == f else torch.cat([x, x[:, :, -1:].expand(-1, -1, target - f, -1, -1)], dim=2)

    def encode(self, rgb: torch.Tensor) -> torch.Tensor:
        """Deterministic (posterior mean), normalised latents. Differentiable w.r.t. `rgb`."""
        z = self.vae.encode(self.pad_frames(rgb).to(self.dtype)).latent_dist.mode()
        return (z - self.latents_mean) / self.latents_std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(z * self.latents_std + self.latents_mean).sample


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
