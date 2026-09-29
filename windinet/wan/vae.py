"""Wan 2.1 VAE with normalised latents, frozen unless `train` says otherwise.

    rgb [B, 3, F, H, W] in (-1, 1), F = 4k + 1   --encode-->   z [B, 16, (F-1)/4 + 1, H/8, W/8]

`encode` returns latents already standardised with the checkpoint's own
``latents_mean`` / ``latents_std`` (the convention the Wan pipeline uses before
its transformer), so "natural video latents look like N(0, 1) per channel" is
what a reference set should confirm. `decode` undoes that.
"""

from __future__ import annotations

import torch
from diffusers import AutoencoderKLWan
from torch.utils.checkpoint import checkpoint

REPO = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"  # same VAE as every Wan 2.1 variant
TEMPORAL, SPATIAL = 4, 8


class WanVAE:
    def __init__(self, device: torch.device | str = "cuda", dtype: torch.dtype = torch.float32, repo: str = REPO,
                 train: str = "none"):
        """`train`: 'none' (frozen), 'decoder' (decoder + post_quant_conv, latents unchanged) or 'all'."""
        # fp32 on purpose: the Wan pipeline keeps its VAE in fp32; 507 MB fits any GPU.
        self.vae = AutoencoderKLWan.from_pretrained(repo, subfolder="vae", torch_dtype=dtype).to(device).eval().requires_grad_(False)
        trainable = {"none": [], "decoder": [self.vae.decoder, self.vae.post_quant_conv], "all": [self.vae]}[train]
        for module in trainable:
            module.requires_grad_(True)
        self.trainable_params = [p for p in self.vae.parameters() if p.requires_grad]
        if trainable:  # recompute each block's activations in backward: a 5-frame clip's graph does not fit 11 GB
            for block in [*self.vae.encoder.down_blocks, *self.vae.decoder.up_blocks]:
                block.forward = _checkpointed(block.forward)
        self.device, self.dtype = torch.device(device), dtype
        shape = (1, -1, 1, 1, 1)
        self.latents_mean = torch.tensor(self.vae.config.latents_mean, device=device, dtype=dtype).view(shape)
        self.latents_std = torch.tensor(self.vae.config.latents_std, device=device, dtype=dtype).view(shape)
        self.latent_channels = self.vae.config.z_dim

    @staticmethod
    def pad_frames(x: torch.Tensor) -> torch.Tensor:
        """Repeat the last frame until F = 4k + 1."""
        f = x.shape[2]
        target = ((f - 2) // TEMPORAL + 1) * TEMPORAL + 1 if f > 1 else 1
        return x if target == f else torch.cat([x, x[:, :, -1:].expand(-1, -1, target - f, -1, -1)], dim=2)

    def encode(self, rgb: torch.Tensor) -> torch.Tensor:
        """Deterministic (posterior mean), normalised latents. Differentiable w.r.t. `rgb`."""
        z = self.vae.encode(self.pad_frames(rgb).to(self.dtype)).latent_dist.mode()
        self.vae.clear_cache()  # drop the frame caches now, not at the next call: they are big and useless after the pass
        return (z - self.latents_mean) / self.latents_std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = self.vae.decode(z * self.latents_std + self.latents_mean).sample
        self.vae.clear_cache()
        return x


def _checkpointed(block_forward):
    """Wan blocks read and advance a per-chunk frame cache. The forward pass runs on the real cache; the
    recompute in backward gets a scratch copy of just the entries this block read, so the caches of the
    whole pass are not kept alive until backward."""
    def forward(x, feat_cache=None, feat_idx=[0], **kw):
        if feat_cache is None or not torch.is_grad_enabled():
            return block_forward(x, feat_cache, feat_idx, **kw)
        state = {"cache": feat_cache, "idx": feat_idx[0], "n": len(feat_cache)}

        def run(x):
            if "read" not in state:
                return block_forward(x, state["cache"], feat_idx, **kw)
            scratch = [None] * state["n"]
            scratch[state["idx"]:state["idx"] + len(state["read"])] = state["read"]
            return block_forward(x, scratch, [state["idx"]], **kw)

        snapshot = list(feat_cache)
        out = checkpoint(run, x, use_reentrant=False)
        state["read"], state["cache"] = snapshot[state["idx"]:feat_idx[0]], None
        return out
    return forward
