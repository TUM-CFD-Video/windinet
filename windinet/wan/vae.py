"""Frozen Wan 2.1 VAE with normalised latents.

    rgb [B, 3, F, H, W] in (-1, 1), F = 4k + 1   --encode-->   z [B, 16, (F-1)/4 + 1, H/8, W/8]

`encode` returns latents already standardised with the checkpoint's own
``latents_mean`` / ``latents_std`` (the convention the Wan pipeline uses before
its transformer), so "natural video latents look like N(0, 1) per channel" is
what a reference set should confirm. `decode` undoes that.
"""

from __future__ import annotations

import torch
from diffusers import AutoencoderKLWan

REPO = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"  # same VAE as every Wan 2.1 variant
TEMPORAL, SPATIAL = 4, 8


class WanVAE:
    def __init__(self, device: torch.device | str = "cuda", dtype: torch.dtype = torch.float32, repo: str = REPO):
        # fp32 on purpose: the Wan pipeline keeps its VAE in fp32; 507 MB fits any GPU.
        self.vae = AutoencoderKLWan.from_pretrained(repo, subfolder="vae", torch_dtype=dtype).to(device).eval().requires_grad_(False)
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
        return (z - self.latents_mean) / self.latents_std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(z * self.latents_std + self.latents_mean).sample
