"""The VAE-stage round trip fields -> adapter -> Wan VAE -> adapter^-1 -> fields: encode, decode, both."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from windinet.field_adapter import GroupedAdapter

if TYPE_CHECKING:
    from windinet.wan.vae.model import WanVAE


def encode(adapter: GroupedAdapter, x: torch.Tensor, vaes: list[WanVAE], clip: int) -> tuple[torch.Tensor, list]:
    """Frames [N, C, H, W] -> latents [N / clip, G, 16, clip / 4 + 1, h, w] (each group's image through its encoder:
    one shared VAE, or one per group) and the RGB image per group."""
    rgbs = adapter(x)
    clips = [rgb.unflatten(0, (-1, clip)).movedim(1, 2) for rgb in rgbs]  # [N / clip, 3, clip, H, W]
    return torch.stack([vaes[g % len(vaes)].encode(c) for g, c in enumerate(clips)], 1), rgbs


def decode(adapter: GroupedAdapter, latents: torch.Tensor, vaes: list[WanVAE]) -> torch.Tensor:
    """Latents [B, G, 16, F', h, w] -> fields [B, F, C, H, W]: each group through its decoder, then adapter^-1."""
    rgbs = [vaes[g % len(vaes)].decode(latents[:, g]).movedim(2, 1).flatten(0, 1) for g in range(latents.shape[1])]
    return adapter.inverse(rgbs).unflatten(0, (latents.shape[0], -1))


def round_trip(
    adapter: GroupedAdapter, x: torch.Tensor, vaes: list[WanVAE], clip: int = 1
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """x_hat [N, C, H, W] and the RGB image per group; `x` goes through the VAEs as N / clip clips (no VAE: adapter
    only)."""
    if not vaes:
        rgbs = adapter(x)
        return adapter.inverse(rgbs), rgbs
    z, rgbs = encode(adapter, x, vaes, clip)
    return decode(adapter, z, vaes).flatten(0, 1), rgbs
