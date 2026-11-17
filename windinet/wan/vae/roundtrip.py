"""The VAE-stage round trip fields -> adapter -> Wan VAE -> adapter^-1 -> fields: encode, decode, both.

With change statistics (adapter.changes) each clip's first frame passes the adapter as a state and every later frame
as its change from the one before; the way back sums the changes onto the first frame, so callers see states."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from windinet.field_adapter import GroupedAdapter

if TYPE_CHECKING:
    from windinet.wan.vae.model import WanVAE


def to_images(adapter: GroupedAdapter, x: torch.Tensor, clips: int) -> list[torch.Tensor]:
    """Frames [N, C, H, W] of `clips` clips -> the image per group the VAE sees, [N, channels, H, W]."""
    if adapter.changes is None:
        return adapter(x)
    x = x.unflatten(0, (clips, -1))
    first, later = adapter(x[:, 0]), adapter.changes(x.diff(dim=1).flatten(0, 1))
    later = [c.unflatten(0, (clips, -1)) for c in later]
    return [torch.cat([f[:, None], c], 1).flatten(0, 1) for f, c in zip(first, later, strict=True)]


def to_fields(adapter: GroupedAdapter, rgbs: list[torch.Tensor], clips: int) -> torch.Tensor:
    """Inverse of `to_images`: fields [clips, F, C, H, W]."""
    if adapter.changes is None:
        return adapter.inverse(rgbs).unflatten(0, (clips, -1))
    rgbs = [r.unflatten(0, (clips, -1)) for r in rgbs]
    first = adapter.inverse([r[:, 0] for r in rgbs])
    later = adapter.changes.inverse([r[:, 1:].flatten(0, 1) for r in rgbs]).unflatten(0, (clips, -1))
    return torch.cat([first[:, None], later], 1).cumsum(1)


def encode(
    adapter: GroupedAdapter, x: torch.Tensor, vaes: list[WanVAE], clip: int, sample: bool = False
) -> tuple[torch.Tensor, list, torch.Tensor]:
    """Frames [N, C, H, W] -> latents [N / clip, G, 16, clip / 4 + 1, h, w] (each group's image through its encoder:
    one shared VAE, or one per group; posterior mean unless `sample`), the image per group and the KL of the
    posteriors, mean over groups (WanVAE.posterior)."""
    rgbs = to_images(adapter, x, len(x) // clip)
    clips = [rgb.unflatten(0, (-1, clip)).movedim(1, 2) for rgb in rgbs]  # [N / clip, 3, clip, H, W]
    zs, kls = zip(*(vaes[g % len(vaes)].posterior(c, sample) for g, c in enumerate(clips)), strict=True)
    return torch.stack(zs, 1), rgbs, torch.stack(kls).mean()


def decode(adapter: GroupedAdapter, latents: torch.Tensor, vaes: list[WanVAE]) -> torch.Tensor:
    """Latents [B, G, 16, F', h, w] -> fields [B, F, C, H, W]: each group through its decoder, then adapter^-1."""
    rgbs = [vaes[g % len(vaes)].decode(latents[:, g]).movedim(2, 1).flatten(0, 1) for g in range(latents.shape[1])]
    return to_fields(adapter, rgbs, latents.shape[0])


def round_trip(
    adapter: GroupedAdapter, x: torch.Tensor, vaes: list[WanVAE], clip: int = 1
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """x_hat [N, C, H, W] and the image per group; `x` goes through the VAEs as N / clip clips (no VAE: adapter
    only)."""
    if not vaes:
        rgbs = to_images(adapter, x, len(x) // clip)
        return to_fields(adapter, rgbs, len(x) // clip).flatten(0, 1), rgbs
    z, rgbs, _ = encode(adapter, x, vaes, clip)
    return decode(adapter, z, vaes).flatten(0, 1), rgbs
