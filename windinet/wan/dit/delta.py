"""Delta targets (latents.delta): the DiT generates every latent frame after the first as its change from the frame
before, divided by the train std of such changes; the context frame stays a state. Wan's latent 0 holds one frame and
every later latent four, so the change into latent 1 gets its own scale."""

from __future__ import annotations

import torch
import torch.distributed as dist


def scales(loader, device: torch.device) -> torch.Tensor:
    """[2, C]: per latent channel the std of the change into latent 1 and of the changes between latents 2.., over
    the latents of `loader` (latents [B, G, 16, F', h, w]) on every rank."""
    sums = 0  # [2, 3, C]: (into latent 1, later) x (count, sum, sum of squares) x channels
    for latents, _ in loader:
        change = latents.flatten(1, 2).to(device, torch.float64).diff(dim=2)  # [B, C, F' - 1, h, w]
        parts = change.split([1, change.shape[2] - 1], 2)
        dims = (0, 2, 3, 4)
        stats = [torch.stack([d.new_ones(d.shape).sum(dims), d.sum(dims), d.square().sum(dims)]) for d in parts]
        sums = sums + torch.stack(stats)
    if dist.is_initialized():
        dist.all_reduce(sums)
    n, total, squares = sums.unbind(1)
    return (squares / n - (total / n) ** 2).sqrt().float()


def to_delta(z: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """States [B, C, F', h, w] -> latent 0 as is, every later latent as its scaled change from the one before."""
    change = z.diff(dim=2)
    first, later = change[:, :, :1] / scale[0, :, None, None, None], change[:, :, 1:] / scale[1, :, None, None, None]
    return torch.cat([z[:, :, :1], first, later], 2)


def from_delta(d: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Inverse of to_delta: the states, summed up from latent 0."""
    change = torch.cat([d[:, :, 1:2] * scale[0, :, None, None, None], d[:, :, 2:] * scale[1, :, None, None, None]], 2)
    return torch.cat([d[:, :, :1], d[:, :, :1] + change.cumsum(2)], 2)
