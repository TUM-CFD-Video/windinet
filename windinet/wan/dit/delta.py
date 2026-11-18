"""Delta targets (latents.delta): the DiT generates every latent frame after the first as its change from the frame
before, standardised with the train mean and std of such changes; latent 0 stays a state. Wan's latent 0
holds one frame and every later latent four, so the change into latent 1 gets its own statistics."""

from __future__ import annotations

import torch
import torch.distributed as dist


def change_stats(loader, device: torch.device) -> torch.Tensor:
    """[2, 2, C]: (mean, std) x (change into latent 1, changes between latents 2..) per latent channel, over every
    latent the training `loader` draws (latents [B, G, 16, F', h, w]) on every rank."""
    sums = 0  # [3, 2, C]: (count, sum, sum of squares) x (into latent 1, later) x channels
    for latents, *_ in loader:
        change = latents.flatten(1, 2).to(device, torch.float64).diff(dim=2)  # [B, C, F' - 1, h, w]
        sums = sums + torch.stack([moments(d) for d in change.split([1, change.shape[2] - 1], 2)], 1)
    if dist.is_initialized():
        dist.all_reduce(sums)
    n, total, squares = sums
    mean = total / n
    return torch.stack([mean, (squares / n - mean.square()).sqrt()]).float()


def moments(d: torch.Tensor) -> torch.Tensor:
    """[3, C]: count, sum and sum of squares per channel of `d` [B, C, ...]."""
    dims = [0, *range(2, d.dim())]
    total = d.sum(dims)
    return torch.stack([torch.full_like(total, d.numel() / d.shape[1]), total, d.square().sum(dims)])


def per_change(stats: torch.Tensor, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean and std [C, n, 1, 1] of each of n changes: the first one's, then the later ones'."""
    s = stats[:, [0] + [1] * (n - 1)].transpose(1, 2)[..., None, None]
    return s[0], s[1]


def to_delta(z: torch.Tensor, stats: torch.Tensor) -> torch.Tensor:
    """States [B, C, F', h, w] -> latent 0 as is, every later latent as its standardised change from the one before."""
    mean, std = per_change(stats, z.shape[2] - 1)
    return torch.cat([z[:, :, :1], (z.diff(dim=2) - mean) / std], 2)


def from_delta(d: torch.Tensor, stats: torch.Tensor) -> torch.Tensor:
    """Inverse of to_delta: the states, summed up from latent 0."""
    mean, std = per_change(stats, d.shape[2] - 1)
    return torch.cat([d[:, :, :1], d[:, :, :1] + (d[:, :, 1:] * std + mean).cumsum(2)], 2)


def load_stats(path, device: torch.device) -> torch.Tensor:
    """delta.pt of a run; files before the mean ([2, C], std only) get mean 0."""
    stats = torch.load(path, map_location=device)
    return torch.stack([torch.zeros_like(stats), stats]) if stats.dim() == 2 else stats
