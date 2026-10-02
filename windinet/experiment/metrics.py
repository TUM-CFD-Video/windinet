"""Reconstruction metric of the thesis baseline: variance-normalised RMSE per field."""

from __future__ import annotations

import torch


def vrmse(x_hat: torch.Tensor, x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Per field: sqrt(mean(err^2) / var(x)), population variance over frames and pixels (Li 2026, eq. 5.11).

    `x`, `x_hat`: [N, C, H, W] frames of ONE simulation, so the variance is that simulation's own. Scale-free:
    physical units and z-scored fields give the same number. Returns [C]."""
    dims = (0, 2, 3)
    return ((x_hat - x).square().mean(dims) / (x.var(dims, unbiased=False) + eps)).sqrt()


def clipped(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, k: float = 5.0) -> torch.Tensor:
    """The LTX baseline's normalised space (Li 2026, eq. 3.14): (x - mean) / (k std) clipped to [-1, 1].

    `mean`, `std`: [C] in physical units. VRMSE here ignores whatever a pipeline loses beyond k sigma."""
    shape = (1, -1) + (1,) * (x.ndim - 2)
    return ((x - mean.view(shape)) / (k * std.view(shape))).clamp(-1, 1)
