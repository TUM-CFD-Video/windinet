"""Gaussian statistics of [B, C, *spatial] tensors and the Fréchet distance between two of them.

Diagnostic only: how far VAE latents (C = 16 for Wan 2.1) are from a natural-video reference.
Using it as a training loss was ablated and did not help (docs/field_adapter/ablations.md,
experiment 5; code in commit 0a525c9).
"""

from __future__ import annotations

import torch


def channel_stats(x: torch.Tensor) -> dict[str, torch.Tensor]:
    """Per-channel mean and C x C covariance over every pixel/latent site."""
    f = x.movedim(1, -1).reshape(-1, x.shape[1]).float()
    return {"mean": f.mean(0), "cov": torch.cov(f.T)}


def _sqrtm(a: torch.Tensor) -> torch.Tensor:
    evals, evecs = torch.linalg.eigh(a)
    return (evecs * evals.clamp_min(0).sqrt()) @ evecs.T


def frechet_distance(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> torch.Tensor:
    """Squared Gaussian W2 between N(a) and N(b), the FID formula."""
    sr = _sqrtm(b["cov"])
    return (a["mean"] - b["mean"]).square().sum() + torch.trace(a["cov"] + b["cov"] - 2 * _sqrtm(sr @ a["cov"] @ sr))
