"""Distribution statistics on [B, C, *spatial] tensors and the Fréchet distance between two of them.

Used as a diagnostic on the adapter's RGB output (C = 3) and on VAE latents (C = 16 for
Wan 2.1) against a natural-video reference. Using these as training losses was ablated
and did not help (docs/field_adapter/ablations.md, experiment 5; code in commit 0a525c9).
"""

from __future__ import annotations

import torch


def _flat(x: torch.Tensor) -> torch.Tensor:
    """[B, C, ...] -> [N, C]: every pixel/latent site is one observation."""
    return x.movedim(1, -1).reshape(-1, x.shape[1])


def channel_stats(x: torch.Tensor) -> dict[str, torch.Tensor]:
    """Per-channel mean and variance plus the full C x C covariance."""
    f = _flat(x).float()
    return {"mean": f.mean(0), "var": f.var(0), "cov": torch.cov(f.T)}


def _sqrtm(a: torch.Tensor) -> torch.Tensor:
    evals, evecs = torch.linalg.eigh(a)
    return (evecs * evals.clamp_min(0).sqrt()) @ evecs.T


def frechet_distance(mean: torch.Tensor, cov: torch.Tensor, mean_ref: torch.Tensor, cov_ref: torch.Tensor) -> torch.Tensor:
    """Squared Gaussian W2 between N(mean, cov) and N(mean_ref, cov_ref), the FID formula."""
    sr = _sqrtm(cov_ref)
    cross = _sqrtm(sr @ cov @ sr)
    return (mean - mean_ref).square().sum() + torch.trace(cov + cov_ref - 2 * cross)
