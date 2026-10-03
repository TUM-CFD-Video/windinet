"""Rectified flow for the DiT: the noising of a clean latent, the velocity target, and the sigma (timestep) draw."""

from __future__ import annotations

import torch


def draw_sigmas(n: int, device: torch.device, mean: float = 0.0, std: float = 1.0, shift: float = 1.0) -> torch.Tensor:
    """Logit-normal sigmas in (0, 1), then the time shift Wan's scheduler applies: s σ / (1 + (s - 1) σ)."""
    sigma = torch.sigmoid(torch.randn(n, device=device) * std + mean)
    return shift * sigma / (1 + (shift - 1) * sigma)


def noised(x0: torch.Tensor, noise: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """x_σ = (1 - σ) x0 + σ noise, one σ per sample."""
    s = sigma.view(-1, *[1] * (x0.ndim - 1))
    return (1 - s) * x0 + s * noise


def velocity_target(x0: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    return noise - x0
