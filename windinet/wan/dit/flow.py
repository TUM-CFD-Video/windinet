"""Rectified flow for the DiT: the noising of a clean latent, the velocity target, the sigma (timestep) draw and its
layout over the latent frames (clean context, Diffusion Forcing's noisy history)."""

from __future__ import annotations

import torch


def draw_sigmas(n: int, device: torch.device, mean: float = 0.0, std: float = 1.0, shift: float = 1.0) -> torch.Tensor:
    """Logit-normal sigmas in (0, 1), then the time shift Wan's scheduler applies: s σ / (1 + (s - 1) σ)."""
    sigma = torch.sigmoid(torch.randn(n, device=device) * std + mean)
    return shift * sigma / (1 + (shift - 1) * sigma)


def noised(x0: torch.Tensor, noise: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """x_σ = (1 - σ) x0 + σ noise, σ per sample [B] or per latent frame [B, F'] of x0 [B, C, F', h, w]."""
    s = sigma.reshape(len(sigma), 1, -1, 1, 1)
    return (1 - s) * x0 + s * noise


def clean_estimate(x_sigma: torch.Tensor, v: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """The clean latent a velocity prediction implies: x0 = x_σ - σ v."""
    return x_sigma - sigma.reshape(len(sigma), 1, -1, 1, 1) * v


def gated(sigma: torch.Tensor, gate: float, n: int) -> torch.Tensor:
    """Indices of the (at most) n lowest-sigma samples with sigma <= gate; sigma [B, F'], the last frame's is the
    sample's."""
    low = sigma[:, -1].argsort()[:n]
    return low[sigma[low, -1] <= gate]


def frame_sigmas(draws: torch.Tensor, context: int, history_noise: float | None) -> tuple[torch.Tensor, torch.Tensor]:
    """From per-frame draws [B, F']: the sigma of every frame and the frames in the loss. Plain: one sigma per sample
    (its first draw), the `context` frames clean. With `history_noise` (Diffusion Forcing): every target frame its own
    draw, a random history of at least `context` frames at one level in [0, history_noise] (0 for a fifth of the
    samples, frame 0 always clean) and out of the loss."""
    (b, f), dev = draws.shape, draws.device
    frames = torch.arange(f, device=dev)
    if history_noise is None:
        return torch.where(frames < context, 0.0, draws[:, :1]), (frames >= context).expand(b, f)
    history = torch.randint(context, f, (b, 1), device=dev)  # frames before this one are history
    level = torch.rand(b, 1, device=dev) * history_noise * (torch.rand(b, 1, device=dev) > 0.2)
    target = frames >= history
    return torch.where(target, draws, torch.where(frames == 0, 0.0, level)), target


def velocity_target(x0: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    return noise - x0
