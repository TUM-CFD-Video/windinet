"""Distribution statistics and matching losses on [B, C, *spatial] tensors.

Used twice: on the adapter's RGB output (C = 3, reference = natural images) and
on frozen-VAE latents (C = 16 for Wan 2.1, reference = encoded natural clips).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# [VERIFY] ImageNet per-channel RGB mean/std on [0, 1] pixels, here mapped to the
# (-1, 1) range the adapter emits: mean' = 2m - 1, std' = 2s.
IMAGENET_RGB_MEAN = torch.tensor([0.485, 0.456, 0.406]) * 2 - 1
IMAGENET_RGB_STD = torch.tensor([0.229, 0.224, 0.225]) * 2


def _flat(x: torch.Tensor) -> torch.Tensor:
    """[B, C, ...] -> [N, C]: every pixel/latent site is one observation."""
    return x.movedim(1, -1).reshape(-1, x.shape[1])


def channel_stats(x: torch.Tensor) -> dict[str, torch.Tensor]:
    """Per-channel mean and variance plus the full C x C covariance."""
    f = _flat(x).float()
    return {"mean": f.mean(0), "var": f.var(0), "cov": torch.cov(f.T)}


def moment_loss(x: torch.Tensor, mean_ref: torch.Tensor, var_ref: torch.Tensor) -> torch.Tensor:
    s = channel_stats(x)
    return (s["mean"] - mean_ref).square().mean() + (s["var"] - var_ref).square().mean()


def coral_loss(x: torch.Tensor, cov_ref: torch.Tensor) -> torch.Tensor:
    """CORAL (Sun & Saenko 2016): squared Frobenius distance of covariances, normalised by 4 C^2."""
    c = x.shape[1]
    return (channel_stats(x)["cov"] - cov_ref).square().sum() / (4 * c * c)


def _sqrtm(a: torch.Tensor) -> torch.Tensor:
    evals, evecs = torch.linalg.eigh(a)
    return (evecs * evals.clamp_min(0).sqrt()) @ evecs.T


def frechet_distance(mean: torch.Tensor, cov: torch.Tensor, mean_ref: torch.Tensor, cov_ref: torch.Tensor) -> torch.Tensor:
    """Squared Gaussian W2 between N(mean, cov) and N(mean_ref, cov_ref), the FID formula."""
    sr = _sqrtm(cov_ref)
    cross = _sqrtm(sr @ cov @ sr)
    return (mean - mean_ref).square().sum() + torch.trace(cov + cov_ref - 2 * cross)


def radial_log_psd(x: torch.Tensor, bins: int = 64) -> torch.Tensor:
    """Radially averaged log power spectrum over the last two dims, mean over everything else -> [bins]."""
    p = torch.fft.rfft2(x.float(), norm="ortho").abs().square()
    h, w = p.shape[-2:]
    fy = torch.fft.fftfreq(h, device=x.device)[:, None]
    fx = torch.fft.rfftfreq(2 * (w - 1), device=x.device)[None, :]
    r = ((fy * fy + fx * fx).sqrt() / 0.5 * (bins - 1)).round().long().clamp_max(bins - 1).flatten()
    p = p.reshape(-1, h * w).mean(0)
    power = torch.zeros(bins, device=x.device).index_add_(0, r, p)
    count = torch.zeros(bins, device=x.device).index_add_(0, r, torch.ones_like(p))
    return torch.log(power / count.clamp_min(1) + 1e-12)


def psd_loss(x: torch.Tensor, log_psd_ref: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(radial_log_psd(x, bins=log_psd_ref.numel()), log_psd_ref)
