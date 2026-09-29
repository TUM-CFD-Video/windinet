"""Per-pixel field -> RGB adapter with an exact inverse.

    x [B, n, ...]  --PreNorm--> --Mix (n -> 3)--> --SoftClip--> rgb in (-1, 1)

PreNorm and SoftClip are exact bijections. Mix is exactly invertible for n <= 3
(its inverse is the pseudo-inverse, which is exact on the row space of W); for
n > 3 the round trip drops the null-space component, which the ablations showed
is not recoverable (docs/field_adapter/ablations.md, experiment 1). Tensors are
[B, C, *spatial]; every stage acts channel-wise, so the same module handles
frames [B, C, H, W] and clips [B, C, F, H, W].

Ablated and removed (commit 0a525c9 has them): per-field monotone splines and
the PCA/luma initialisation.
"""

from __future__ import annotations

import math

import torch
from torch import nn


def _per_channel(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Reshape a [C] parameter so it broadcasts over [B, C, *spatial]."""
    return t.view(1, -1, *([1] * (x.ndim - 2)))


class PreNorm(nn.Module):
    """Fixed: log for strictly positive fields, then z-score with dataset statistics.

    `lo` / `hi` (same space as mean/std, i.e. log space for log fields) bound the inverse to the
    dataset's observed range: a decoder value in the tanh tail would otherwise blow up through
    atanh -> exp and dominate any physical-unit metric.
    """

    def __init__(
        self,
        mean: list[float],
        std: list[float],
        log_channels: list[bool],
        lo: list[float] | None = None,
        hi: list[float] | None = None,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean))
        self.register_buffer("std", torch.tensor(std))
        self.register_buffer("log_mask", torch.tensor(log_channels))
        self.register_buffer(
            "lo", torch.tensor(lo if lo is not None else [-float("inf")] * len(mean)), persistent=False
        )
        self.register_buffer("hi", torch.tensor(hi if hi is not None else [float("inf")] * len(mean)), persistent=False)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.where(_per_channel(self.log_mask, x), torch.log(x.clamp_min(self.eps)), x)
        return (x - _per_channel(self.mean, x)) / _per_channel(self.std, x)

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        x = z * _per_channel(self.std, z) + _per_channel(self.mean, z)
        x = torch.maximum(torch.minimum(x, _per_channel(self.hi, x)), _per_channel(self.lo, x))
        return torch.where(_per_channel(self.log_mask, x), x.exp(), x)


class Mix(nn.Module):
    """Linear channel mix n -> 3, initialised as identity (field i -> colour i). Inverse is the pseudo-inverse."""

    def __init__(self, n_in: int, n_out: int = 3):
        super().__init__()
        self.weight = nn.Parameter(torch.eye(n_out, n_in))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("oi,bi...->bo...", self.weight, x)

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        return torch.einsum("io,bo...->bi...", torch.linalg.pinv(self.weight), y)


class SoftClip(nn.Module):
    """z = tanh(y / s) with one learnable scale s; maps to the (-1, 1) range video VAEs expect."""

    def __init__(self, target_quantile: float = 0.98, target_value: float = 0.9):
        super().__init__()
        self.log_scale = nn.Parameter(torch.zeros(()))
        self.q, self.v = target_quantile, target_value

    @torch.no_grad()
    def init_from_data(self, y: torch.Tensor) -> None:
        """Pick s so that `target_quantile` of |y| lands inside +-`target_value`."""
        sample = y.abs().flatten()[torch.randperm(y.numel(), device=y.device)[:1_000_000]]
        self.log_scale.fill_(math.log(torch.quantile(sample, self.q).item() / math.atanh(self.v)))

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return torch.tanh(y / self.log_scale.exp())

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        return torch.atanh(z.clamp(-1 + 1e-6, 1 - 1e-6)) * self.log_scale.exp()


class FieldAdapter(nn.Module):
    """The full chain; ``forward`` fields -> rgb, ``inverse`` rgb -> fields."""

    def __init__(self, mean: list[float], std: list[float], log_channels: list[bool], lo=None, hi=None):
        super().__init__()
        self.prenorm = PreNorm(mean, std, log_channels, lo, hi)
        self.mix = Mix(len(mean), 3)
        self.clip = SoftClip()
        self.stages = [self.prenorm, self.mix, self.clip]

    @torch.no_grad()
    def init_from_data(self, x: torch.Tensor) -> None:
        self.clip.init_from_data(self.mix(self.prenorm(x)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for stage in self.stages:
            x = stage(x)
        return x

    def inverse(self, rgb: torch.Tensor) -> torch.Tensor:
        for stage in reversed(self.stages):
            rgb = stage.inverse(rgb)
        return rgb
