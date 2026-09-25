"""Per-pixel field -> RGB adapter built from (pseudo-)invertible stages.

    x [B, n, ...]  --PreNorm--> --MonotoneWarp--> --Mix (n->3)--> --SoftClip--> rgb in (-1, 1)

Every stage has ``forward`` and ``inverse``. PreNorm, MonotoneWarp and SoftClip
are exact bijections; Mix is exactly invertible only for n == 3, for n > 3 its
inverse is the Moore-Penrose pseudo-inverse, so the round trip drops the
component of the field vector in Mix's null space -- that is the whole
information loss of the adapter, and training moves it to where it hurts
least. Tensors are [B, C, *spatial]; every stage acts channel-wise, so the
same module handles single frames [B, C, H, W] and clips [B, C, F, H, W].
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _per_channel(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Reshape a [C] or [C, K] parameter so it broadcasts over [B, C, *spatial]."""
    return t.view(1, -1, *([1] * (x.ndim - 2)), *t.shape[1:])


class Identity(nn.Identity):
    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return x


class PreNorm(nn.Module):
    """Fixed: log for strictly positive fields, then z-score with dataset statistics.

    `lo` / `hi` (same space as mean/std, i.e. log space for log fields) bound the inverse to the
    dataset's observed range: a decoder value in the tanh tail would otherwise blow up through
    atanh -> exp and dominate any physical-unit metric.
    """

    def __init__(self, mean: list[float], std: list[float], log_channels: list[bool],
                 lo: list[float] | None = None, hi: list[float] | None = None, eps: float = 1e-6):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean))
        self.register_buffer("std", torch.tensor(std))
        self.register_buffer("log_mask", torch.tensor(log_channels))
        self.register_buffer("lo", torch.tensor(lo if lo is not None else [-float("inf")] * len(mean)), persistent=False)
        self.register_buffer("hi", torch.tensor(hi if hi is not None else [float("inf")] * len(mean)), persistent=False)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.where(_per_channel(self.log_mask, x), torch.log(x.clamp_min(self.eps)), x)
        return (x - _per_channel(self.mean, x)) / _per_channel(self.std, x)

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        x = z * _per_channel(self.std, z) + _per_channel(self.mean, z)
        x = torch.maximum(torch.minimum(x, _per_channel(self.hi, x)), _per_channel(self.lo, x))
        return torch.where(_per_channel(self.log_mask, x), x.exp(), x)


class MonotoneWarp(nn.Module):
    """Per-channel monotone rational-quadratic spline (Durkan et al. 2019), identity at init.

    K bins on [-bound, bound], linear identity tails outside. Widths/heights are
    softmax(logits) (zeros -> uniform), knot slopes are softplus(raw + c) with c
    chosen so raw = 0 gives slope 1 -- hence zero parameters == identity.
    ``slope_penalty`` keeps knots from collapsing (log-slope -> -inf).
    """

    def __init__(self, channels: int, bins: int = 8, bound: float = 5.0, min_bin: float = 1e-3, min_slope: float = 1e-3):
        super().__init__()
        self.bound, self.min_bin, self.min_slope = bound, min_bin, min_slope
        self.w_logits = nn.Parameter(torch.zeros(channels, bins))
        self.h_logits = nn.Parameter(torch.zeros(channels, bins))
        self.s_raw = nn.Parameter(torch.zeros(channels, bins - 1))
        self._slope_offset = math.log(math.expm1(1.0 - min_slope))  # min_slope + softplus(offset) == 1 exactly

    def _knots(self):
        """Per channel: bin widths, bin heights, knot x positions, knot y positions, knot slopes."""
        n_bins = self.w_logits.shape[1]
        widths = (self.min_bin + (1 - self.min_bin * n_bins) * F.softmax(self.w_logits, -1)) * 2 * self.bound
        heights = (self.min_bin + (1 - self.min_bin * n_bins) * F.softmax(self.h_logits, -1)) * 2 * self.bound
        knots_x = F.pad(torch.cumsum(widths, -1), (1, 0)) - self.bound
        knots_y = F.pad(torch.cumsum(heights, -1), (1, 0)) - self.bound
        slopes = self.min_slope + F.softplus(self.s_raw + self._slope_offset)
        slopes = F.pad(slopes, (1, 1), value=1.0)  # slope 1 at both ends, continuous with the identity tails
        return widths, heights, knots_x, knots_y, slopes

    def _spline(self, x: torch.Tensor, inverse: bool) -> torch.Tensor:
        widths, heights, knots_x, knots_y, slopes = self._knots()
        inside = (x > -self.bound) & (x < self.bound)
        x_in = x.clamp(-self.bound + 1e-6, self.bound - 1e-6)

        # Which bin each value falls in (per channel), then gather that bin's parameters.
        knots = _per_channel(knots_y if inverse else knots_x, x).expand(*x.shape, -1)
        bin_idx = (torch.searchsorted(knots.contiguous(), x_in.unsqueeze(-1).contiguous()) - 1).clamp(0, widths.shape[-1] - 1)

        def pick(param):  # [C, K] -> the value of this channel's bin at every pixel
            return torch.gather(_per_channel(param, x).expand(*x.shape, -1), -1, bin_idx).squeeze(-1)

        w, h = pick(widths), pick(heights)
        x0, y0 = pick(knots_x), pick(knots_y)
        d0, d1 = pick(slopes[:, :-1]), pick(slopes[:, 1:])
        s = h / w  # mean slope of the bin

        if not inverse:
            t = (x_in - x0) / w
            out = y0 + h * (s * t * t + d0 * t * (1 - t)) / (s + (d0 + d1 - 2 * s) * t * (1 - t))
        else:  # solve the quadratic in t for a given y
            y = x_in - y0
            a = h * (s - d0) + y * (d0 + d1 - 2 * s)
            b = h * d0 - y * (d0 + d1 - 2 * s)
            c = -s * y
            t = 2 * c / (-b - torch.sqrt(b * b - 4 * a * c))
            out = x0 + t * w
        return torch.where(inside, out, x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._spline(x, inverse=False)

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        return self._spline(y, inverse=True)

    def slope_penalty(self) -> torch.Tensor:
        return torch.log(self.min_slope + F.softplus(self.s_raw + self._slope_offset)).square().mean()


class Mix(nn.Module):
    """Linear channel mix, n -> 3. Inverse is the pseudo-inverse (exact iff n == 3 and W is non-singular).

    ``init_from_data`` sets W = Qᵀ·D·P: P = top-3 principal axes of the (warped) fields,
    Q sends PC1 to the RGB luma axis (1,1,1)/√3 and PC2/PC3 to two opponent axes,
    D rescales the three PC variances to natural-image RGB PCA eigenvalues.
    """

    # [VERIFY] AlexNet "fancy PCA" eigenvalues of ImageNet RGB pixels in [0,1]
    # (Krizhevsky et al. 2012); only the ratio is used here.
    RGB_PCA_EIGVALS = (0.2175, 0.0188, 0.0045)
    RGB_AXES = torch.tensor([[1, 1, 1], [1, -1, 0], [1, 1, -2]], dtype=torch.float32)

    def __init__(self, n_in: int, n_out: int = 3):
        super().__init__()
        self.weight = nn.Parameter(torch.eye(n_out, n_in))

    @torch.no_grad()
    def init_from_data(self, x: torch.Tensor) -> None:
        flat = x.movedim(1, -1).reshape(-1, x.shape[1])
        k = min(3, x.shape[1])  # fewer than 3 fields: use as many principal axes as there are fields
        evals, evecs = torch.linalg.eigh(torch.cov(flat.T))  # ascending
        P = evecs[:, -k:].flip(-1).T  # [k, n]: top-k axes as rows
        D = torch.diag(torch.tensor(self.RGB_PCA_EIGVALS[:k], device=x.device).sqrt() / evals[-k:].flip(0).clamp_min(1e-8).sqrt())
        Q = F.normalize(self.RGB_AXES[:k].to(x), dim=1)
        self.weight.copy_(Q.T @ D @ P)

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

    def __init__(self, mean: list[float], std: list[float], log_channels: list[bool], lo=None, hi=None, bins: int = 8, warp: bool = True):
        super().__init__()
        n = len(mean)
        self.prenorm = PreNorm(mean, std, log_channels, lo, hi)
        self.warp = MonotoneWarp(n, bins) if warp else Identity()
        self.mix = Mix(n, 3)
        self.clip = SoftClip()
        self.stages = [self.prenorm, self.warp, self.mix, self.clip]

    @torch.no_grad()
    def init_from_data(self, x: torch.Tensor, mix: str = "pca") -> None:
        """Data-driven init of Mix and SoftClip (range); warp stays identity.

        mix='pca': principal axes -> luma/opponent RGB axes. mix='naive': the first three
        z-scored fields become R, G, B as they are and the fourth is dropped (zero column).
        """
        z = self.warp(self.prenorm(x))
        if mix == "pca":
            self.mix.init_from_data(z)
        else:
            self.mix.weight.copy_(torch.eye(3, z.shape[1], device=z.device))
        self.clip.init_from_data(self.mix(z))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for stage in self.stages:
            x = stage(x)
        return x

    def inverse(self, rgb: torch.Tensor) -> torch.Tensor:
        for stage in reversed(self.stages):
            rgb = stage.inverse(rgb)
        return rgb

    def normalized(self, x: torch.Tensor) -> torch.Tensor:
        """Fields in z-scored space -- the space reconstruction losses are measured in."""
        return self.prenorm(x)
