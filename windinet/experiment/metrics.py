"""Metrics of one simulation's trajectory [N, C, H, W] against its prediction, per field: VRMSE (per trajectory and
per frame), relative gradient error (H1), The Well's binned spectral error and SSIM; aggregated over simulations."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from windinet.experiment.distributed import all_sum


def vrmse(x_hat: torch.Tensor, x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Per field: sqrt(mean(err^2) / var(x)), population variance over frames and pixels (Li 2026, eq. 5.11).

    `x`, `x_hat`: [N, C, H, W] frames of ONE simulation, so the variance is that simulation's own. Scale-free:
    physical units and z-scored fields give the same number. Returns [C]."""
    dims = (0, 2, 3)
    return ((x_hat - x).square().mean(dims) / (x.var(dims, unbiased=False) + eps)).sqrt()


def h1(x_hat: torch.Tensor, x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Relative gradient error per field, sqrt(sum |d(x_hat - x)|^2 / sum |dx|^2) with forward differences: the VRMSE
    of the gradient (1 = a flat prediction). [C]."""

    def grads(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return t[..., 1:, :] - t[..., :-1, :], t[..., :, 1:] - t[..., :, :-1]

    num = sum((a - b).square().sum((0, 2, 3)) for a, b in zip(grads(x_hat), grads(x), strict=True))
    return (num / (sum(g.square().sum((0, 2, 3)) for g in grads(x)) + eps)).sqrt()


def spectral(x_hat: torch.Tensor, x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """The Well's binned spectral error per field: the error energy in three isotropic wavenumber bins (log-spaced
    from 2 pi / N to pi sqrt 2) relative to the truth's energy in the bin, as a root (1 = predicting zero at those
    scales). [C, 3] for low, mid, high."""
    h, w = x.shape[-2:]
    freqs = torch.meshgrid(torch.fft.fftfreq(h, device=x.device), torch.fft.fftfreq(w, device=x.device), indexing="ij")
    k = torch.stack(freqs).norm(dim=0).flatten() * 2 * math.pi
    edges = torch.logspace(math.log10(2 * math.pi / max(h, w)), math.log10(math.pi * math.sqrt(2) + 1e-6), 4).to(k)
    masks = F.one_hot((k[:, None] > edges[1:-1]).sum(1), 3).to(x.dtype)  # [H W, 3]; k = 0 counts as low

    def energy(t: torch.Tensor) -> torch.Tensor:  # [C, 3]
        return torch.fft.fft2(t).abs().square().flatten(-2).sum(0) @ masks

    return (energy(x_hat - x) / (energy(x) + eps)).sqrt()


def ssim(x_hat: torch.Tensor, x: torch.Tensor, window: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """SSIM per field (Wang et al.: Gaussian window, C1 = (0.01 L)^2, C2 = (0.03 L)^2), L the truth's range over the
    trajectory; mean over frames and window positions. [C]."""
    c = x.shape[1]
    g = torch.exp(-((torch.arange(window, device=x.device) - window // 2) ** 2) / (2 * sigma**2))
    g = g / g.sum()
    kernel = (g[:, None] * g[None, :]).expand(c, 1, window, window).to(x.dtype)

    def blur(t: torch.Tensor) -> torch.Tensor:
        return F.conv2d(t, kernel, groups=c)

    mu_hat, mu = blur(x_hat), blur(x)
    var_hat, var, cov = blur(x_hat * x_hat) - mu_hat**2, blur(x * x) - mu**2, blur(x_hat * x) - mu_hat * mu
    span = (x.amax((0, 2, 3)) - x.amin((0, 2, 3)))[:, None, None] + 1e-8
    c1, c2 = (0.01 * span) ** 2, (0.03 * span) ** 2
    s = (2 * mu_hat * mu + c1) * (2 * cov + c2) / ((mu_hat**2 + mu**2 + c1) * (var_hat + var + c2))
    return s.mean((0, 2, 3))


FIELD_METRICS = ("vrmse", "h1", "ssim", "spectral_low", "spectral_mid", "spectral_high")


def field_metrics(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Every per-field metric of one trajectory, stacked in the order of FIELD_METRICS. [len(FIELD_METRICS), C]."""
    return torch.cat([vrmse(x_hat, x)[None], h1(x_hat, x)[None], ssim(x_hat, x)[None], spectral(x_hat, x).T])


WELL_WINDOWS = ((2, 8), (9, 26), (27, 56), (57, 100))  # rollout windows of The Well's benchmark tables (frames)


def vrmse_per_frame(x_hat: torch.Tensor, x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """The Well's rollout metric: VRMSE of each frame against that frame's own variance, mean over fields. [N]."""
    dims = (2, 3)
    return ((x_hat - x).square().mean(dims) / (x.var(dims, unbiased=False) + eps)).sqrt().mean(1)


class Aggregate:
    """Per-field metrics of every evaluated simulation, aggregated: `add` one simulation at a time, `reduce` the sums
    over ranks, `metrics` at the end. `trajectory`: the frames are one trajectory in time order (else random clips:
    no per-frame curve)."""

    def __init__(self, fields: list[str], device, trajectory: bool = True):
        self.fields = fields
        self.n = torch.zeros((), device=device)
        self.sum = torch.zeros(len(FIELD_METRICS), len(fields), device=device)  # of field_metrics over sims
        self.sum_sq = torch.zeros((), device=device)  # of each sim's mean VRMSE: its std over sims
        self.curve = torch.zeros(0, device=device) if trajectory else None  # per-frame VRMSE summed over sims

    def add(self, x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Scores one simulation [N, C, H, W]; returns its field_metrics."""
        m = field_metrics(x_hat, x)
        self.n += 1
        self.sum += m
        self.sum_sq += m[0].mean() ** 2
        if self.curve is not None:
            self.curve = self.curve + vrmse_per_frame(x_hat, x) if len(self.curve) else vrmse_per_frame(x_hat, x)
        return m

    def reduce(self) -> None:
        all_sum(self.n, self.sum, self.sum_sq, self.curve)

    def metrics(self, stride: int = 1) -> dict:
        """<metric>_<field> and <metric>_mean, vrmse_std over sims; on trajectories also vrmse_per_frame and its means
        over The Well's windows (given in original frames: `stride` frames per curve entry)."""
        per_field = self.sum / self.n
        out = {}
        for name, values in zip(FIELD_METRICS, per_field, strict=True):
            out |= {f"{name}_{f}": v.item() for f, v in zip(self.fields, values, strict=True)}
            out[f"{name}_mean"] = values.mean().item()
        out["vrmse_std"] = (self.sum_sq / self.n - per_field[0].mean() ** 2).clamp(min=0).sqrt().item()
        if self.curve is not None:
            curve = self.curve / self.n
            out["vrmse_per_frame"] = curve.tolist()
            windows = {w: curve[math.ceil(w[0] / stride) : w[1] // stride + 1] for w in WELL_WINDOWS}
            out |= {f"vrmse_frames_{lo}_{hi}": c.mean().item() for (lo, hi), c in windows.items() if len(c)}
        return out
