"""Metrics of one simulation's trajectory [N, C, H, W] against its prediction, per field: VRMSE (per trajectory and
per frame), relative gradient error (H1), The Well's binned spectral error and SSIM; aggregated over simulations."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from windinet.experiment.distributed import all_sum, gather


def vrmse(x_hat: torch.Tensor, x: torch.Tensor, var: torch.Tensor | None = None, eps: float = 1e-8) -> torch.Tensor:
    """Per field: sqrt(mean(err^2) / var(x)), population variance over frames and pixels (Li 2026, eq. 5.11).

    `x`, `x_hat`: [N, C, H, W] frames of ONE simulation, so the variance is that simulation's own. Scale-free:
    physical units and z-scored fields give the same number. `var` [C]: a given normaliser instead. Returns [C]."""
    dims = (0, 2, 3)
    var = x.var(dims, unbiased=False) if var is None else var
    return ((x_hat - x).square().mean(dims) / (var + eps)).sqrt()


def h1(x_hat: torch.Tensor, x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Relative gradient error per field, sqrt(sum |d(x_hat - x)|^2 / sum |dx|^2) with forward differences: the VRMSE
    of the gradient (1 = a flat prediction). [C]."""

    def grads(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return t[..., 1:, :] - t[..., :-1, :], t[..., :, 1:] - t[..., :, :-1]

    num = sum((a - b).square().sum((0, 2, 3)) for a, b in zip(grads(x_hat), grads(x), strict=True))
    return (num / (sum(g.square().sum((0, 2, 3)) for g in grads(x)) + eps)).sqrt()


def _wavenumbers(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """|k| of every FFT coefficient of [..., H, W] (flattened) and The Well's bin edges: three isotropic bins,
    log-spaced from 2 pi / N to pi sqrt 2."""
    h, w = x.shape[-2:]
    freqs = torch.meshgrid(torch.fft.fftfreq(h, device=x.device), torch.fft.fftfreq(w, device=x.device), indexing="ij")
    k = torch.stack(freqs).norm(dim=0).flatten() * 2 * math.pi
    edges = torch.logspace(math.log10(2 * math.pi / max(h, w)), math.log10(math.pi * math.sqrt(2) + 1e-6), 4).to(k)
    return k, edges


def shells(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Integer wavenumber shell of every FFT coefficient of [..., H, W] (flattened) and The Well's bin edges, both in
    units of 2 pi / N."""
    k, edges = _wavenumbers(x)
    unit = 2 * math.pi / max(x.shape[-2:])
    return (k / unit).round().long(), edges / unit


def _energy(t: torch.Tensor) -> torch.Tensor:
    """Spectral energy of [N, C, H, W] per field and FFT coefficient, summed over frames. [C, H W]."""
    return torch.fft.fft2(t).abs().square().flatten(-2).sum(0)


def spectral(x_hat: torch.Tensor, x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """The Well's binned spectral error per field: the error energy per bin relative to the truth's (k = 0 counts as
    low), as a root (1 = predicting zero at those scales). [C, 3] for low, mid, high."""
    k, edges = _wavenumbers(x)
    masks = F.one_hot((k[:, None] > edges[1:-1]).sum(1), 3).to(x.dtype)  # [H W, 3]
    return ((_energy(x_hat - x) @ masks) / (_energy(x) @ masks + eps)).sqrt()


def power(x_hat: torch.Tensor, x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Power ratio per field, prediction over truth, per integer wavenumber shell (Subich et al. 2025) from k = 1 to
    the Nyquist circle, geometric mean over the shells of each of The Well's bins: below 1 = smoothed, above 1 =
    spurious detail; a position error alone leaves it at 1. [C, 3]."""
    shell, edges = shells(x)
    n = max(x.shape[-2:]) // 2 + 1  # beyond Nyquist the shells are incomplete

    def per_shell(t: torch.Tensor) -> torch.Tensor:
        e = torch.zeros(t.shape[1], int(shell.max()) + 1, device=x.device, dtype=x.dtype)
        return e.index_add_(1, shell, _energy(t))[:, 1:n]

    log_ratio = (per_shell(x_hat) + eps).log() - (per_shell(x) + eps).log()
    band = (torch.arange(1, n, device=x.device)[:, None] > edges[1:-1]).sum(1)
    return torch.stack([log_ratio[:, band == b].mean(1).exp() for b in range(3)], 1)


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


def gauss(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur of [N, C, H, W], radius 3 sigma, edge padding."""
    r = int(3 * sigma + 0.5)
    k = torch.exp(-(torch.arange(-r, r + 1, device=x.device, dtype=x.dtype) ** 2) / (2 * sigma**2))
    k, (n, c, h, w) = k / k.sum(), x.shape
    y = F.conv2d(F.pad(x.reshape(n * c, 1, h, w), (r, r, 0, 0), mode="replicate"), k.view(1, 1, 1, -1))
    return F.conv2d(F.pad(y, (0, 0, r, r), mode="replicate"), k.view(1, 1, -1, 1)).reshape(n, c, h, w)


def lde(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Local detail-energy error per field: sum |A(x_hat) - A(x)| / sum A(x), A the local (8 px) amplitude of the
    detail below 2 px blur. Tolerates small shifts, penalises blur and spurious detail (0 = the truth's detail). [C]."""

    def amp(t: torch.Tensor) -> torch.Tensor:
        return gauss((t - gauss(t, 2.0)).square(), 8.0).clamp(min=0).sqrt()

    a_hat, a = amp(x_hat), amp(x)
    return (a_hat - a).abs().sum((0, 2, 3)) / (a.sum((0, 2, 3)) + 1e-8)


def shift(x_hat: torch.Tensor, x: torch.Tensor, radius: int = 12, tile: int = 32, top: float = 0.25) -> torch.Tensor:
    """Front displacement per field in pixels: on the `top` share of tiles with the steepest truth, the best integer
    shift of the prediction within `radius`; mean |shift| over the tiles it fits (removes half their error or more),
    0 if none. [C]."""
    tile = min(tile, *x.shape[-2:])

    def err(a: torch.Tensor) -> torch.Tensor:  # squared error per tile [N, C, tiles]
        return F.avg_pool2d((a - x).square(), tile).flatten(2)

    grad = F.pad(x.diff(dim=-1)[..., :-1, :].square() + x.diff(dim=-2)[..., :-1].square(), (0, 1, 0, 1))
    steep = F.avg_pool2d(grad, tile).flatten(2)
    front = steep >= steep.quantile(1 - top, dim=2, keepdim=True)
    pad, (h, w) = F.pad(x_hat, (radius,) * 4, mode="replicate"), x.shape[-2:]
    err0 = best = err(x_hat)
    dist = torch.zeros_like(best)
    offsets = [(dy, dx) for dy in range(-radius, radius + 1) for dx in range(-radius, radius + 1)]
    for dy, dx in sorted(offsets, key=lambda o: math.hypot(*o)):  # nearest first; a farther shift must gain 1 %, so
        e = err(pad[..., radius - dy : radius - dy + h, radius - dx : radius - dx + w])  # one along a front loses
        better = e < 0.99 * best
        dist, best = torch.where(better, math.hypot(dy, dx), dist), torch.where(better, e, best)
    fits = front & (best <= 0.5 * err0)
    return (dist * fits).sum((0, 2)) / fits.sum((0, 2)).clamp(min=1)


BINS = ("low", "mid", "high")
FIELD_METRICS = ("vrmse", "h1", "ssim", "lde", "shift") + tuple(f"{m}_{b}" for m in ("spectral", "power") for b in BINS)


def field_metrics(x_hat: torch.Tensor, x: torch.Tensor, var: torch.Tensor | None = None) -> torch.Tensor:
    """Every per-field metric of one trajectory, stacked in the order of FIELD_METRICS; `var`: the VRMSE's
    normaliser. [len(FIELD_METRICS), C]."""
    rows = [vrmse(x_hat, x, var)[None]] + [f(x_hat, x)[None] for f in (h1, ssim, lde, shift)]
    return torch.cat([*rows, spectral(x_hat, x).T, power(x_hat, x).T])


def ensemble_scores(samples: torch.Tensor, x: torch.Tensor, given: int = 0) -> torch.Tensor:
    """Per frame and field, of n >= 2 samples [n, N, C, H, W] of one simulation against it [N, C, H, W], each over
    the simulation's variance after the `given` frames (as the VRMSE): the ensemble variance (unbiased, mean over
    pixels), the squared error of the ensemble mean, and the fair CRPS (Ferro 2014) over the std. [3, N, C]; summed
    over sims and frames, sqrt((n + 1) / n * var / err) is the spread-skill ratio (Fortin et al. 2014; 1 =
    calibrated)."""
    n, dims = len(samples), (2, 3)
    scale = x[given:].var((0, 2, 3), unbiased=False) + 1e-8
    spread = samples.var(0).mean(dims) / scale
    skill = (samples.mean(0) - x).square().mean(dims) / scale
    rank = 2 * torch.arange(1, n + 1, device=x.device, dtype=x.dtype) - n - 1  # half of sum_ij |s_i - s_j|, sorted
    pairs = (rank.view(-1, 1, 1, 1, 1) * samples.sort(0).values).sum(0) / (n * (n - 1))
    crps = ((samples - x).abs().mean(0) - pairs).mean(dims) / scale.sqrt()
    return torch.stack([spread, skill, crps])


WELL_WINDOWS = ((2, 8), (9, 26), (27, 56), (57, 100))  # rollout windows of The Well's benchmark tables (frames)


def well_windows(per_frame: torch.Tensor, stride: int = 1) -> dict[str, torch.Tensor]:
    """The entries of a per-frame tensor [N, ...] in each of The Well's windows (given in original frames: `stride`
    frames per entry), keyed "<lo>_<hi>"; windows the trajectory does not reach are left out."""
    windows = {f"{lo}_{hi}": per_frame[math.ceil(lo / stride) : hi // stride + 1] for lo, hi in WELL_WINDOWS}
    return {w: v for w, v in windows.items() if len(v)}


def vrmse_per_frame(
    x_hat: torch.Tensor, x: torch.Tensor, var: torch.Tensor | None = None, eps: float = 1e-8
) -> torch.Tensor:
    """The Well's rollout metric: VRMSE of each frame against that frame's own variance (or `var` [C]), mean over
    fields. [N]."""
    dims = (2, 3)
    var = x.var(dims, unbiased=False) if var is None else var
    return ((x_hat - x).square().mean(dims) / (var + eps)).sqrt().mean(1)


class Aggregate:
    """Per-field metrics of every evaluated simulation, aggregated: `add` one simulation at a time, `reduce` the sums
    over ranks, `metrics` at the end. `trajectory`: the frames are one trajectory in time order (else random clips:
    no per-frame curve). `given`: leading frames the model was shown; the per-field metrics skip them, the per-frame
    curve keeps them."""

    def __init__(self, fields: list[str], device, trajectory: bool = True, given: int = 0):
        self.fields, self.given = fields, given
        self.n = torch.zeros((), device=device)
        self.sum = torch.zeros(len(FIELD_METRICS), len(fields), device=device)  # of field_metrics over sims
        self.sum_sq = torch.zeros((), device=device)  # of each sim's mean VRMSE: its std over sims
        self.curve = torch.zeros(0, device=device) if trajectory else None  # per-frame VRMSE summed over sims
        self.per_sim: list[torch.Tensor] = []  # each sim's VRMSE per field: the median over sims

    def add(self, x_hat: torch.Tensor, x: torch.Tensor, var: torch.Tensor | None = None) -> torch.Tensor:
        """Scores one simulation [N, C, H, W]; returns its field_metrics. `var` [C]: the normaliser of the VRMSE and
        the per-frame curve (None: the scored frames' variance, each frame's own for the curve)."""
        m = field_metrics(x_hat[self.given :], x[self.given :], var)
        self.n += 1
        self.sum += m
        self.sum_sq += m[0].mean() ** 2
        self.per_sim.append(m[0].cpu())
        if self.curve is not None:
            frames = vrmse_per_frame(x_hat, x, var)
            self.curve = self.curve + frames if len(self.curve) else frames
        return m

    def reduce(self) -> None:
        all_sum(self.n, self.sum, self.sum_sq, self.curve)
        self.per_sim = [v for part in gather(self.per_sim) for v in part]

    def metrics(self, stride: int = 1) -> dict:
        """<metric>_<field> and <metric>_mean, vrmse_std and vrmse_median_<field | mean> over sims; on trajectories also
        vrmse_per_frame and its means over The Well's windows (given in original frames: `stride` frames per entry)."""
        per_field = self.sum / self.n
        out = {}
        for name, values in zip(FIELD_METRICS, per_field, strict=True):
            out |= {f"{name}_{f}": v.item() for f, v in zip(self.fields, values, strict=True)}
            out[f"{name}_mean"] = values.mean().item()
        out["vrmse_std"] = (self.sum_sq / self.n - per_field[0].mean() ** 2).clamp(min=0).sqrt().item()
        per_sim = torch.stack(self.per_sim)  # [sims, C]: robust to the few low-variance sims that dominate the mean
        out |= {f"vrmse_median_{f}": v.item() for f, v in zip(self.fields, per_sim.median(0).values, strict=True)}
        out["vrmse_median_mean"] = per_sim.mean(1).median().item()
        if self.curve is not None:
            curve = self.curve / self.n
            out["vrmse_per_frame"] = curve.tolist()
            scored = torch.cat([torch.full_like(curve[: self.given], torch.nan), curve[self.given :]])
            windows = well_windows(scored, stride).items()
            out |= {f"vrmse_frames_{w}": c.nanmean().item() for w, c in windows if not c.isnan().all()}
        return out
