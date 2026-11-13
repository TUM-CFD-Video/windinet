"""Console lines of the stage scripts: metrics and training progress."""

from __future__ import annotations

import re

import torch

from windinet.data.specs import SPECS
from windinet.experiment.metrics import BINS

SYMBOLS = {f: s for spec in SPECS.values() for f, s in spec.display.symbols.items()}  # shared fields share a symbol


def hours(seconds: float) -> str:
    return f"{seconds / 3600:.1f} h" if seconds >= 3600 else f"{seconds / 60:.0f} min"


def num(v: float, fmt: str = ".3f") -> str:
    """`v` in `fmt` when that shows at most 5 significant digits, else 5 of them (scientific outside [1, 1e5))."""
    s = format(v, fmt)
    digits = s.lstrip("-").replace(".", "").lstrip("0")
    if "e" in s or (len(digits) <= 5 and (digits or v == 0)):
        return s
    whole = len(str(int(abs(v)))) if abs(v) >= 1 else 6
    return f"{v:.{5 - whole}f}" if whole <= 5 else f"{v:.4e}"


def _windows(metrics: dict, prefix: str = "vrmse_frames_", digits: int = 3) -> str:
    """`prefix`<window> values over The Well's frame windows, when present."""
    n = len(prefix)
    return "  ".join(
        f"{k[n:].replace('_', '-')} {num(v, f'.{digits}f')}" for k, v in metrics.items() if k.startswith(prefix)
    )


def log_metrics(tag: str, metrics: dict[str, float], loss: float | None = None, elapsed: float | None = None) -> None:
    """Named lines: the VRMSE (mean over fields ± std over sims; the train-sim mean and the VAE drift when present)
    with the train loss and elapsed seconds when given, then per field, per frame window, structure and power when
    present; then the VAE and DiT shares, the ensemble and cfd_bench lines when present."""
    head = f"VRMSE {num(metrics['vrmse_mean'])} ± {num(metrics.get('vrmse_std', 0))}"
    if "vrmse_median_mean" in metrics:
        head += f" (median {num(metrics['vrmse_median_mean'])})"
    if "train_vrmse_mean" in metrics:
        head += f"  (train sims {num(metrics['train_vrmse_mean'])})"
    if "vae_drift" in metrics:
        head += f"  (VAE drift {num(metrics['vae_drift'], '.3g')})"
    if loss is not None:
        head += f" | train loss {num(loss, '.4f')}"
    if elapsed is not None:
        head += f" | {hours(elapsed)}"
    pad = " " * len(tag)
    fields = [k[6:] for k in metrics if k.startswith("vrmse_") and k[6:] in SYMBOLS]  # the run's dataset order
    per_field = "  ".join(f"{SYMBOLS[f]} {num(metrics[f'vrmse_{f}'])}" for f in fields)
    lines = [f"{tag} | {head}", f"{pad} | per field  {per_field}"]
    if all(f"vrmse_median_{f}" in metrics for f in fields):
        medians = "  ".join(f"{SYMBOLS[f]} {num(metrics[f'vrmse_median_{f}'])}" for f in fields)
        lines.append(f"{pad} | median     {medians}")
    by_scalar: dict[str, list[str]] = {}
    for k, v in metrics.items():
        if m := re.fullmatch(r"vrmse_mean_([a-z]+)(.+)", k):
            by_scalar.setdefault(m[1], []).append(f"{m[2]} {num(v)}")
    lines += [f"{pad} | {'per ' + name:<10} {'  '.join(values)}" for name, values in by_scalar.items()]
    if windows := _windows(metrics):
        lines.append(f"{pad} | per frame  {windows}")
    if "h1_mean" in metrics:
        spectral = "  ".join(f"{b} {num(metrics[f'spectral_{b}_mean'])}" for b in BINS)
        lines.append(
            f"{pad} | structure  H1 {num(metrics['h1_mean'])}  SSIM {num(metrics['ssim_mean'])}  spectral {spectral}"
        )
    if "lde_mean" in metrics:
        lines.append(
            f"{pad} | detail     LDE {num(metrics['lde_mean'])}  front shift {num(metrics['shift_mean'], '.1f')} px"
        )
    if "power_high_mean" in metrics:
        power = "  ".join(f"{b} {num(metrics[f'power_{b}_mean'], '.2f')}" for b in BINS)
        lines.append(f"{pad} | power      {power}  (prediction / truth, 1 = no blur)")
    if "teacher_cos" in metrics:
        close = f"velocity cos {num(metrics['teacher_cos'], '.4f')}  rel MSE {num(metrics['teacher_rel_mse'], '.4f')}"
        lines.append(f"{pad} | teacher    {close} (fixed val batch)")
    for prefix, label in (("vae", "VAE share"), ("dit", "DiT share"), ("wan", "Wan alone")):
        if f"{prefix}_vrmse_mean" in metrics:
            sub = {k[len(prefix) + 1 :]: v for k, v in metrics.items() if k.startswith(f"{prefix}_")}
            lines.append(f"{pad} | {label}  VRMSE {num(sub['vrmse_mean'])}  {_windows(sub)}")
    if "ens_samples" in metrics:
        m = {k[4:]: v for k, v in metrics.items() if k.startswith("ens_")}
        mean = f"VRMSE {num(m['vrmse_mean'])}  {_windows(m)}  power high {num(m['power_high_mean'], '.2f')}"
        lines.append(f"{pad} | ensemble   {m['samples']} samples, their mean: {mean}")
        spread = f"spread/skill {num(m['spread_skill_mean'], '.2f')}  {_windows(m, 'spread_skill_frames_', 2)}"
        lines.append(f"{pad} |            {spread}  CRPS {num(m['crps_mean'])}")
    for prefix, label in (("bench", "cfd_bench "), ("bench_vae", "  its VAE ")):
        if f"{prefix}_rmse" in metrics:
            names = ("rmse", "vrmse", "ssim", "mlw", "h1", "h2")
            bench = "  ".join(f"{k} {num(metrics[f'{prefix}_{k}'])}" for k in names if f"{prefix}_{k}" in metrics)
            lines.append(f"{pad} | {label} frames {metrics[f'{prefix}_frames']}  {bench} (ssim = 1 - SSIM)")
    print("\n".join(lines))


def log_progress(
    step: int,
    steps: int,
    epoch: float,
    loss: float,
    lrs: list[float],
    updates: list[float],
    eta: float,
    extra: str = "",
) -> None:
    """One training line: mean loss over the log window, lrs, update ratios (step |dw|/|w|), eta, `extra`, peak
    memory."""
    upd = f" | upd {' '.join(f'{u:.1e}' for u in updates)}" if updates else ""
    line = (
        f"step {step}/{steps} ({100 * step / steps:3.0f} %) | epoch {epoch:.2f} | loss {num(loss, '.4f')} | "
        f"lr {' '.join(f'{lr:.1e}' for lr in lrs)}{upd} | eta {hours(eta)}"
    )
    line += f" | {extra}" if extra else ""
    if torch.cuda.is_available():
        line += f" | peak {torch.cuda.max_memory_allocated() / 2**30:.0f} GB"
    print(line)
