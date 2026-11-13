"""Console lines of the stage scripts: metrics and training progress."""

from __future__ import annotations

import torch

from windinet.data.frames import SYMBOLS
from windinet.experiment.metrics import BINS


def hours(seconds: float) -> str:
    return f"{seconds / 3600:.1f} h" if seconds >= 3600 else f"{seconds / 60:.0f} min"


def _windows(metrics: dict, prefix: str = "vrmse_frames_", digits: int = 3) -> str:
    """`prefix`<window> values over The Well's frame windows, when present."""
    n = len(prefix)
    return "  ".join(f"{k[n:].replace('_', '-')} {v:.{digits}f}" for k, v in metrics.items() if k.startswith(prefix))


def log_metrics(tag: str, metrics: dict[str, float], loss: float | None = None, elapsed: float | None = None) -> None:
    """Named lines: the VRMSE (mean over fields ± std over sims; the train-sim mean and the VAE drift when present)
    with the train loss and elapsed seconds when given, then per field, per frame window, structure and power when
    present; then the VAE and DiT shares, the ensemble and cfd_bench lines when present."""
    head = f"VRMSE {metrics['vrmse_mean']:.3f} ± {metrics.get('vrmse_std', 0):.3f}"
    if "train_vrmse_mean" in metrics:
        head += f"  (train sims {metrics['train_vrmse_mean']:.3f})"
    if "vae_drift" in metrics:
        head += f"  (VAE drift {metrics['vae_drift']:.3g})"
    if loss is not None:
        head += f" | train loss {loss:.4f}"
    if elapsed is not None:
        head += f" | {hours(elapsed)}"
    pad = " " * len(tag)
    per_field = "  ".join(f"{SYMBOLS[f]} {metrics[f'vrmse_{f}']:.3f}" for f in SYMBOLS if f"vrmse_{f}" in metrics)
    lines = [f"{tag} | {head}", f"{pad} | per field  {per_field}"]
    if windows := _windows(metrics):
        lines.append(f"{pad} | per frame  {windows}")
    if "h1_mean" in metrics:
        spectral = "  ".join(f"{b} {metrics[f'spectral_{b}_mean']:.3f}" for b in BINS)
        lines.append(
            f"{pad} | structure  H1 {metrics['h1_mean']:.3f}  SSIM {metrics['ssim_mean']:.3f}  spectral {spectral}"
        )
    if "lde_mean" in metrics:
        lines.append(f"{pad} | detail     LDE {metrics['lde_mean']:.3f}  front shift {metrics['shift_mean']:.1f} px")
    if "power_high_mean" in metrics:
        power = "  ".join(f"{b} {metrics[f'power_{b}_mean']:.2f}" for b in BINS)
        lines.append(f"{pad} | power      {power}  (prediction / truth, 1 = no blur)")
    if "teacher_cos" in metrics:
        close = f"velocity cos {metrics['teacher_cos']:.4f}  rel MSE {metrics['teacher_rel_mse']:.4f}"
        lines.append(f"{pad} | teacher    {close} (fixed val batch)")
    for prefix, label in (("vae", "VAE share"), ("dit", "DiT share"), ("wan", "Wan alone")):
        if f"{prefix}_vrmse_mean" in metrics:
            sub = {k[len(prefix) + 1 :]: v for k, v in metrics.items() if k.startswith(f"{prefix}_")}
            lines.append(f"{pad} | {label}  VRMSE {sub['vrmse_mean']:.3f}  {_windows(sub)}")
    if "ens_samples" in metrics:
        m = {k[4:]: v for k, v in metrics.items() if k.startswith("ens_")}
        mean = f"VRMSE {m['vrmse_mean']:.3f}  {_windows(m)}  power high {m['power_high_mean']:.2f}"
        lines.append(f"{pad} | ensemble   {m['samples']} samples, their mean: {mean}")
        spread = f"spread/skill {m['spread_skill_mean']:.2f}  {_windows(m, 'spread_skill_frames_', 2)}"
        lines.append(f"{pad} |            {spread}  CRPS {m['crps_mean']:.3f}")
    for prefix, label in (("bench", "cfd_bench "), ("bench_vae", "  its VAE ")):
        if f"{prefix}_rmse" in metrics:
            names = ("rmse", "vrmse", "ssim", "mlw", "h1", "h2")
            bench = "  ".join(f"{k} {metrics[f'{prefix}_{k}']:.3f}" for k in names if f"{prefix}_{k}" in metrics)
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
        f"step {step}/{steps} ({100 * step / steps:3.0f} %) | epoch {epoch:.2f} | loss {loss:.4f} | "
        f"lr {' '.join(f'{lr:.1e}' for lr in lrs)}{upd} | eta {hours(eta)}"
    )
    line += f" | {extra}" if extra else ""
    if torch.cuda.is_available():
        line += f" | peak {torch.cuda.max_memory_allocated() / 2**30:.0f} GB"
    print(line)
