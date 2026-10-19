"""Console lines of the stage scripts: metrics and training progress."""

from __future__ import annotations

import torch

from windinet.eulermq.data import SYMBOLS


def hours(seconds: float) -> str:
    return f"{seconds / 3600:.1f} h" if seconds >= 3600 else f"{seconds / 60:.0f} min"


def log_metrics(tag: str, metrics: dict[str, float], loss: float | None = None, elapsed: float | None = None) -> None:
    """Named lines: the VRMSE (mean over fields ± std over sims; the train-sim mean and the VAE drift when present)
    with the train loss and elapsed seconds when given, then per field in physical units, then the structure metrics
    and the per frame windows when present (rollouts)."""
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
    if "h1_mean" in metrics:
        spectral = "  ".join(f"{b} {metrics[f'spectral_{b}_mean']:.3f}" for b in ("low", "mid", "high"))
        structure = f"H1 {metrics['h1_mean']:.3f}  SSIM {metrics['ssim_mean']:.3f}  spectral {spectral}"
        lines.append(f"{pad} | structure  {structure}")
    prefix = "vrmse_frames_"
    windows = {k[len(prefix) :].replace("_", "-"): v for k, v in metrics.items() if k.startswith(prefix)}
    if windows:
        lines.append(f"{pad} | per frame  " + "  ".join(f"{w} {v:.3f}" for w, v in windows.items()))
    if "bench_rmse" in metrics:
        bench = "  ".join(f"{k} {metrics[f'bench_{k}']:.3f}" for k in ("rmse", "vrmse", "ssim", "mlw"))
        lines.append(f"{pad} | cfd_bench  frames {metrics['bench_frames']}  {bench} (ssim = 1 - SSIM)")
    print("\n".join(lines))


def log_progress(
    step: int, steps: int, epoch: float, loss: float, lrs: list[float], updates: list[float], eta: float
) -> None:
    """One training line: mean loss over the log window, lrs, update ratios (step |dw|/|w|), eta, peak memory."""
    upd = f" | upd {' '.join(f'{u:.1e}' for u in updates)}" if updates else ""
    line = (
        f"step {step}/{steps} ({100 * step / steps:3.0f} %) | epoch {epoch:.2f} | loss {loss:.4f} | "
        f"lr {' '.join(f'{lr:.1e}' for lr in lrs)}{upd} | eta {hours(eta)}"
    )
    if torch.cuda.is_available():
        line += f" | peak {torch.cuda.max_memory_allocated() / 2**30:.0f} GB"
    print(line)
