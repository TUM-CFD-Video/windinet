"""Console lines of the stage scripts: metric reports and training progress."""

from __future__ import annotations

import torch

from windinet.eulermq.data import SYMBOLS


def hours(seconds: float) -> str:
    return f"{seconds / 3600:.1f} h" if seconds >= 3600 else f"{seconds / 60:.0f} min"


def report(tag: str, metrics: dict[str, float], fields: list[str], suffix: str = "") -> None:
    """Two lines: the means (and the VAE drift when present), then the per-field VRMSE in physical units."""
    means = f"mean {metrics['vrmse_mean']:.4f}  clipped {metrics['vrmse_clipped_mean']:.4f}"
    if "vae_drift" in metrics:
        means += f"  drift {metrics['vae_drift']:.3g}"
    per_field = "  ".join(f"{SYMBOLS[f]} {metrics['vrmse_' + f]:.4f}" for f in fields)
    print(f"{tag} | {means}{suffix}\n{' ' * len(tag)} | {per_field}")


def progress(
    step: int, steps: int, epoch: float, loss: float, lrs: list[float], updates: list[float], eta: float
) -> str:
    """One training line: mean loss over the log window, lrs, update ratios (step |dw|/|w|), eta, peak memory."""
    line = (
        f"step {step}/{steps} ({100 * step / steps:3.0f} %) | epoch {epoch:.2f} | loss {loss:.4f} | "
        f"lr {' '.join(f'{lr:.1e}' for lr in lrs)} | upd {' '.join(f'{u:.1e}' for u in updates)} | eta {hours(eta)}"
    )
    if torch.cuda.is_available():
        line += f" | peak {torch.cuda.max_memory_allocated() / 2**30:.0f} GB"
    return line
