"""Figures written into a run folder: training curves and a reconstruction panel."""

from __future__ import annotations

from pathlib import Path

import torch


def save_curves(steps: list[dict], history: list[dict], blocks: dict[str, float], fields: list[str], path: Path):
    """Train loss per update, validation VRMSE per field, relative weight change per group, drift per block."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    loss = torch.tensor([s["loss"] for s in steps])
    ax[0, 0].semilogy([s["step"] for s in steps], loss, lw=0.5, alpha=0.5)
    if len(loss) >= 10:  # heavy-tailed (shock frames): a running mean shows the trend
        ax[0, 0].semilogy(range(10, len(loss) + 1), loss.unfold(0, 10, 1).mean(1), "k", label="10-update mean")
        ax[0, 0].legend(fontsize=8)
    ax[0, 0].set(title="train loss per update", xlabel="step")
    for name in fields:
        ax[0, 1].plot([h["step"] for h in history], [h[f"vrmse_{name}"] for h in history], marker="o", label=name)
    ax[0, 1].set(title="validation VRMSE", xlabel="step", ylim=(0, None))
    ax[0, 1].legend(fontsize=8)
    n_groups = len(steps[0]["update_ratio"]) if steps else 0
    for g, label in enumerate(["adapter", "vae"][:n_groups]):
        ax[1, 0].semilogy(
            [s["step"] for s in steps], [s["update_ratio"][g] for s in steps], lw=0.8, label=f"{label}: step |dw|/|w|"
        )
    if history and "vae_drift" in history[-1]:
        ax[1, 0].semilogy(
            [h["step"] for h in history], [h["vae_drift"] for h in history], "k--", marker="o", label="vae: drift"
        )
    ax[1, 0].set(title="relative weight change", xlabel="step")
    ax[1, 0].legend(fontsize=8)
    ax[1, 1].barh(list(blocks), list(blocks.values()))
    ax[1, 1].set(title="drift from start weights per block" if blocks else "VAE frozen")
    ax[1, 1].tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def save_panel(example: tuple, fields: list[str], path: Path) -> None:
    """Per field: GT / reconstruction / residual; last column the RGB image(s) fed to the VAE. Small PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x, x_hat, rgbs = example
    fig, axes = plt.subplots(len(fields), 4, figsize=(12, 2.75 * len(fields)), squeeze=False)
    for i, name in enumerate(fields):
        lo, hi = x[i].min().item(), x[i].max().item()
        same = {"vmin": lo, "vmax": hi}
        panels = [
            (x[i], f"{name} GT", same),
            (x_hat[i], "recon", same),
            (x_hat[i] - x[i], "residual", {"cmap": "RdBu_r"}),
        ]
        for j, (img, title, kw) in enumerate(panels):
            im = axes[i, j].imshow(img, **kw)
            axes[i, j].set_title(title, fontsize=9)
            plt.colorbar(im, ax=axes[i, j], fraction=0.046)
    for g, rgb in enumerate(rgbs):
        axes[g, 3].imshow((rgb.permute(1, 2, 0) + 1) / 2)
        axes[g, 3].set_title(f"adapter RGB {'ABC'[g]}", fontsize=9)
    for ax in axes.flat:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=70)
    plt.close(fig)
