"""Figures and videos written into a run folder: training curves, a reconstruction panel, frame strips, mp4."""

from __future__ import annotations

from pathlib import Path

import torch

from windinet.field_adapter import group_tag


def to_uint8(video: torch.Tensor) -> torch.Tensor:
    """RGB [3, F, H, W] in (-1, 1) -> [F, H, W, 3] uint8."""
    return ((video.permute(1, 2, 3, 0) + 1) * 127.5).round().clamp(0, 255).to(torch.uint8).cpu()


def save_video(video: torch.Tensor, path: Path, fps: int) -> None:
    """mp4 of an RGB [3, F, H, W] video in (-1, 1)."""
    from diffusers.utils import export_to_video

    export_to_video(list(to_uint8(video).numpy()), str(path), fps=fps)


def save_strip(video: torch.Tensor, path: Path, reference: torch.Tensor | None = None, n: int = 5) -> None:
    """`n` frames spread over an RGB [3, F, H, W] video in (-1, 1), side by side; with `reference` (same shape) two
    more rows: the reference and |video - reference| on a fixed scale (0 to a quarter of the colour range), so strips of
    different runs compare directly."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [("generated", video)] if reference is None else [("generated", video), ("ground truth", reference)]
    idx = torch.linspace(0, video.shape[1] - 1, n).round().long()
    n_rows = len(rows) + (reference is not None)
    h, w = video.shape[-2:]
    fig, axes = plt.subplots(n_rows, n, figsize=(3 * n, 3 * n_rows * h / w), squeeze=False)
    for r, (label, v) in enumerate(rows):
        frames = to_uint8(v)
        for ax, i in zip(axes[r], idx, strict=True):
            ax.imshow(frames[i])
            ax.set_title(f"{label}, frame {i}", fontsize=9)
    if reference is not None:
        diff = (video - reference).abs().mean(0).cpu()  # [F, H, W], mean over RGB
        for ax, i in zip(axes[-1], idx, strict=True):
            ax.imshow(diff[i], cmap="magma", vmin=0, vmax=0.5)
            ax.set_title(f"|difference|, frame {i}", fontsize=9)
    for ax in axes.flat:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def save_fields_strip(
    pred: torch.Tensor, truth: torch.Tensor, fields: list[str], stds: torch.Tensor, path: Path, n: int = 5
):
    """Per field three rows at `n` frames: ground truth, prediction (both on the trajectory's own range) and their
    signed difference on ±0.5 dataset std, the same scale in every run. `pred`, `truth`: [F, C, H, W]."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    idx = torch.linspace(0, len(truth) - 1, n).round().long()
    fig, axes = plt.subplots(3 * len(fields), n, figsize=(2.6 * n, 2.6 * 3 * len(fields)), squeeze=False)
    for f, name in enumerate(fields):
        same = {"vmin": truth[:, f].min().item(), "vmax": truth[:, f].max().item()}
        diff = {"vmin": -0.5 * stds[f].item(), "vmax": 0.5 * stds[f].item(), "cmap": "RdBu_r"}
        for j, i in enumerate(idx):
            rows = [
                (truth[i, f], f"{name} GT", same),
                (pred[i, f], "prediction", same),
                (pred[i, f] - truth[i, f], "difference", diff),
            ]
            for r, (img, title, kw) in enumerate(rows):
                axes[3 * f + r, j].imshow(img, **kw)
                axes[3 * f + r, j].set_title(f"{title}, frame {i}", fontsize=8)
    for ax in axes.flat:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=70)
    plt.close(fig)


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
    if ax[1, 0].lines:  # empty for an eval-only run
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
        axes[g, 3].set_title(f"adapter RGB {group_tag(g)}", fontsize=9)
    for ax in axes.flat:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=70)
    plt.close(fig)
