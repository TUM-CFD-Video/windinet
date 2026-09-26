#!/usr/bin/env python
"""Chapter 3/6 thesis field figures from the eval sample npz files. CPU only.

The inputs are the per-sim .npz samples written by the two test-set eval
scripts. Both use the standalone 256x256 test.h5 and pick the same
gamma-spread sims (0000/0250/0499), with frames 0/25/50/75/100 saved:
  * scripts/eval_vae_test.py -> $SCRATCH/windinet/figure_data/vae_test/<run>/<sim>.npz
  * scripts/eval_dit_vrmse.py --test_h5 -> $SCRATCH/windinet/figure_data/dit/<arm>/<sim>.npz

The npz files stay on cluster scratch and the login nodes have no python, so
this runs as a batch job there: jobs/sng_pvc/thesis_figures.sbatch.

Subcommands, one per figure:
  dataset        Ch3  the four raw fields of one sim at several frames
  vae-dit-frame  Ch6  baseline only, one frame: GT | VAE only | VAE+DiT | the two |error| maps

The Chapter 6 ablation numbers are reported as tables, not figures.

Colours follow the dataviz reference palette. Magnitude fields (density,
pressure) use a one-hue blue ramp. Signed fields (momentum) use a blue<->red
diverging map with a gray midpoint and symmetric limits. |error| maps use a
one-hue orange ramp.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import typer
from matplotlib.colors import LinearSegmentedColormap

app = typer.Typer(pretty_exceptions_enable=False, no_args_is_help=True)

# --- palette (dataviz reference instance, light mode) -----------------------
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
AXIS = "#c3c2b7"
BLUE_RAMP = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
ORANGE_RAMP = ["#fcfcfb", "#fbe0d3", "#f5b597", "#eb6834", "#c24f1f", "#8a3413"]
DIVERGING = ["#0d366b", "#256abf", "#6da7ec", "#f0efec", "#ef8a89", "#e34948", "#8f1f1f"]
# starts at a light tint, not white: low values must not merge into the page
CMAP_SEQ = LinearSegmentedColormap.from_list("seq_blue", BLUE_RAMP[1:])
CMAP_ERR = LinearSegmentedColormap.from_list("seq_orange", ORANGE_RAMP)
CMAP_DIV = LinearSegmentedColormap.from_list("div_blue_red", DIVERGING)

CHANNEL_TEX = {"density": r"$\rho$", "momentum_x": r"$m_x$", "momentum_y": r"$m_y$", "pressure": r"$p$"}
SIGNED = {"momentum_x", "momentum_y"}
CHANNELS = ["density", "momentum_x", "momentum_y", "pressure"]

def _style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 9,
        "xtick.color": MUTED,
        "ytick.color": INK_2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "legend.frameon": False,
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
    })


def _save(fig, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"wrote {out}")


def _denorm(x: np.ndarray, mean: np.ndarray, std: np.ndarray, clip: float) -> np.ndarray:
    """Normalized [-1,1] fields [C, F, H, W] back to physical units (inverse of build_shockwave_video)."""
    return x.astype(np.float32) * (std * clip)[:, None, None, None] + mean[:, None, None, None]


def _field_limits(name: str, arrays: list[np.ndarray]) -> tuple[float, float]:
    stacked = np.concatenate([a.ravel() for a in arrays])
    if name in SIGNED:
        m = float(np.percentile(np.abs(stacked), 99.5))
        return -m, m
    return float(np.percentile(stacked, 0.5)), float(np.percentile(stacked, 99.5))


def _field_ax(ax, img, name, vmin, vmax):
    im = ax.imshow(img, cmap=CMAP_DIV if name in SIGNED else CMAP_SEQ, vmin=vmin, vmax=vmax, origin="lower")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    return im


def _err_ax(ax, img, vmax):
    im = ax.imshow(img, cmap=CMAP_ERR, vmin=0, vmax=vmax, origin="lower")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    return im


@app.command("vae-dit-frame")
def vae_dit_frame(
    npz: Path = typer.Option(..., help="Baseline DiT eval sample, figure_data/dit/loss_rmse_h1/<sim>.npz"),
    frame: int = typer.Option(100, help="0-indexed frame (must be one of the saved frames)"),
    out: Path = typer.Option(Path("figures/ch6_vae_dit_frame100.pdf")),
) -> None:
    """Rows = channels. Columns = GT, VAE-only reconstruction, VAE+DiT rollout, |error| of each.

    Both |error| maps of a row share one colour scale, so the VAE's share of
    the rollout error can be read off directly.
    """
    _style()
    d = np.load(npz)
    fi = list(d["frames"]).index(frame)
    mean, std, clip = d["channel_mean"], d["channel_std"], float(d["normalization_clip"])
    gt = _denorm(d["gt_norm"], mean, std, clip)[:, fi]
    vae = _denorm(d["vae_only_norm"], mean, std, clip)[:, fi]
    dit = _denorm(d["vae_dit_norm"], mean, std, clip)[:, fi]
    fig, axes = plt.subplots(4, 5, figsize=(8.2, 6.2), squeeze=False, gridspec_kw={"wspace": 0.08})
    for c, name in enumerate(CHANNELS):
        vmin, vmax = _field_limits(name, [gt[c], vae[c], dit[c]])
        im = _field_ax(axes[c, 0], gt[c], name, vmin, vmax)
        _field_ax(axes[c, 1], vae[c], name, vmin, vmax)
        _field_ax(axes[c, 2], dit[c], name, vmin, vmax)
        errs = [np.abs(vae[c] - gt[c]), np.abs(dit[c] - gt[c])]
        emax = float(np.percentile(np.concatenate([e.ravel() for e in errs]), 99.5)) or 1e-6
        _err_ax(axes[c, 3], errs[0], emax)
        eim = _err_ax(axes[c, 4], errs[1], emax)
        axes[c, 0].set_ylabel(CHANNEL_TEX[name], rotation=0, labelpad=12, color=INK, fontsize=11)
        fig.colorbar(im, ax=list(axes[c, :3]), fraction=0.02, pad=0.01, aspect=12).outline.set_visible(False)
        fig.colorbar(eim, ax=list(axes[c, 3:]), fraction=0.03, pad=0.01, aspect=12).outline.set_visible(False)
    for k, title in enumerate(["Ground truth", "VAE only", "VAE + DiT"]):
        axes[0, k].set_title(title)
    axes[0, 3].set_title("|error|\nVAE only", color=INK_2)
    axes[0, 4].set_title("|error|\nVAE + DiT", color=INK_2)
    # no suptitle: sim id and numbers belong in the LaTeX caption
    print(f"caption: test sim {npz.stem}, gamma = {float(d['gamma']):.3f}, frame {frame}; "
          f"whole-sim VRMSE (channel mean): VAE only {float(d['vae_only_vrmse_chmean']):.4f}, "
          f"VAE + DiT {float(d['vae_dit_vrmse_chmean']):.4f}")
    _save(fig, out)


@app.command("dataset")
def dataset(
    npz: Path = typer.Option(..., help="Any figure sample npz (uses its raw, unclipped gt_raw)"),
    frames: list[int] = typer.Option([0, 50, 100]),
    out: Path = typer.Option(Path("figures/ch3_euler_mq_sample.pdf")),
) -> None:
    """Rows = the four raw fields, columns = frames. Physical units, colour limits at the 0.5/99.5 percentiles."""
    _style()
    d = np.load(npz)
    saved = list(d["frames"])
    raw = d["gt_raw"].astype(np.float32)
    idx = [saved.index(f) for f in frames]
    fig, axes = plt.subplots(4, len(frames), figsize=(1.5 * len(frames) + 0.9, 5.6), squeeze=False)
    for c, name in enumerate(CHANNELS):
        vmin, vmax = _field_limits(name, [raw[c, idx]])
        for k, fi in enumerate(idx):
            im = _field_ax(axes[c, k], raw[c, fi], name, vmin, vmax)
        axes[c, 0].set_ylabel(CHANNEL_TEX[name], rotation=0, labelpad=12, color=INK, fontsize=11)
        fig.colorbar(im, ax=list(axes[c, :]), fraction=0.04, pad=0.02).outline.set_visible(False)
    for k, f in enumerate(frames):
        axes[0, k].set_title(f"t = {f}")
    print(f"caption: test sim {npz.stem}, gamma = {float(d['gamma']):.3f}")  # no suptitle, see vae-dit-frame
    _save(fig, out)


if __name__ == "__main__":
    app()
