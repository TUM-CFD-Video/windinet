#!/usr/bin/env python
"""Chapter 3/6 thesis figures from the eval outputs. Local, CPU only.

Inputs come from two eval scripts. Both use the standalone 256x256 test.h5
and pick the same gamma-spread sims:
  * scripts/eval_vae_test.py -> finetune_vae_outputs/sng_pvc/<run>/test_eval_256.json
    plus sample fields in $SCRATCH/windinet/figure_data/vae_test/<run>/<sim>.npz
  * scripts/eval_dit_vrmse.py --test_h5 -> logs/sng_pvc/eval_dit_vrmse/<job>/vrmse_summary.json
    plus sample fields in $SCRATCH/windinet/figure_data/dit/<arm>/<sim>.npz
Before running this, rsync the figure_data folders from the cluster.

Subcommands, one per figure:
  vae-bars     Ch6  test chmean per arm, one panel per group, SEM whiskers
  vae-channels Ch6  per-channel test VRMSE heatmap (arms x channels)
  vae-recon    Ch6  GT vs several VAEs for one sim and frame, plus |error| maps
  dit-bars     Ch6  VAE-only vs VAE+DiT test chmean per DiT arm (dumbbell)
  rollout      Ch6  GT / VAE+DiT / |error| rows per channel, frames as columns, one sim
  dataset      Ch3  the four raw fields of one sim at several frames

Colours follow the dataviz reference palette. Magnitude fields (density,
pressure) use a one-hue blue ramp. Signed fields (momentum) use a blue<->red
diverging map with a gray midpoint and symmetric limits. |error| maps use a
one-hue orange ramp. Categorical slots 1-2 (blue, orange) mark the two passes
in dit-bars.
"""

import json
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
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES_1 = "#2a78d6"  # blue
SERIES_2 = "#eb6834"  # orange
BLUE_RAMP = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
ORANGE_RAMP = ["#fcfcfb", "#fbe0d3", "#f5b597", "#eb6834", "#c24f1f", "#8a3413"]
DIVERGING = ["#0d366b", "#256abf", "#6da7ec", "#f0efec", "#ef8a89", "#e34948", "#8f1f1f"]
CMAP_SEQ = LinearSegmentedColormap.from_list("seq_blue", BLUE_RAMP)
CMAP_ERR = LinearSegmentedColormap.from_list("seq_orange", ORANGE_RAMP)
CMAP_DIV = LinearSegmentedColormap.from_list("div_blue_red", DIVERGING)

CHANNEL_TEX = {"density": r"$\rho$", "momentum_x": r"$m_x$", "momentum_y": r"$m_y$", "pressure": r"$p$"}
SIGNED = {"momentum_x", "momentum_y"}
CHANNELS = ["density", "momentum_x", "momentum_y", "pressure"]

# run folder (minus the finetune_vae_ch6_ prefix) -> (group, label)
ARMS = {
    "unfinetuned_256res": ("G1 adaptation", "Unfinetuned"),
    "color_adapter_nossim_256res": ("G1 adaptation", "Color adapter"),
    "decoder_only_nossim_256res": ("G1 adaptation", "Decoder-only"),
    "conv_in_only_256res": ("G1 adaptation", "Input + decoder"),
    "loss_rmse_h1_256res": ("G1 adaptation", "Full FT (baseline)"),
    "loss_rmse_only_256res": ("G2 loss", "RMSE"),
    "loss_rmse_h1_ssimfix_256res": ("G2 loss", "RMSE + H1 + SSIM"),
    "lr_1e5_nossim_256res": ("G3 learning rate", "LR 1e-5"),
    "loss_rmse_h1_128res": ("Training resolution", "Trained at 128²"),
    "kl_1e7_256res": ("G5 latent regularization", "KL 1e-7"),
    "kl_1e5_256res": ("G5 latent regularization", "KL 1e-5"),
    "sds_stock_w1e4_256res": ("G5 latent regularization", "SDS w=1e-4"),
    "sds_stock_w1e2_256res": ("G5 latent regularization", "SDS w=1e-2"),
    "sds_stock_w1_256res": ("G5 latent regularization", "SDS w=1"),
}
BASELINE = "loss_rmse_h1_256res"
GROUP_ORDER = ["G1 adaptation", "G2 loss", "G3 learning rate", "Training resolution", "G5 latent regularization"]


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


def _load_vae_reports(root: Path) -> dict[str, dict]:
    reports = {}
    for p in sorted(root.glob("finetune_vae_ch6_*/test_eval_256.json")):
        key = p.parent.name.removeprefix("finetune_vae_ch6_")
        if key in ARMS:
            reports[key] = json.loads(p.read_text())
    if not reports:
        raise SystemExit(f"no known test_eval_256.json under {root}")
    return reports


def _save(fig, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"wrote {out}")


@app.command("vae-bars")
def vae_bars(
    root: Path = typer.Option(Path("finetune_vae_outputs/sng_pvc")),
    out: Path = typer.Option(Path("figures/ch6_vae_ablation.pdf")),
) -> None:
    """Test chmean per arm, one horizontal-bar panel per experiment group."""
    _style()
    reports = _load_vae_reports(root)
    base = reports.get(BASELINE, {}).get("metrics", {}).get("vrmse_chmean")
    groups = [g for g in GROUP_ORDER if any(ARMS[k][0] == g for k in reports)]
    rows = {g: [k for k in ARMS if ARMS[k][0] == g and (k in reports or (k == BASELINE and base))] for g in groups}
    # the baseline row is repeated in every group as the reference bar
    for g in groups:
        if base is not None and BASELINE not in rows[g]:
            rows[g].append(BASELINE)
    heights = [len(rows[g]) for g in groups]
    fig, axes = plt.subplots(len(groups), 1, figsize=(5.2, 0.32 * sum(heights) + 0.55 * len(groups)),
                             gridspec_kw={"height_ratios": heights}, squeeze=False)
    for ax, g in zip(axes[:, 0], groups):
        keys = rows[g]
        vals = [reports[k]["metrics"]["vrmse_chmean"] for k in keys]
        sems = [reports[k]["metrics"]["vrmse_chmean_sem"] for k in keys]
        y = np.arange(len(keys))[::-1]
        colors = [SERIES_1 if k == BASELINE else "#9ec5f4" for k in keys]
        ax.barh(y, vals, height=0.62, color=colors, edgecolor="none")
        ax.errorbar(vals, y, xerr=sems, fmt="none", ecolor=INK_2, elinewidth=0.8, capsize=0)
        if base is not None:
            ax.axvline(base, color=MUTED, lw=0.8, ls=(0, (3, 2)), zorder=0)
        xmax = max(v + s for v, s in zip(vals, sems)) * 1.28
        for yi, v, se, k in zip(y, vals, sems, keys):
            delta = "" if base is None or k == BASELINE else f"  ({(v / base - 1):+.1%})"
            ax.text(v + se + xmax * 0.015, yi, f"{v:.4f}{delta}", va="center", ha="left", fontsize=7.5, color=INK_2)
        ax.set_yticks(y, [ARMS[k][1] for k in keys])
        ax.set_xlim(0, xmax)
        ax.set_title(g, loc="left", fontweight="bold")
        ax.tick_params(axis="y", length=0)
    axes[-1, 0].set_xlabel("Test VRMSE, mean over channels (lower is better)")
    fig.tight_layout(h_pad=0.8)
    _save(fig, out)


@app.command("vae-channels")
def vae_channels(
    root: Path = typer.Option(Path("finetune_vae_outputs/sng_pvc")),
    out: Path = typer.Option(Path("figures/ch6_vae_per_channel.pdf")),
) -> None:
    """Per-channel test VRMSE, arms x channels, log-scaled blue ramp, values printed in each cell."""
    from matplotlib.colors import LogNorm

    _style()
    reports = _load_vae_reports(root)
    keys = [k for g in GROUP_ORDER for k in ARMS if ARMS[k][0] == g and k in reports]
    data = np.array([[reports[k]["metrics"][f"vrmse_{c}"] for c in CHANNELS] for k in keys])
    fig, ax = plt.subplots(figsize=(4.6, 0.3 * len(keys) + 0.9))
    norm = LogNorm(vmin=data.min(), vmax=data.max())
    im = ax.imshow(data, cmap=CMAP_SEQ, norm=norm, aspect="auto")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            dark = norm(data[i, j]) > 0.55
            ax.text(j, i, f"{data[i, j]:.3f}", ha="center", va="center", fontsize=7,
                    color="#ffffff" if dark else INK)
    ax.set_xticks(range(4), [CHANNEL_TEX[c] for c in CHANNELS])
    ax.set_yticks(range(len(keys)), [ARMS[k][1] for k in keys])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.xaxis.tick_top()
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.set_label("Test VRMSE (log scale)", color=INK_2)
    cb.outline.set_visible(False)
    _save(fig, out)


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


@app.command("vae-recon")
def vae_recon(
    samples_root: Path = typer.Option(..., help="Local copy of figure_data/vae_test (one folder per run)"),
    sim: str = typer.Option(..., help="Sample id, e.g. one of test_eval_256.json's figure_samples.ids"),
    frame: int = typer.Option(50, help="0-indexed frame (must be one of the saved frames)"),
    runs: list[str] = typer.Option(
        ["loss_rmse_h1_256res", "decoder_only_nossim_256res", "color_adapter_nossim_256res"],
        help="Run keys (folder name minus finetune_vae_ch6_), repeat the option",
    ),
    out: Path = typer.Option(Path("figures/ch6_vae_recon.pdf")),
) -> None:
    """Rows = channels. Columns = GT, one reconstruction per run, then one |error| map per run."""
    _style()
    loaded = []
    for r in runs:
        d = np.load(samples_root / f"finetune_vae_ch6_{r}" / f"{sim}.npz")
        fi = list(d["frames"]).index(frame)
        mean, std, clip = d["channel_mean"], d["channel_std"], float(d["normalization_clip"])
        loaded.append((r, _denorm(d["gt_norm"], mean, std, clip)[:, fi], _denorm(d["recon_norm"], mean, std, clip)[:, fi]))
    gt = loaded[0][1]
    n = len(runs)
    fig, axes = plt.subplots(4, 1 + 2 * n, figsize=(1.35 * (1 + 2 * n) + 1.4, 5.4), squeeze=False,
                             gridspec_kw={"wspace": 0.08})
    for c, name in enumerate(CHANNELS):
        vmin, vmax = _field_limits(name, [gt[c]] + [rec[c] for _, _, rec in loaded])
        im = _field_ax(axes[c, 0], gt[c], name, vmin, vmax)
        errs = [np.abs(rec[c] - gt[c]) for _, _, rec in loaded]
        emax = float(np.percentile(np.concatenate([e.ravel() for e in errs]), 99.5)) or 1e-6
        for j, (_, _, rec) in enumerate(loaded):
            _field_ax(axes[c, 1 + j], rec[c], name, vmin, vmax)
            eim = _err_ax(axes[c, 1 + n + j], errs[j], emax)
        axes[c, 0].set_ylabel(CHANNEL_TEX[name], rotation=0, labelpad=12, color=INK, fontsize=11)
        fig.colorbar(im, ax=list(axes[c, : 1 + n]), fraction=0.02, pad=0.01, aspect=12).outline.set_visible(False)
        fig.colorbar(eim, ax=list(axes[c, 1 + n :]), fraction=0.03, pad=0.01, aspect=12).outline.set_visible(False)
    axes[0, 0].set_title("Ground truth")
    for j, r in enumerate(runs):
        label = ARMS.get(r, ("", r))[1]
        axes[0, 1 + j].set_title(label)
        axes[0, 1 + n + j].set_title(f"|error|\n{label}", color=INK_2)
    fig.suptitle(f"Test sim {sim}, frame {frame}", color=INK_2, fontsize=8, y=0.995)
    _save(fig, out)


def ax_pad(rows) -> float:
    return max(max(r[1], r[2]) for r in rows) * 0.02


@app.command("dit-bars")
def dit_bars(
    arms: list[str] = typer.Option(..., help="label=path/to/vrmse_summary.json, repeat per DiT arm (split on the last =)"),
    out: Path = typer.Option(Path("figures/ch6_vae_dit.pdf")),
) -> None:
    """Dumbbell: VAE-only (blue) vs VAE+DiT (orange) test chmean per arm."""
    _style()
    rows = []
    for spec in arms:
        label, path = spec.rsplit("=", 1)
        s = json.loads(Path(path).read_text())
        rows.append((label, s["vae_only_vrmse_chmean"], s["vae_dit_vrmse_chmean"], s.get("split", "?")))
    splits = {r[3] for r in rows}
    rows.sort(key=lambda r: r[2])
    y = np.arange(len(rows))[::-1]
    fig, ax = plt.subplots(figsize=(5.2, 0.38 * len(rows) + 0.9))
    for yi, (_, vo, vd, _) in zip(y, rows):
        ax.plot([vo, vd], [yi, yi], color=GRID, lw=2, zorder=1)
    ax.scatter([r[1] for r in rows], y, s=36, color=SERIES_1, edgecolor="#fcfcfb", linewidth=1.5, zorder=2, label="VAE only (reconstruction)")
    ax.scatter([r[2] for r in rows], y, s=36, color=SERIES_2, edgecolor="#fcfcfb", linewidth=1.5, zorder=3, label="VAE + DiT (rollout)")
    for yi, (_, vo, vd, _) in zip(y, rows):
        right = max(vo, vd)
        ax.text(right + ax_pad(rows), yi, f"{vd:.3f}", ha="left", va="center", fontsize=7, color=INK_2)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, max(max(r[1], r[2]) for r in rows) * 1.15)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel(f"VRMSE, mean over channels ({'/'.join(sorted(splits))} set, lower is better)")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=7.5, handletextpad=0.2, borderaxespad=0.2)
    _save(fig, out)


@app.command("rollout")
def rollout(
    npz: Path = typer.Option(..., help="figure_data/dit/<arm>/<sim>.npz"),
    frames: list[int] = typer.Option([25, 50, 100], help="0-indexed frames among the saved ones"),
    out: Path = typer.Option(Path("figures/ch6_rollout.pdf")),
) -> None:
    """Columns = frames. Per channel three rows: GT, VAE+DiT prediction, |error|; one colour bar per row."""
    _style()
    d = np.load(npz)
    saved = list(d["frames"])
    mean, std, clip = d["channel_mean"], d["channel_std"], float(d["normalization_clip"])
    gt = _denorm(d["gt_norm"], mean, std, clip)
    pred = _denorm(d["vae_dit_norm"], mean, std, clip)
    idx = [saved.index(f) for f in frames]
    nf = len(frames)
    fig, axes = plt.subplots(12, nf, figsize=(0.95 * nf + 1.6, 9.0), squeeze=False,
                             gridspec_kw={"hspace": 0.06, "wspace": 0.04})
    for c, name in enumerate(CHANNELS):
        vmin, vmax = _field_limits(name, [gt[c, idx], pred[c, idx]])
        err = np.abs(pred[c, idx] - gt[c, idx])
        emax = float(np.percentile(err, 99.5)) or 1e-6
        r = 3 * c
        for k, fi in enumerate(idx):
            im = _field_ax(axes[r, k], gt[c, fi], name, vmin, vmax)
            _field_ax(axes[r + 1, k], pred[c, fi], name, vmin, vmax)
            eim = _err_ax(axes[r + 2, k], np.abs(pred[c, fi] - gt[c, fi]), emax)
        for row, label in ((r, f"{CHANNEL_TEX[name]}  GT"), (r + 1, "VAE+DiT"), (r + 2, "|error|")):
            axes[row, 0].set_ylabel(label, rotation=0, ha="right", va="center", labelpad=4,
                                    color=INK if row == r else INK_2, fontsize=9 if row == r else 8)
        for row, mappable in ((r, im), (r + 1, im), (r + 2, eim)):
            cb = fig.colorbar(mappable, ax=list(axes[row, :]), fraction=0.04, pad=0.02, aspect=6)
            cb.outline.set_visible(False)
            cb.ax.tick_params(labelsize=6.5)
    for k, f in enumerate(frames):
        axes[0, k].set_title(f"t = {f}")
    fig.suptitle(f"{npz.stem}  (gamma = {float(d['gamma']):.3f}; frame 0 is the conditioning input)",
                 color=INK_2, fontsize=8, y=0.93)
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
    fig.suptitle(f"{npz.stem}  (gamma = {float(d['gamma']):.3f})", color=INK_2, fontsize=8, y=0.995)
    _save(fig, out)


if __name__ == "__main__":
    app()
