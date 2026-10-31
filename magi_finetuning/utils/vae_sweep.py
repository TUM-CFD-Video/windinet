"""Randomized hyperparameter sweep for the MAGI-1 VAE: sample configurations, then collect the result matrix.

Every run is a short COMPLETE training (5 epochs = 633 steps with its own warmup and cosine decay), so final
scores are comparable across runs. Two anchors are always included: the official recipe and experiment E.
Standard library only, so it runs with plain python3 on the login node (no torch, no MAGI imports).

    python3 magi_finetuning/utils/vae_sweep.py sample  --name <sweep> --n 10 [--seed 0]   # what to submit
    python3 magi_finetuning/utils/vae_sweep.py collect --name <sweep>                      # results matrix
    <venv python> magi_finetuning/utils/vae_sweep.py plot --name <sweep>                    # curves.png (matplotlib)
jobs/jupiter/magi1/magi1_vae_sweep.sh does both (one 4-GPU job per run, a collect job when all are done).

Outputs: $MAGI1_OUTPUTS/vae_sweep/<sweep>/
    configs.json                    the sampled runs (name, hyperparameters, train_vae arguments)
    <run>/args.yaml, val_history.jsonl   written by train_vae.py
    results.md, results.csv         the matrix (per field at each run's best step) + "where it leads" summary
    curves.png                      val VRMSE over training, one panel per run (ranked), per field + mean
    params.png                      one panel per hyperparameter: each run's best mean and its pressure there
"""

from __future__ import annotations  # login-node python3 is 3.9

import argparse
import csv
import json
import math
import os
import random
from pathlib import Path

OUTPUTS = Path(os.environ.get("MAGI1_OUTPUTS", "/e/scratch/e-dev-2026d09-262/ig_work/magi1_outputs"))
FIELDS = ["density", "momentum_x", "momentum_y", "pressure"]

# The search space. lr is log-uniform; everything else is a uniform choice.
SPACE = {
    "lr": (1e-5, 3e-4),
    "io_lr_mult": [1, 5, 20, 50],
    "recon_loss": ["rmse", "vrmse_per_channel", "l1"],
    "h1_weight": [0, 10, 50, 200],
    "sds_weight": [0, 0.01],
    "decoder_extra_channels": [0, 16, 32, 64],
    "decoder_refine_channels": [0, 32, 64, 128],
    "init": ["zeros", "random"],
}
ANCHORS = {
    "anchor_official": dict(lr=5e-5, io_lr_mult=1, recon_loss="rmse", h1_weight=50, sds_weight=0.01,
                            decoder_extra_channels=0, decoder_refine_channels=0, init="zeros"),
    "anchor_E": dict(lr=5e-5, io_lr_mult=20, recon_loss="vrmse_per_channel", h1_weight=50, sds_weight=0.01,
                     decoder_extra_channels=16, decoder_refine_channels=64, init="zeros"),
}
FIXED_ARGS = "--epochs 5 --warmup 50"  # same short schedule for every run
LOSS_SHORT = {"rmse": "rmse", "vrmse_per_channel": "vrmse", "l1": "l1"}


def sample_config(rng: random.Random) -> dict:
    low, high = SPACE["lr"]
    config = {"lr": float(f"{math.exp(rng.uniform(math.log(low), math.log(high))):.1e}")}
    for key, choices in SPACE.items():
        if key != "lr":
            config[key] = rng.choice(choices)
    return config


def run_name(index: str, c: dict) -> str:
    """The name says exactly what the run is, e.g. 03_lr1.2e-04_io20_vrmse_h1-50_sds0.01_x32_r64_zeros."""
    return (f"{index}_lr{c['lr']:.1e}_io{c['io_lr_mult']}_{LOSS_SHORT[c['recon_loss']]}_h1-{c['h1_weight']}"
            f"_sds{c['sds_weight']}_x{c['decoder_extra_channels']}_r{c['decoder_refine_channels']}_{c['init']}")


def train_args(c: dict) -> str:
    return (f"{FIXED_ARGS} --lr {c['lr']} --io-lr-mult {c['io_lr_mult']} --recon-loss {c['recon_loss']} "
            f"--h1-weight {c['h1_weight']} --sds-weight {c['sds_weight']} "
            f"--decoder-extra-channels {c['decoder_extra_channels']} "
            f"--decoder-refine-channels {c['decoder_refine_channels']} --init {c['init']}")


def sample(name: str, n: int, seed: int) -> None:
    """Writes configs.json and prints one 'run<TAB>train_vae args' line per run for the submit script."""
    rng = random.Random(seed)
    runs = [{"name": key, **cfg} for key, cfg in ANCHORS.items()]
    runs += [{"name": run_name(f"{i:02d}", cfg), **cfg} for i, cfg in enumerate((sample_config(rng) for _ in range(n)), 1)]
    folder = OUTPUTS / "vae_sweep" / name
    folder.mkdir(parents=True, exist_ok=True)
    for run in runs:
        run["args"] = train_args(run)
    (folder / "configs.json").write_text(json.dumps({"seed": seed, "fixed": FIXED_ARGS, "space": SPACE, "runs": runs}, indent=1))
    for run in runs:
        print(f"{run['name']}\t{run['args']}")


def best_of(history_file: Path) -> tuple[dict | None, dict | None, int]:
    """(best validation, last validation, number of validations) from a run's val_history.jsonl."""
    if not history_file.is_file():
        return None, None, 0
    records = [json.loads(line) for line in history_file.read_text().splitlines() if line.strip()]
    if not records:
        return None, None, 0
    return min(records, key=lambda r: r["mean"]), records[-1], len(records)


def collect(name: str) -> None:
    folder = OUTPUTS / "vae_sweep" / name
    runs = json.loads((folder / "configs.json").read_text())["runs"]
    rows = []
    for run in runs:
        best, last, count = best_of(folder / run["name"] / "val_history.jsonl")
        row = {k: run[k] for k in ["name", *SPACE]}
        row.update(best_mean=best and best["mean"], best_step=best and best["step"], last_step=last and last["step"],
                   validations=count, **{f: best and best[f] for f in FIELDS})
        rows.append(row)
    done = sorted((r for r in rows if r["best_mean"] is not None), key=lambda r: r["best_mean"])
    missing = [r["name"] for r in rows if r["best_mean"] is None]

    lines = [f"# VAE sweep `{name}`", "",
             f"Each run: {json.loads((folder / 'configs.json').read_text())['fixed']} (633 steps, 4 GH200), val VRMSE on "
             "90 val sims (normalized, per field), at the run's best validation step. Target: < 0.08 mean.", "",
             "| rank | run | best mean | density | mom_x | mom_y | pressure | step |", "|---|---|---|---|---|---|---|---|"]
    for rank, r in enumerate(done, 1):
        lines.append(f"| {rank} | `{r['name']}` | **{r['best_mean']:.4f}** | " +
                     " | ".join(f"{r[f]:.3f}" for f in FIELDS) + f" | {r['best_step']}/{r['last_step']} |")
    if missing:
        lines += ["", "No validation yet (still running, or failed -- check its log): " + ", ".join(f"`{m}`" for m in missing)]

    lines += ["", "## Where it leads: mean of the runs' best val VRMSE per hyperparameter value", "",
              "| hyperparameter | value: average (runs) |", "|---|---|"]
    for key in SPACE:
        if key == "lr":
            buckets = {"< 3e-5": [], "3e-5 .. 1e-4": [], "> 1e-4": []}
            for r in done:
                buckets["< 3e-5" if r["lr"] < 3e-5 else "> 1e-4" if r["lr"] > 1e-4 else "3e-5 .. 1e-4"].append(r["best_mean"])
        else:
            buckets = {str(v): [r["best_mean"] for r in done if r[key] == v] for v in SPACE[key]}
        cells = [f"{value}: {sum(v) / len(v):.3f} ({len(v)})" for value, v in buckets.items() if v]
        lines.append(f"| {key} | " + " ; ".join(cells) + " |")
    lines += ["", "With ~10 random runs these averages are rough indications, not effects; read them together with "
                  "the per-field columns (a low mean can hide one bad field)."]
    (folder / "results.md").write_text("\n".join(lines) + "\n")
    with open(folder / "results.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("\n".join(lines))
    print(f"\nwritten: {folder / 'results.md'}")


# Field colours: the first four slots of the dataviz reference palette (validated for line charts); the mean is
# drawn in primary ink. A field keeps its colour in every panel.
FIELD_COLOURS = {"density": "#2a78d6", "momentum_x": "#eb6834", "momentum_y": "#1baf7a", "pressure": "#eda100"}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def plot(name: str) -> None:
    """curves.png: val VRMSE vs step, one small panel per run ordered by best mean, shared log y-axis,
    dashed target line at 0.08."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder = OUTPUTS / "vae_sweep" / name
    runs = json.loads((folder / "configs.json").read_text())["runs"]
    curves = []
    for run in runs:
        path = folder / run["name"] / "val_history.jsonl"
        records = [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.is_file() else []
        if records:
            curves.append((min(r["mean"] for r in records), run["name"], records))
    curves.sort()

    columns = 4
    rows = -(-len(curves) // columns)
    fig, axes = plt.subplots(rows, columns, figsize=(4.2 * columns, 3.3 * rows), sharex=True, sharey=True,
                             facecolor=SURFACE, constrained_layout=True)
    for index, ax in enumerate(axes.flat):
        if index >= len(curves):
            ax.set_visible(False)
            continue
        best, run, records = curves[index]
        steps = [r["step"] for r in records]
        ax.set_facecolor(SURFACE)
        for field, colour in FIELD_COLOURS.items():
            ax.plot(steps, [r[field] for r in records], color=colour, lw=1.6)
        ax.plot(steps, [r["mean"] for r in records], color=INK, lw=2.6)
        ax.axhline(0.08, color=MUTED, lw=1, ls="--")
        ax.set_yscale("log")
        ax.set_ylim(0.05, 3)
        ax.grid(axis="y", color=GRID, lw=0.8, which="both")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=8)
        label = run.split("_", 1)[1].replace("_", " ") if run[:2].isdigit() else run.replace("_", " ")
        ax.set_title(f"#{index + 1}  best mean {best:.3f}\n{label}", fontsize=8.5, color=INK, loc="left")
        if index == 0:  # the legend names the fields; only the target line is labelled in place
            ax.annotate("target 0.08", (steps[0], 0.08), xytext=(0, -9), textcoords="offset points",
                        fontsize=7.5, color=MUTED)
    for ax in axes[-1]:
        ax.set_xlabel("training step", fontsize=8.5, color=MUTED)
    for ax in axes[:, 0]:
        ax.set_ylabel("val VRMSE (log)", fontsize=8.5, color=MUTED)
    handles = [plt.Line2D([], [], color=c, lw=1.6, label=f) for f, c in FIELD_COLOURS.items()]
    handles.append(plt.Line2D([], [], color=INK, lw=2.6, label="mean"))
    handles.append(plt.Line2D([], [], color=MUTED, lw=1, ls="--", label="target 0.08"))
    fig.legend(handles=handles, loc="outside lower center", ncol=6, frameon=False, fontsize=9)
    fig.suptitle(f"VAE sweep {name}: validation VRMSE per field during training (ranked by best mean)",
                 fontsize=11, color=INK)
    fig.savefig(folder / "curves.png", dpi=140, facecolor=SURFACE)
    print(f"written: {folder / 'curves.png'}")
    plot_parameters(name, folder, runs, plt)


def plot_parameters(name: str, folder: Path, runs: list[dict], plt) -> None:
    """params.png: one panel per hyperparameter; x = its value, y = each run's best mean VRMSE (ink dot) and the
    pressure VRMSE at that step (open marker), log scale; anchors labelled. One parameter at a time: with ~10
    runs, pairs of parameters would leave most combinations empty."""
    import matplotlib.ticker

    points = []
    for run in runs:
        best, _, _ = best_of(folder / run["name"] / "val_history.jsonl")
        if best:
            points.append((run, best))
    keys = list(SPACE)
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.2), sharey=True, facecolor=SURFACE, constrained_layout=True)
    for ax, key in zip(axes.flat, keys):
        ax.set_facecolor(SURFACE)
        if key == "lr":
            xs = [r[key] for r, _ in points]
            ax.set_xscale("log")
            ax.set_xticks([1e-5, 3e-5, 1e-4, 3e-4], ["1e-5", "3e-5", "1e-4", "3e-4"])
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        else:
            order = [str(v) for v in SPACE[key]]
            xs = [order.index(str(r[key])) for r, _ in points]
            ax.set_xticks(range(len(order)), order)
            ax.set_xlim(-0.6, len(order) - 0.4)
        jitter = [0.0 if key == "lr" else (i % 3 - 1) * 0.08 for i in range(len(points))]
        x_plot = [x * (1 + j / 4) if key == "lr" else x + j for x, j in zip(xs, jitter)]
        ax.scatter(x_plot, [b["pressure"] for _, b in points], s=46, facecolors="none",
                   edgecolors=FIELD_COLOURS["pressure"], linewidths=1.6, zorder=2)
        ax.scatter(x_plot, [b["mean"] for _, b in points], s=46, color=INK, zorder=3,
                   edgecolors=SURFACE, linewidths=1.5)
        for (run, b), x in zip(points, x_plot):
            if run["name"].startswith("anchor"):
                ax.annotate(run["name"].replace("anchor_", ""), (x, b["mean"]), xytext=(6, 0),
                            textcoords="offset points", fontsize=7.5, color=MUTED, va="center")
        ax.axhline(0.08, color=MUTED, lw=1, ls="--")
        ax.set_yscale("log")
        ax.set_ylim(0.05, 3)
        ax.grid(axis="y", color=GRID, lw=0.8, which="both")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.set_title(key, fontsize=10, color=INK, loc="left")
    for ax in axes[:, 0]:
        ax.set_ylabel("best val VRMSE (log)", fontsize=9, color=MUTED)
    handles = [plt.Line2D([], [], marker="o", ls="", color=INK, label="mean (best step)"),
               plt.Line2D([], [], marker="o", ls="", markerfacecolor="none", markeredgecolor=FIELD_COLOURS["pressure"],
                          label="pressure at that step"),
               plt.Line2D([], [], color=MUTED, lw=1, ls="--", label="target 0.08")]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle(f"VAE sweep {name}: each run's result against each hyperparameter (one dot per run)",
                 fontsize=11, color=INK)
    fig.savefig(folder / "params.png", dpi=140, facecolor=SURFACE)
    print(f"written: {folder / 'params.png'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("command", choices=["sample", "collect", "plot"])
    p.add_argument("--name", required=True, help="sweep name (folder under $MAGI1_OUTPUTS/vae_sweep/)")
    p.add_argument("--n", type=int, default=10, help="random runs (sample), besides the 2 anchors")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    {"sample": lambda: sample(a.name, a.n, a.seed), "collect": lambda: collect(a.name), "plot": lambda: plot(a.name)}[a.command]()
