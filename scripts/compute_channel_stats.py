"""
Compute per-channel mean/std/min/max over an entire ShockWave HDF5 dataset.

Motivation: `configs/finetune_vae/euler_mq_128_only_train.yaml` carries a
hand-computed `data_normalization_stats` block (density/momentum_x/
momentum_y/pressure/log_density -- no log_pressure) that nothing in the
codebase actually loads or references (it isn't a valid VaeTrainerConfig,
just a standalone notes file) -- there's no way to tell whether it's still
accurate, was computed on the same data cut, or used the same std
convention as the rest of this codebase without recomputing it directly.
This script does that recomputation, plus the log_pressure that file never
had, from the real HDF5 file.

This project's "train"/"val" split (VaeDataConfig.eval_sims) is a runtime
random split of ONE HDF5 file via torch.Generator, not separate files --
so processing every simulation in the given --h5 path already covers both
halves; there's no separate val.h5 to also read.

Streams one simulation at a time (running sum/sum-of-squares/min/max)
rather than concatenating the whole dataset into memory, so this scales to
the 256x256_ds / 512x512_orig resolutions too, not just 128x128_ds
(4x / 16x the pixels per frame respectively) -- pass any of the three via
--h5 unchanged.

Usage:
    python scripts/compute_channel_stats.py euler_mq_dataset/128x128_ds/train.h5
    python scripts/compute_channel_stats.py euler_mq_dataset/256x256_ds/train.h5 \
        --output euler_mq_dataset/256x256_ds/channel_stats.json

    # rollout-video statistics (docs/video_spec.md), percentiles in the same single pass:
    python scripts/compute_channel_stats.py euler_mq_dataset/256x256_ds/train.h5 \
        --video-stats configs/video/euler_mq_256_video_stats.json \
        --hist-range-from configs/finetune_vae/euler_mq_256_only_train.yaml
"""

import json
import math
from pathlib import Path

import h5py
import numpy as np
import typer
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table

console = Console()
app = typer.Typer(pretty_exceptions_enable=False, no_args_is_help=True)

RAW_CHANNELS = ["density", "momentum_x", "momentum_y", "pressure"]
LOG_CHANNELS = {"log_density": "density", "log_pressure": "pressure"}
ALL_STATS_NAMES = RAW_CHANNELS + list(LOG_CHANNELS)


class RunningStats:
    """Streaming mean/std/min/max (population std, ddof=0 -- matches this
    codebase's variance convention elsewhere, e.g. windinet.losses.vrms's
    `unbiased=False`), accumulated one simulation's worth of samples at a time.
    """

    def __init__(self) -> None:
        self.count = 0
        self.sum = 0.0
        self.sumsq = 0.0
        self.min = math.inf
        self.max = -math.inf

    def update(self, values: np.ndarray) -> None:
        flat = values.reshape(-1).astype(np.float64)
        self.count += flat.size
        self.sum += float(flat.sum())
        self.sumsq += float(np.square(flat).sum())
        self.min = min(self.min, float(flat.min()))
        self.max = max(self.max, float(flat.max()))

    def finalize(self) -> dict[str, float]:
        mean = self.sum / self.count
        variance = self.sumsq / self.count - mean**2
        return {
            "mean": mean,
            "std": math.sqrt(max(variance, 0.0)),
            "min": self.min,
            "max": self.max,
            "count": self.count,
        }


class RunningHistogram:
    """Fine fixed-range histogram for streaming quantiles (second pass, once the
    exact min/max from RunningStats are known). With 2**20 bins the quantile is
    off by at most (max - min) / 2**20 -- ~6e-5 for momentum's ±30 range, far
    below the 0.5σ the video spec's percentiles are read at."""

    def __init__(self, lo: float, hi: float, bins: int = 2**20) -> None:
        self.lo, self.hi, self.bins = lo, hi, bins
        self.counts = np.zeros(bins, dtype=np.int64)

    def update(self, values: np.ndarray) -> None:
        flat = values.reshape(-1).astype(np.float64)
        idx = ((flat - self.lo) / (self.hi - self.lo) * self.bins).astype(np.int64)
        self.counts += np.bincount(np.clip(idx, 0, self.bins - 1), minlength=self.bins)

    def quantile(self, p: float) -> float:
        cum = np.cumsum(self.counts)
        target = p * cum[-1]
        i = int(np.searchsorted(cum, target))
        below = cum[i - 1] if i > 0 else 0
        frac = (target - below) / max(self.counts[i], 1)  # linear within the bin
        return float(self.lo + (i + frac) * (self.hi - self.lo) / self.bins)


# Video spec (docs/video_spec.md): channels drawn diverging around physical 0.
SIGNED_CHANNELS = {"momentum_x", "momentum_y"}


def scan(f, ids: list[str], description: str, visit) -> None:
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(description, total=len(ids))
        for sid in ids:
            visit(f[sid])
            progress.advance(task)


@app.command()
def main(
    h5_path: Path = typer.Argument(..., help="Path to the ShockWave HDF5 file (e.g. .../128x128_ds/train.h5)"),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Write results as JSON here, shaped like euler_mq_128_only_train.yaml's data_normalization_stats"
    ),
    limit: int | None = typer.Option(
        None,
        "--limit",
        help=(
            "Only process a random sample of N simulations (quick look, e.g. on a "
            "not-yet-fully-downloaded 256x256_ds) -- NOT the first N by sorted id, "
            "since sample ids may be grouped by gamma/generation batch and an "
            "unshuffled prefix can silently be a biased, non-representative slice"
        ),
    ),
    limit_seed: int = typer.Option(42, "--limit-seed", help="Seed for the --limit random sample, for reproducibility"),
    compare_to: Path | None = typer.Option(
        None,
        "--compare-to",
        help="A YAML file with a data_normalization_stats block (e.g. configs/finetune_vae/euler_mq_128_only_train.yaml) to diff the computed stats against",
    ),
    video_stats: Path | None = typer.Option(
        None,
        "--video-stats",
        help=(
            "Also write the rollout-video statistics (docs/video_spec.md, windinet/video.py's "
            "{channel: {mean, std, min, max, p0_5, p99_5, unit, signed}}) for the raw channels here. "
            "Costs a second pass over the file for the percentiles."
        ),
    ),
    hist_range_from: Path | None = typer.Option(
        None,
        "--hist-range-from",
        help=(
            "--video-stats in ONE pass: a YAML with a data_normalization_stats block whose min/max for this "
            "same file (e.g. configs/finetune_vae/euler_mq_256_only_train.yaml) set the histogram range up front, "
            "instead of a second pass after the min/max are known. Checked against the computed min/max."
        ),
    ),
    unit: str = typer.Option("", "--unit", help="--video-stats: the channels' unit (The Well euler: dimensionless)"),
) -> None:
    """Stream every simulation in --h5 and report mean/std/min/max per channel."""
    import os

    os.environ["HDF5_EXT_PREFIX"] = str(h5_path.parent)
    with h5py.File(h5_path, "r") as f:
        ids = sorted(f.keys())
        if limit is not None:
            # Random sample, not ids[:limit] -- sample ids may be grouped by
            # gamma/generation batch, so an unshuffled prefix can silently be
            # a biased, non-representative slice (seen in practice: a --limit
            # 50 run landed ~15-35% high on std across every raw channel).
            rng = np.random.default_rng(limit_seed)
            ids = list(rng.choice(ids, size=min(limit, len(ids)), replace=False))
        console.print(f"[cyan]{len(ids)} simulations in {h5_path}[/cyan]")

        stats = {name: RunningStats() for name in ALL_STATS_NAMES}

        hists = {}
        if video_stats is not None and hist_range_from is not None:
            import yaml

            ranges = yaml.safe_load(hist_range_from.read_text())["data_normalization_stats"]
            hists = {name: RunningHistogram(ranges[name]["min"], ranges[name]["max"]) for name in RAW_CHANNELS}

        def first_pass(group) -> None:
            raw_values = {}
            for name in RAW_CHANNELS:
                values = group[name][:]
                raw_values[name] = values
                stats[name].update(values)
                if name in hists:
                    hists[name].update(values)
            for log_name, source in LOG_CHANNELS.items():
                source_values = raw_values[source]
                if np.any(source_values <= 0):
                    raise ValueError(
                        f"{group.name}/{source} has non-positive values (min={source_values.min()}); "
                        f"log({source}) is undefined there"
                    )
                stats[log_name].update(np.log(source_values))

        scan(f, ids, "Scanning simulations", first_pass)

        percentiles = {}
        if video_stats is not None and hists:
            for name, h in hists.items():  # values beyond the given range would pile up in the edge bins
                if stats[name].min < h.lo or stats[name].max > h.hi:
                    raise ValueError(
                        f"{name}: data range [{stats[name].min}, {stats[name].max}] exceeds --hist-range-from's "
                        f"[{h.lo}, {h.hi}]: drop --hist-range-from for an exact two-pass run"
                    )
        elif video_stats is not None:
            hists = {name: RunningHistogram(stats[name].min, stats[name].max) for name in RAW_CHANNELS}

            def second_pass(group) -> None:
                for name in RAW_CHANNELS:
                    hists[name].update(group[name][:])

            scan(f, ids, "Percentiles (2nd pass)", second_pass)
        if video_stats is not None:
            percentiles = {n: {"p0_5": h.quantile(0.005), "p99_5": h.quantile(0.995)} for n, h in hists.items()}

    results = {name: stat.finalize() for name, stat in stats.items()}

    table = Table(title=f"Channel statistics -- {h5_path}")
    for col in ("channel", "mean", "std", "min", "max", "count"):
        table.add_column(col)
    for name in ALL_STATS_NAMES:
        r = results[name]
        table.add_row(name, f"{r['mean']:.6f}", f"{r['std']:.6f}", f"{r['min']:.6f}", f"{r['max']:.6f}", f"{r['count']:,}")
    console.print(table)

    if compare_to is not None:
        import yaml

        with open(compare_to) as fh:
            reference = yaml.safe_load(fh).get("data_normalization_stats", {})
        diff_table = Table(title=f"Diff vs {compare_to}")
        for col in ("channel", "stat", "computed", "reference", "abs diff", "rel diff"):
            diff_table.add_column(col)
        for name in ALL_STATS_NAMES:
            ref = reference.get(name)
            if ref is None:
                diff_table.add_row(name, "-", "-", "MISSING", "-", "-")
                continue
            for stat_name in ("mean", "std", "min", "max"):
                computed = results[name][stat_name]
                reference_value = ref[stat_name]
                abs_diff = abs(computed - reference_value)
                rel_diff = abs_diff / abs(reference_value) if reference_value != 0 else float("inf")
                diff_table.add_row(
                    name, stat_name, f"{computed:.6f}", f"{reference_value:.6f}", f"{abs_diff:.2e}", f"{rel_diff:.2%}"
                )
        console.print(diff_table)

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as fh:
            json.dump(
                {
                    "source_h5": str(h5_path),
                    "num_simulations": len(ids),
                    "data_normalization_stats": {
                        name: {k: v for k, v in r.items() if k != "count"} for name, r in results.items()
                    },
                },
                fh,
                indent=2,
            )
        console.print(f"[green]Wrote {output}[/green]")

    if video_stats is not None:
        video = {}
        for name in RAW_CHANNELS:
            r = results[name]
            video[name] = {k: r[k] for k in ("mean", "std", "min", "max")} | percentiles[name]
            video[name] |= {"unit": unit, "signed": name in SIGNED_CHANNELS}
            if not (r["min"] <= video[name]["p0_5"] <= video[name]["p99_5"] <= r["max"]):
                raise ValueError(f"{name}: percentiles outside the extrema: {video[name]}")
        pct = Table(title="Video statistics (percentiles)")
        for col in ("channel", "p0_5", "p99_5", "signed"):
            pct.add_column(col)
        for name, v in video.items():
            pct.add_row(name, f"{v['p0_5']:.6g}", f"{v['p99_5']:.6g}", str(v["signed"]))
        console.print(pct)
        video_stats.parent.mkdir(parents=True, exist_ok=True)
        video_stats.write_text(json.dumps(video, indent=2))
        console.print(f"[green]Wrote {video_stats}[/green]")


if __name__ == "__main__":
    app()
