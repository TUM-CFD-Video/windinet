"""Statistics of every field of one HDF5 file over all its sims (data.stats_json): mean, std, min, max and
percentiles per spec field, and per log_<f> of each positive field."""

from __future__ import annotations

import json
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np
import typer

from windinet.data.frames import read_field, sim_ids
from windinet.data.specs import SPECS

BINS = 2**20
PERCENTILES = {"p0_1": 0.1, "p0_5": 0.5, "p1": 1.0, "p50": 50.0, "p99": 99.0, "p99_5": 99.5, "p99_9": 99.9}


def quantities(h5: str, dataset: str, sid: str) -> dict[str, np.ndarray]:
    """One sim's spec fields, flat float64, plus log_<f> of each field not itself a log that is positive here."""
    with h5py.File(h5, "r") as f:
        x = {name: read_field(f[sid], name, slice(None)).ravel() for name in SPECS[dataset].fields}
    for name in list(x):
        if not name.startswith("log_") and x[name].min() > 0:
            x[f"log_{name}"] = np.log(x[name])
    return x


def sums(task: tuple[str, str, str]) -> dict[str, tuple]:
    """Pass 1: n, sum, min, max per quantity."""
    return {k: (v.size, v.sum(), v.min(), v.max()) for k, v in quantities(*task).items()}


def spread(task: tuple[str, str, str], ranges: dict[str, tuple[float, float, float]]) -> dict[str, tuple]:
    """Pass 2: sum of squares around the global mean and a BINS-bin histogram over [min, max] per quantity."""
    x = quantities(*task)
    out = {}
    for k, (lo, hi, mean) in ranges.items():
        idx = np.minimum(((x[k] - lo) * (BINS / ((hi - lo) or 1.0))).astype(np.int64), BINS - 1)
        out[k] = (np.square(x[k] - mean).sum(), np.bincount(idx, minlength=BINS))
    return out


def main(
    dataset: str = typer.Argument(..., help=f"one of {', '.join(SPECS)}"),
    h5: Path = typer.Argument(..., help="the simulations"),
    out: Path = typer.Argument(..., help="output JSON"),
    workers: int = typer.Option(8, help="processes, one sim per task"),
    sims: int | None = typer.Option(None, help="only the first n sims (sorted ids), for checks"),
) -> None:
    """Write the stats JSON in the shape windinet.data.frames.load_field_stats reads."""
    if dataset not in SPECS:
        raise typer.BadParameter(f"unknown dataset {dataset!r}, expected one of {list(SPECS)}")
    ids = sim_ids(h5)[:sims]
    tasks = [(str(h5), dataset, sid) for sid in ids]
    with Pool(workers) as pool:
        first = pool.map(sums, tasks)
        # a log stays only if every sim is positive, i.e. its global min is > 0
        keys = [k for k in first[0] if all(k in r for r in first)]
        n = {k: sum(r[k][0] for r in first) for k in keys}
        mean = {k: sum(r[k][1] for r in first) / n[k] for k in keys}
        lo = {k: min(r[k][2] for r in first) for k in keys}
        hi = {k: max(r[k][3] for r in first) for k in keys}
        ss = dict.fromkeys(keys, 0.0)
        hist = {k: np.zeros(BINS, np.int64) for k in keys}
        ranges = {k: (lo[k], hi[k], mean[k]) for k in keys}
        for r in pool.imap_unordered(partial(spread, ranges=ranges), tasks):
            for k in keys:
                ss[k] += r[k][0]
                hist[k] += r[k][1]
    stats = {}
    for k in keys:
        cdf, width = np.cumsum(hist[k]), ((hi[k] - lo[k]) or 1.0) / BINS
        stats[k] = {
            "mean": float(mean[k]),
            "std": float(np.sqrt(ss[k] / n[k])),
            "min": float(lo[k]),
            "max": float(hi[k]),
        }
        for key, q in PERCENTILES.items():
            stats[k][key] = float(lo[k] + (np.searchsorted(cdf, q / 100 * n[k]) + 0.5) * width)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"source_h5": str(h5), "num_simulations": len(ids), "data_normalization_stats": stats}, indent=2)
    )


if __name__ == "__main__":
    typer.run(main)
