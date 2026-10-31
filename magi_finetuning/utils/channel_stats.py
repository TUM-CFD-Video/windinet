"""Normalization statistics from the TRAIN split only (meeting item 9), incl. log_density and log_pressure.

Writes a new yaml in the format of configs/finetune_vae/euler_mq_256_only_train.yaml (the old file is
left untouched: the first official run keeps using it). Every channel gets mean / std / min / max / count
for x, and log_density / log_pressure for log(x).

How (numerically safe): per simulation, in float64, count, mean and M2 = sum((x - mean)^2); simulations
are then merged with Chan et al.'s pairwise formula, which never subtracts two large sums. std is the
population std (ddof = 0), as everywhere else in this codebase.

The same per-simulation results are also merged over ALL of train.h5 (train + val), the population the
old yaml was most likely computed on, and the three versions are printed side by side.

Run on a compute node (reads ~all of train.h5):
    sbatch jobs/jupiter/magi1/channel_stats.sbatch
"""

import argparse
import os
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np
import yaml

from magi_finetuning.data import load_split
from magi_finetuning.settings import NORMALIZATION_STATS, REPO, SPLIT_FILE, TRAIN_DATA

RAW = ["density", "momentum_x", "momentum_y", "pressure"]
LOGGED = {"log_density": "density", "log_pressure": "pressure"}
NAMES = RAW + list(LOGGED)
OUTPUT = REPO / "configs" / "magi1" / "euler_mq_256_train_split_stats.yaml"

_file = None  # one open HDF5 file per worker process


def _open(path: str) -> None:
    global _file
    os.environ["HDF5_EXT_PREFIX"] = str(Path(path).parent)  # train.h5 links to per-gamma files next to it
    _file = h5py.File(path, "r")


def simulation_stats(sim_id: str) -> tuple[str, dict]:
    """(sim_id, {name: (count, mean, M2, min, max)}) for one simulation, all in float64."""
    group = _file[sim_id]
    values = {name: group[name][:].astype(np.float64).ravel() for name in RAW}
    for log_name, source in LOGGED.items():
        if values[source].min() <= 0:
            raise ValueError(f"{sim_id}/{source} has values <= 0; log undefined")
        values[log_name] = np.log(values[source])
    stats = {}
    for name, x in values.items():
        mean = x.mean()
        stats[name] = (x.size, mean, float(((x - mean) ** 2).sum()), float(x.min()), float(x.max()))
    return sim_id, stats


def merge(a: tuple, b: tuple) -> tuple:
    """Chan et al.: combine (count, mean, M2, min, max) of two disjoint sets."""
    n_a, mean_a, m2_a, min_a, max_a = a
    n_b, mean_b, m2_b, min_b, max_b = b
    n = n_a + n_b
    delta = mean_b - mean_a
    return n, mean_a + delta * n_b / n, m2_a + m2_b + delta**2 * n_a * n_b / n, min(min_a, min_b), max(max_a, max_b)


def combine(per_sim: dict, sim_ids) -> dict:
    """Merged stats over sim_ids, as yaml-ready {name: {mean, std, min, max, count}}."""
    result = {}
    for name in NAMES:
        total = None
        for sim_id in sim_ids:
            total = per_sim[sim_id][name] if total is None else merge(total, per_sim[sim_id][name])
        n, mean, m2, low, high = total
        result[name] = {"mean": float(mean), "std": float(np.sqrt(m2 / n)), "min": low, "max": high, "count": int(n)}
    return result


def print_comparison(old: dict, all_sims: dict, train: dict) -> None:
    print(f"\n{'channel':14s}{'stat':6s}{'old yaml':>16s}{'all train.h5':>16s}{'train split':>16s}   old vs all")
    for name in NAMES:
        for stat in ("mean", "std", "min", "max"):
            previous = old.get(name, {}).get(stat)
            agreement = (f"{abs(previous - all_sims[name][stat]) / max(abs(all_sims[name][stat]), 1e-12):.1e} rel"
                         if previous is not None else "(not in old yaml)")
            print(f"{name:14s}{stat:6s}{'' if previous is None else f'{previous:16.8f}':>16s}"
                  f"{all_sims[name][stat]:16.8f}{train[name][stat]:16.8f}   {agreement}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data", default=TRAIN_DATA)
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--out", default=str(OUTPUT))
    args = p.parse_args()

    split = load_split()
    train_ids, all_ids = split["train"], split["train"] + split["val"]
    with Pool(args.workers, initializer=_open, initargs=(args.data,)) as pool:
        per_sim = {}
        for count, (sim_id, stats) in enumerate(pool.imap_unordered(simulation_stats, all_ids, chunksize=4), 1):
            per_sim[sim_id] = stats
            if count % 250 == 0:
                print(f"{count}/{len(all_ids)} simulations", flush=True)

    train = combine(per_sim, train_ids)
    all_sims = combine(per_sim, all_ids)
    old = yaml.safe_load(open(NORMALIZATION_STATS))["data_normalization_stats"]
    print_comparison(old, all_sims, train)

    header = (f"# Normalization statistics of {args.data},\n"
              f"# TRAIN split only ({len(train_ids)} sims, fixed 90/10 split in {SPLIT_FILE.name}; val and test.h5 excluded).\n"
              "# Written by magi_finetuning/utils/channel_stats.py: float64, per-simulation mean/M2 merged with\n"
              "# Chan et al.'s formula, population std. log_* = stats of log(x).\n")
    Path(args.out).write_text(header + yaml.safe_dump({"data_normalization_stats": train}, sort_keys=False))
    print(f"\nwritten: {args.out}")


if __name__ == "__main__":
    main()
