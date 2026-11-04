"""Channel statistics for the standardized rollout videos (CFD_Rollout_Visualization_Spec, section 2): per channel
of train.h5 ground truth in physical units, pooled over all 4500 sims (train + val split: the dataset's training
file, as in its shared euler_mq_256_only_train.yaml), all 101 frames and all pixels:

    mean, std (population), min, max, p0_5 = Q_0.005, p99_5 = Q_0.995

One pass on one GPU: float64 sums for mean / std, and a 2^24-bin histogram per channel over [min, max] of the
shared yaml for the two quantiles (numpy's default "linear" definition, interpolated inside the bin; the error is
below one bin width, written to the file). mean / std / min / max are checked against the yaml.

    sbatch --time=02:00:00 jobs/jupiter/magi1/quick_test.sbatch -m magi_finetuning.utils.rollout_video_stats
        -> SCRATCH/magi1_outputs/rollout_video_stats.json (what every collaborator's videos use)
"""

import argparse
import json
import time
from pathlib import Path

import torch
import yaml

from magi_finetuning.settings import OUTPUTS, TRAIN_DATA
from windinet.training.shockwave_data import CHANNEL_NAMES, ShockWaveDataset

SHARED_YAML = Path(TRAIN_DATA).with_name("euler_mq_256_only_train.yaml")
BINS = 2 ** 24
QUANTILES = {"p0_5": 0.005, "p99_5": 0.995}
UNIT = "dimensionless"  # the Euler multi-quadrant fields are non-dimensional; the files carry no unit


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=TRAIN_DATA)
    p.add_argument("--out", default=str(OUTPUTS / "rollout_video_stats.json"))
    p.add_argument("--workers", type=int, default=16)
    return p.parse_args()


def quantile(counts: torch.Tensor, lo: float, hi: float, p: float) -> float:
    """numpy-style linear quantile (rank (n - 1) p) from a histogram over [lo, hi], values spread evenly per bin."""
    cumulative = counts.cumsum(0)
    rank = (cumulative[-1].item() - 1) * p
    k = int(torch.searchsorted(cumulative, torch.tensor(int(rank), device=counts.device), right=True).item())
    before = cumulative[k - 1].item() if k else 0
    width = (hi - lo) / BINS
    return lo + width * (k + (rank - before + 0.5) / counts[k].item())


@torch.no_grad()
def main():
    args = parse_args()
    shared = yaml.safe_load(SHARED_YAML.read_text())["data_normalization_stats"]
    ranges = torch.tensor([[shared[c]["min"], shared[c]["max"]] for c in CHANNEL_NAMES], dtype=torch.float64).cuda()
    dataset = ShockWaveDataset(args.data, num_sim_frames=101)
    loader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=args.workers, prefetch_factor=4)

    counts = torch.zeros(len(CHANNEL_NAMES), BINS, dtype=torch.int64, device="cuda")
    total = torch.zeros(len(CHANNEL_NAMES), dtype=torch.float64, device="cuda")
    total_sq = torch.zeros_like(total)
    low = torch.full_like(total, float("inf"))
    high = torch.full_like(total, -float("inf"))
    n, started = 0, time.perf_counter()
    for index, sample in enumerate(loader):
        for c, name in enumerate(CHANNEL_NAMES):
            x = sample[name].cuda().double().flatten()
            total[c] += x.sum()
            total_sq[c] += x.square().sum()
            low[c] = torch.minimum(low[c], x.min())
            high[c] = torch.maximum(high[c], x.max())
            lo, hi = ranges[c]
            bins = ((x - lo) / (hi - lo) * BINS).floor().long().clamp_(0, BINS - 1)
            counts[c] += torch.bincount(bins, minlength=BINS)
        n += sample[CHANNEL_NAMES[0]].numel()
        if index % 500 == 0:
            print(f"{index + 1}/{len(dataset)} sims, {time.perf_counter() - started:.0f} s", flush=True)

    stats, checks = {}, []
    for c, name in enumerate(CHANNEL_NAMES):
        mean = (total[c] / n).item()
        std = (total_sq[c] / n - mean ** 2).sqrt().item()
        lo, hi = ranges[c].tolist()
        entry = {"mean": mean, "std": std, "min": low[c].item(), "max": high[c].item(),
                 **{key: quantile(counts[c], lo, hi, p) for key, p in QUANTILES.items()},
                 "unit": UNIT, "quantile_precision": (hi - lo) / BINS}
        if not (entry["min"] <= entry["p0_5"] <= entry["p99_5"] <= entry["max"] and std > 0):
            raise ValueError(f"{name}: inconsistent statistics {entry}")
        stats[name] = entry
        for key in ("mean", "std", "min", "max"):
            relative = abs(entry[key] - shared[name][key]) / max(abs(shared[name][key]), 1e-12)
            checks.append(f"{name} {key}: {entry[key]:.9g} vs shared yaml {shared[name][key]:.9g} (rel diff {relative:.1e})")

    output = {"dataset": args.data, "sims": len(dataset), "frames_per_sim": 101, "values_per_channel": n,
              "pooled_over": "all sims of the training file (train + val split), all frames 0..100, all pixels",
              "space": "physical (raw dataset values)", "quantile_definition": "numpy linear, rank (n - 1) p",
              "shared_yaml": str(SHARED_YAML), "channels": stats}
    Path(args.out).write_text(json.dumps(output, indent=1))
    print("\n".join(checks))
    for name, entry in stats.items():
        print(f"{name:11s} " + "  ".join(f"{k} {entry[k]:.6g}" for k in ("mean", "std", "min", "p0_5", "p99_5", "max")))
    print(f"wrote {args.out} ({time.perf_counter() - started:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
