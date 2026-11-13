"""One-off diagnostics on saved rollouts (evaluate.py --save-rollouts), normalized frames 1..F:
1. Seed spread: the same model sampled with K seeds. The simulation is deterministic, so the spread between
   seeds is pure error. E[mse(sample, truth)] = bias^2 + variance, per field, relative to the truth's variance.
2. Entropy rule: p / density^gamma never falls below its frame-0 minimum in the truth. Share of generated pixels
   that break it, and the VRMSE after raising the pressure to that floor.
Run from the repo root: python -m magi_finetuning.eval.spread_and_entropy --seed-runs <dirs> --entropy-runs <dirs> --out <dir>"""
import argparse
import json
import traceback
from pathlib import Path

import torch
import yaml

from magi_finetuning.data import Normalization
from magi_finetuning.eval.evaluate import load_physical_fields
from windinet.training.shockwave_data import ShockWaveDataset, parse_gamma

DIMS = (0, 2, 3, 4)  # everything but the field axis of (1, 4, F, H, W)


def sims_of(run):
    return {p.stem for p in (run / "rollouts").glob("*.pt")}


def load(run, sim):
    return torch.load(run / "rollouts" / f"{sim}.pt").cuda().float().clamp(-1, 1)  # (1, 4, F, H, W)


def mean(values):
    return sum(values) / len(values)


def decompose(samples, truth):
    """samples (K, 1, 4, F, H, W), truth (1, 4, F, H, W) -> per field: E mse(sample, truth), unbiased seed variance,
    mse(seed mean, truth)."""
    k = samples.shape[0]
    center = samples.mean(dim=0)
    single = torch.stack([(s - truth).pow(2).mean(dim=DIMS) for s in samples]).mean(dim=0)
    variance = (samples - center).pow(2).sum(dim=0).mean(dim=DIMS) / (k - 1)
    ensemble = (center - truth).pow(2).mean(dim=DIMS)
    return single, variance, ensemble


@torch.no_grad()
def seed_spread(runs, normalization, test_set):
    sims = sorted(set.intersection(*(sims_of(run) for run in runs)))
    rows = []
    for sim in sims:
        truth = normalization.apply(load_physical_fields(test_set, sim))[:, :, 1:]
        scale = truth.var(dim=DIMS)
        samples = torch.stack([load(run, sim) for run in runs])
        single, variance, ensemble = decompose(samples, truth)
        frames = truth.shape[2]
        quarters = []
        for q in range(4):
            part = slice(q * frames // 4, (q + 1) * frames // 4)
            s, v, _ = decompose(samples[:, :, :, part], truth[:, :, part])
            quarters.append((v / s).mean().item())
        rows.append({
            "sim_id": sim,
            "vrmse_per_seed": [((s - truth).pow(2).mean(dim=DIMS) / scale).sqrt().mean().item() for s in samples],
            "vrmse_seed_mean": (ensemble / scale).sqrt().mean().item(),
            "spread": (variance / scale).sqrt().mean().item(),
            "variance_share": (variance / single).mean().item(),
            "variance_share_per_field": (variance / single).tolist(),
            "variance_share_by_quarter": quarters,
        })
        print(f"[spread] {sim}", flush=True)
    return rows


@torch.no_grad()
def entropy_rule(runs, normalization, test_set):
    result = {}
    for run in runs:
        rows = []
        for sim in sorted(sims_of(run)):
            physical = load_physical_fields(test_set, sim)  # (1, 4, 1 + F, H, W): density, momentum x, y, pressure
            gamma = parse_gamma(sim)
            lowest = (physical[:, 3, 0] / physical[:, 0, 0].pow(gamma)).min()  # frame-0 minimum of p / density^gamma
            truth = normalization.apply(physical)[:, :, 1:]
            scale = truth.var(dim=DIMS)
            prediction = load(run, sim)
            fields = normalization.invert(prediction)
            floor = lowest * fields[:, 0].clamp_min(1e-6).pow(gamma)
            ratio = fields[:, 3] / floor
            fixed = fields.clone()
            fixed[:, 3] = torch.maximum(fields[:, 3], floor)
            error = lambda x: ((x - truth).pow(2).mean(dim=DIMS) / scale).sqrt()
            before, after = error(prediction), error(normalization.apply(fixed))
            truth_ratio = physical[:, 3, 1:] / (lowest * physical[:, 0, 1:].pow(gamma))
            rows.append({
                "sim_id": sim,
                "below_1pct": (ratio < 0.99).float().mean().item(),
                "below_5pct": (ratio < 0.95).float().mean().item(),
                "below_20pct": (ratio < 0.80).float().mean().item(),
                "lowest": ratio.min().item(),
                "truth_below_1pct": (truth_ratio < 0.99).float().mean().item(),
                "vrmse_before": before.mean().item(), "vrmse_after": after.mean().item(),
                "pressure_before": before[3].item(), "pressure_after": after[3].item(),
            })
        result[run.name] = rows
        print(f"[entropy] {run.name}: {len(rows)} sims", flush=True)
    return result


def report(spread_rows, seed_runs, entropy_rows):
    lines = ["# Seed spread and entropy rule on saved rollouts", ""]
    if spread_rows:
        per_seed = [mean([row["vrmse_per_seed"][k] for row in spread_rows]) for k in range(len(seed_runs))]
        lines += [f"## Seed spread ({len(seed_runs)} seeds, {len(spread_rows)} sims, all frames)", "",
                  "Runs: " + ", ".join(run.name for run in seed_runs), "",
                  "| measure | value |", "|---|---|",
                  "| VRMSE of one sample, per seed | " + ", ".join(f"{v:.4f}" for v in per_seed) + " |",
                  f"| VRMSE of the mean over seeds | {mean([r['vrmse_seed_mean'] for r in spread_rows]):.4f} |",
                  f"| spread between seeds (std / std of the truth) | {mean([r['spread'] for r in spread_rows]):.4f} |",
                  f"| share of the squared error that is seed variance | {mean([r['variance_share'] for r in spread_rows]):.3f} |",
                  "| the same per field (density, momentum x, y, pressure) | "
                  + ", ".join(f"{mean([r['variance_share_per_field'][c] for r in spread_rows]):.3f}" for c in range(4)) + " |",
                  "| the same per quarter of the frames | "
                  + ", ".join(f"{mean([r['variance_share_by_quarter'][q] for r in spread_rows]):.3f}" for q in range(4)) + " |",
                  ""]
    if entropy_rows:
        lines += ["## Entropy rule: p / density^gamma >= its frame-0 minimum", "",
                  "| run | sims | pixels > 1% below | > 5% below | > 20% below | truth > 1% below | VRMSE | VRMSE, pressure raised to the floor | pressure VRMSE | after |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for name, rows in entropy_rows.items():
            m = lambda key: mean([row[key] for row in rows])
            lines.append(f"| {name} | {len(rows)} | {100 * m('below_1pct'):.3f}% | {100 * m('below_5pct'):.3f}% | "
                         f"{100 * m('below_20pct'):.3f}% | {100 * m('truth_below_1pct'):.4f}% | {m('vrmse_before'):.4f} | "
                         f"{m('vrmse_after'):.4f} | {m('pressure_before'):.4f} | {m('pressure_after'):.4f} |")
    return lines


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed-runs", nargs="*", default=[])
    p.add_argument("--entropy-runs", nargs="*", default=[])
    p.add_argument("--out", required=True)
    args = p.parse_args()
    seed_runs = [Path(r) for r in args.seed_runs if (Path(r) / "rollouts").is_dir()]
    entropy_runs = [Path(r) for r in args.entropy_runs if (Path(r) / "rollouts").is_dir()]
    print("seed runs:", [r.name for r in seed_runs], "| entropy runs:", [r.name for r in entropy_runs], flush=True)
    settings = yaml.safe_load(((seed_runs + entropy_runs)[0] / "eval_args.yaml").read_text())
    torch.cuda.set_device(0)
    normalization = Normalization.of_vae_checkpoint(settings["vae_ckpt"])
    test_set = ShockWaveDataset(settings["test_data"], num_sim_frames=1 + settings["frames"])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    spread_rows, entropy_rows = [], {}
    try:
        if len(seed_runs) >= 2:
            spread_rows = seed_spread(seed_runs, normalization, test_set)
    except Exception:
        traceback.print_exc()
    try:
        if entropy_runs:
            entropy_rows = entropy_rule(entropy_runs, normalization, test_set)
    except Exception:
        traceback.print_exc()
    (out / "spread_and_entropy.json").write_text(json.dumps({"seed_runs": [str(r) for r in seed_runs],
                                                             "spread": spread_rows, "entropy": entropy_rows}))
    text = "\n".join(report(spread_rows, seed_runs, entropy_rows))
    (out / "spread_and_entropy.md").write_text(text + "\n")
    print("\n" + text + f"\n\nwritten to {out}", flush=True)


if __name__ == "__main__":
    main()
