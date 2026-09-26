#!/usr/bin/env python
"""Build the Chapter 6 VAE test-set table from every run's test_eval_256.json.

Reads finetune_vae_outputs/sng_pvc/finetune_vae_ch6_*/test_eval_256.json
(written by scripts/eval_vae_test.py via `ch6_submit.sh vaetest`) and prints
one markdown row per run: pooled VRMSE, chmean, delta chmean vs the baseline,
and the four per-channel VRMSEs.

Each delta also gets a paired standard error, computed from the per-sim
chmean differences against the baseline on the same test sims. A delta
within about 2 SE of zero is not distinguishable from the baseline on this
test set. Seed-to-seed variation is a separate noise source, and this SE
does not cover it.

Usage:
    python scripts/vae_test_table.py [--root finetune_vae_outputs/sng_pvc] [--baseline finetune_vae_ch6_loss_rmse_h1_256res]
"""

import json
import math
from pathlib import Path

import typer

app = typer.Typer(pretty_exceptions_enable=False)

CHANNELS = ["density", "momentum_x", "momentum_y", "pressure"]


def _paired_delta(run: dict, base: dict) -> tuple[float, float] | None:
    """Relative chmean delta vs baseline and its paired SE, over sims both runs share."""
    base_by_id = {row["id"]: row["vrmse_chmean"] for row in base["per_sim"]}
    diffs = [row["vrmse_chmean"] - base_by_id[row["id"]] for row in run["per_sim"] if row["id"] in base_by_id]
    if len(diffs) < 2:
        return None
    n = len(diffs)
    mean = sum(diffs) / n
    sd = math.sqrt(sum((d - mean) ** 2 for d in diffs) / (n - 1))
    ref = base["metrics"]["vrmse_chmean"]
    return mean / ref, sd / math.sqrt(n) / ref


@app.command()
def main(
    root: str = typer.Option("finetune_vae_outputs/sng_pvc", help="Folder holding the finetune_vae_ch6_* run mirrors"),
    baseline: str = typer.Option("finetune_vae_ch6_loss_rmse_h1_256res", help="Run the deltas are relative to"),
) -> None:
    reports = {p.parent.name: json.loads(p.read_text()) for p in sorted(Path(root).glob("finetune_vae_ch6_*/test_eval_256.json"))}
    if not reports:
        raise SystemExit(f"no test_eval_256.json under {root}")
    base = reports.get(baseline)

    num_sims = {r["num_sims"] for r in reports.values()}
    print(f"Test set: {next(iter(reports.values()))['test_h5']}, sims per run: {sorted(num_sims)}\n")
    print("| Run | vrmse | chmean | Δ chmean (± paired SE) | " + " | ".join(CHANNELS) + " |")
    print("|---|---|---|---|" + "---|" * len(CHANNELS))
    for name, rep in sorted(reports.items(), key=lambda kv: kv[1]["metrics"]["vrmse_chmean"]):
        m = rep["metrics"]
        if base is None or name == baseline:
            delta = "—"
        else:
            d = _paired_delta(rep, base)
            delta = "n/a" if d is None else f"{d[0]:+.1%} ± {d[1]:.1%}"
        label = name.removeprefix("finetune_vae_ch6_")
        label = f"**{label}**" if name == baseline else label
        cells = [f"{m['vrmse']:.4f}", f"{m['vrmse_chmean']:.4f}", delta] + [f"{m[f'vrmse_{c}']:.4f}" for c in CHANNELS]
        print(f"| {label} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    app()
