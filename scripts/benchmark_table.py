"""Markdown table of the benchmark blocks of one or more eval_dit_vrmse vrmse_summary.json files:
for each block (benchmark = frames 40, 50, ...; benchmark_10 = frames 10, 20, ..., from 2026-11-02),
one row for VAE+DiT and one for its VAE round trip, mean +- std over sims, the generation time per
trajectory, then the per-frame metrics.

    python scripts/benchmark_table.py baseline=logs/jupiter/eval_dit_vrmse/<job>/vrmse_summary.json [name=path ...]
"""
import json
import statistics
import sys

FMT = {"rmse": "{:.3f} ± {:.3f}", "vrmse": "{:.3f} ± {:.3f}", "ssim": "{:.3f} ± {:.3f}", "mlw": "{:.1f} ± {:.1f}",
       "h1": "{:.3f} ± {:.3f}", "h2": "{:.3f} ± {:.3f}"}
HEAD = {"rmse": "RMSE", "vrmse": "VRMSE", "ssim": "1 – SSIM", "mlw": "MLW", "h1": "H1 rel", "h2": "H2 rel"}
# Evals from 2026-10-31 to 11-01 hold the squared, unnormalized H1/H2 seminorms of the error instead
OLD_H = {"h1": ("H1 semi", "{:.2e} ± {:.2e}"), "h2": ("H2 semi", "{:.2e} ± {:.2e}")}
BLOCKS = ("benchmark", "benchmark_10")

runs = [arg.split("=", 1) for arg in sys.argv[1:]]
full = {name: json.load(open(path)) for name, path in runs}
gen_time = {name: s.get("generation_time") for name, s in full.items()}
# Frames the DiT generates: all but the given frame 0.
n_generated = {name: s["num_sim_frames"] - 1 for name, s in full.items()}


def head_fmt(m: str, b: dict) -> tuple[str, str]:
    if m in OLD_H and b.get("h_seminorm") != "relative":
        return OLD_H[m]
    return HEAD[m], FMT[m]


for blk in BLOCKS:
    summaries = {name: s[blk] for name, s in full.items() if blk in s}
    if not summaries:
        continue
    first = next(iter(summaries.values()))
    # h1/h2 only in evals from 2026-10-31 on; relative and squared ones not in one column
    metrics = [m for m in FMT if all(m in b["vae_dit"] for b in summaries.values())
               and len({head_fmt(m, b) for b in summaries.values()}) == 1]
    print(f"\n### {blk}\n")
    print("| Model | " + " | ".join(head_fmt(m, first)[0] for m in metrics) + " |")
    print("|---" * (len(metrics) + 1) + "|")
    for name, b in summaries.items():
        for label, row in (("vae_dit", name), ("vae_only", f"{name}: its VAE round trip")):
            cells = [head_fmt(m, b)[1].format(b[label][m]["mean"], b[label][m]["std_over_sims"]) for m in metrics]
            print(f"| {row} | " + " | ".join(cells) + " |")
    print()
    for name, b in summaries.items():
        print(f"{name}: frames {b['scored_frames']}, clamp={b['clamp_prediction']}")

for name in full:
    gt = gen_time.get(name)
    if gt:
        t = sorted(r["generation_seconds"] for r in full[name]["per_sample"])
        median = statistics.median(t)
        print(f"{name}: generation per trajectory median {median:.2f} s (range {t[0]:.2f}-{t[-1]:.2f}), "
              f"mean {statistics.mean(t):.2f} s over {len(t)} sims ({sum(t):.0f} s total); per generated frame "
              f"{median / n_generated[name]:.3f} s (median / {n_generated[name]} frames); {gt['device']}, "
              f"{gt['num_inference_steps']} steps, batch 1")

# Per frame: the block with the most scored frames (benchmark_10 covers benchmark's frames too)
for name, s in full.items():
    b = max((s[blk] for blk in BLOCKS if blk in s), key=lambda b: len(b["scored_frames"]))
    if "per_frame" not in b:
        continue
    print(f"\n{name} per frame (VAE+DiT, mean ± std over sims)")
    pf = b["per_frame"]["vae_dit"]
    frame_metrics = [m for m in FMT if m in pf]
    print("| Frame | " + " | ".join(head_fmt(m, b)[0] for m in frame_metrics) + " |")
    print("|---" * (len(frame_metrics) + 1) + "|")
    for i, f in enumerate(b["scored_frames"]):
        cells = [head_fmt(m, b)[1].format(pf[m]["mean"][i], pf[m]["std_over_sims"][i]) for m in frame_metrics]
        print(f"| {f} | " + " | ".join(cells) + " |")
