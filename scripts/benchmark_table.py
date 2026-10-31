"""Markdown table of the benchmark block of one or more eval_dit_vrmse vrmse_summary.json files:
one row for VAE+DiT and one for its VAE round trip, mean +- std over sims, the generation time per
trajectory, then the per-frame metrics.

    python scripts/benchmark_table.py baseline=logs/jupiter/eval_dit_vrmse/<job>/vrmse_summary.json [name=path ...]
"""
import json
import statistics
import sys

FMT = {"rmse": "{:.3f} ± {:.3f}", "vrmse": "{:.3f} ± {:.3f}", "ssim": "{:.3f} ± {:.3f}", "mlw": "{:.1f} ± {:.1f}",
       "h1": "{:.2e} ± {:.2e}", "h2": "{:.2e} ± {:.2e}"}
HEAD = {"rmse": "RMSE", "vrmse": "VRMSE", "ssim": "1 – SSIM", "mlw": "MLW", "h1": "H1 semi", "h2": "H2 semi"}

runs = [arg.split("=", 1) for arg in sys.argv[1:]]
full = {name: json.load(open(path)) for name, path in runs}
summaries = {name: s["benchmark"] for name, s in full.items()}
gen_time = {name: s.get("generation_time") for name, s in full.items()}
# Frames the DiT generates: all but the given frame 0.
n_generated = {name: s["num_sim_frames"] - 1 for name, s in full.items()}

# h1/h2 only in evals from 2026-10-31 on
metrics = [m for m in FMT if all(m in b["vae_dit"] for b in summaries.values())]
print("| Model | " + " | ".join(HEAD[m] for m in metrics) + " |")
print("|---" * (len(metrics) + 1) + "|")
for name, b in summaries.items():
    for label, row in (("vae_dit", name), ("vae_only", f"{name}: its VAE round trip")):
        cells = [FMT[m].format(b[label][m]["mean"], b[label][m]["std_over_sims"]) for m in metrics]
        print(f"| {row} | " + " | ".join(cells) + " |")
    print(f"\n{name}: frames {b['scored_frames']}, clamp={b['clamp_prediction']}")
    gt = gen_time.get(name)
    if gt:
        t = sorted(r["generation_seconds"] for r in full[name]["per_sample"])
        median = statistics.median(t)
        print(f"{name}: generation per trajectory median {median:.2f} s (range {t[0]:.2f}-{t[-1]:.2f}), "
              f"mean {statistics.mean(t):.2f} s over {len(t)} sims ({sum(t):.0f} s total); per generated frame "
              f"{median / n_generated[name]:.3f} s (median / {n_generated[name]} frames); {gt['device']}, "
              f"{gt['num_inference_steps']} steps, batch 1")

for name, b in summaries.items():
    if "per_frame" not in b:
        continue
    print(f"\n{name} per frame (VAE+DiT, mean ± std over sims)")
    pf = b["per_frame"]["vae_dit"]
    frame_metrics = [m for m in FMT if m in pf]
    print("| Frame | " + " | ".join(HEAD[m] for m in frame_metrics) + " |")
    print("|---" * (len(frame_metrics) + 1) + "|")
    for i, f in enumerate(b["scored_frames"]):
        cells = [FMT[m].format(pf[m]["mean"][i], pf[m]["std_over_sims"][i]) for m in frame_metrics]
        print(f"| {f} | " + " | ".join(cells) + " |")
