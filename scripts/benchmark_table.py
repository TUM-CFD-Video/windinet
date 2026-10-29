"""Markdown table of the benchmark block of one or more eval_dit_vrmse vrmse_summary.json files:
one row for VAE+DiT and one for its VAE round trip, mean +- std over sims, the generation time per
trajectory, then the per-frame metrics.

    python scripts/benchmark_table.py baseline=logs/jupiter/eval_dit_vrmse/<job>/vrmse_summary.json [name=path ...]
"""
import json
import sys

FMT = {"rmse": "{:.3f} ± {:.3f}", "vrmse": "{:.3f} ± {:.3f}", "ssim": "{:.3f} ± {:.3f}", "mlw": "{:.1f} ± {:.1f}"}

runs = [arg.split("=", 1) for arg in sys.argv[1:]]
full = {name: json.load(open(path)) for name, path in runs}
summaries = {name: s["benchmark"] for name, s in full.items()}
gen_time = {name: s.get("generation_time") for name, s in full.items()}

print("| Model | RMSE | VRMSE | 1 – SSIM | MLW |")
print("|---|---|---|---|---|")
for name, b in summaries.items():
    for label, row in (("vae_dit", name), ("vae_only", f"{name}: its VAE round trip")):
        cells = [FMT[m].format(b[label][m]["mean"], b[label][m]["std_over_sims"]) for m in FMT]
        print(f"| {row} | " + " | ".join(cells) + " |")
    print(f"\n{name}: frames {b['scored_frames']}, clamp={b['clamp_prediction']}")
    gt = gen_time.get(name)
    if gt:
        print(f"{name}: generation {gt['seconds_per_trajectory']:.2f} s/trajectory ({gt['total_seconds']:.0f} s total / "
              f"{full[name]['n_samples']} sims, {gt['device']}, {gt['num_inference_steps']} steps, batch 1)")

for name, b in summaries.items():
    if "per_frame" not in b:
        continue
    print(f"\n{name} per frame (VAE+DiT, mean ± std over sims)")
    print("| Frame | RMSE | VRMSE | 1 – SSIM |")
    print("|---|---|---|---|")
    pf = b["per_frame"]["vae_dit"]
    for i, f in enumerate(b["scored_frames"]):
        cells = [FMT[m].format(pf[m]["mean"][i], pf[m]["std_over_sims"][i]) for m in ("rmse", "vrmse", "ssim")]
        print(f"| {f} | " + " | ".join(cells) + " |")
