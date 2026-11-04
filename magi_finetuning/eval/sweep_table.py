"""One table of every dit_latent_error.py run under a folder (one subfolder per sampler setting, e.g. a sweep),
sorted by VRMSE: benchmark means over the sims, relative latent error (total and per chunk), inference time.

    python magi_finetuning/eval/sweep_table.py <folder>     (run as a file: standard library only, without the
                                                             package's torch imports; fine on the login node;
                                                             writes <folder>/sweep.md)
"""

import argparse
import json
import statistics
from pathlib import Path

METRICS = {"vrmse": "VRMSE", "rmse": "RMSE", "ssim": "1 - SSIM", "mlw": "MLW", "h1": "H1", "h2": "H2"}  # report.BENCHMARK_METRICS


def summarize(scores_file: Path) -> dict:
    rows = json.loads(scores_file.read_text())
    mean = lambda get: statistics.fmean(get(r) for r in rows)
    return {"name": scores_file.parent.name, "sims": len(rows),
            **{m: mean(lambda r: r["dit"][m]) for m in METRICS},
            "latent_error": mean(lambda r: r["relative_latent_error"]),
            "per_chunk": [mean(lambda r: r["per_chunk"][k]) for k in range(len(rows[0]["per_chunk"]))],
            "seconds": mean(lambda r: r["inference_seconds"])}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("folder")
    folder = Path(p.parse_args().folder)
    runs = sorted((summarize(f) for f in folder.glob("*/scores.json")), key=lambda run: run["vrmse"])
    chunks = len(runs[0]["per_chunk"])
    lines = [f"# Sampler sweep: {folder}", "",
             "DiT rollout, mean over the sims (benchmark frames 10, 20, ..., 90). Latent error = RMS((z_DiT - z) / "
             "std(z)) over the video latents, total and per chunk. Sorted by VRMSE.", "",
             "| setting | sims | " + " | ".join(METRICS.values()) + " | latent error | "
             + " | ".join(f"chunk {k + 1}" for k in range(chunks)) + " | s / trajectory |",
             "|---|---|" + "---|" * (len(METRICS) + 1 + chunks + 1)]
    for run in runs:
        lines.append(f"| {run['name']} | {run['sims']} | " + " | ".join(f"{run[m]:.4g}" for m in METRICS)
                     + f" | {run['latent_error']:.3f} | " + " | ".join(f"{v:.3f}" for v in run["per_chunk"])
                     + f" | {run['seconds']:.1f} |")
    (folder / "sweep.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
