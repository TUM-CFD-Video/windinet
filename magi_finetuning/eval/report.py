"""Figures, videos and tables for evaluate.py.

Figures and videos show PHYSICAL (un-normalized) fields. The VRMSE helpers work on whatever they are
given; evaluate.py calls them on normalized fields (Weihao's metric) and on physical ones.
"""

import csv
import os
from pathlib import Path

from magi_finetuning.compat import IG_WORK

os.environ.setdefault("MPLCONFIGDIR", str(IG_WORK / ".cache" / "matplotlib"))  # before matplotlib loads
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from windinet.losses import (h1_seminorm_per_channel, h2_seminorm_per_channel, mlw_loss, rmse_loss,  # noqa: E402
                             vrms_per_channel)
from windinet.training.shockwave_data import CHANNEL_NAMES  # noqa: E402
from windinet.training.vae_visualization import save_reconstruction_panels  # noqa: E402


def vrmse(prediction, target) -> list[float]:
    """Per-channel VRMSE = sqrt(mean((pred - target)^2) / var(target)), inputs (1, 4, T, H, W)."""
    return vrms_per_channel(prediction.float(), target.float()).flatten().tolist()


def vrmse_per_frame(prediction, target) -> np.ndarray:
    """(T,) error of every frame, averaged over the 4 channels: sqrt(frame MSE / trajectory variance).

    Normalized by each field's variance over the WHOLE trajectory (the same normalization as the
    VRMSE in the tables), not per frame: the initial state is nearly uniform, so a per-frame
    variance is ~0 and would blow the first frames up."""
    diff2 = (prediction.float() - target.float()).square().mean(dim=(3, 4))      # (1, 4, T)
    var = target.float().var(dim=(2, 3, 4), unbiased=False).clamp_min(1e-8)       # (1, 4)
    return (diff2 / var[..., None]).sqrt().mean(dim=1)[0].cpu().numpy()


# ---------------------------------------------------------------- per-case figures

def save_case_panels(prediction, target, case_name: str, frame_numbers, output_dir: Path):
    """4 x 3 panel (GT | prediction | residual, one row per field) at each requested frame."""
    return save_reconstruction_panels(
        prediction=prediction[0], target=target[0], sample_id=case_name, label="panels",
        frame_numbers=list(frame_numbers), channel_names=CHANNEL_NAMES, output_dir=output_dir, dpi=120,
    )


def save_error_over_time(curves: dict, chunk_boundaries: list[int], path: Path,
                         comparison: str = "DiT rollout vs dataset") -> Path:
    """Error per frame (frames 1..F; for the rollout: generated from frame 0 only) for each case; dashed
    lines mark where one generated chunk ends and the next begins."""
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    for label, curve in curves.items():
        ax.plot(np.arange(1, len(curve) + 1), curve, lw=1.5, label=label)
    for boundary in chunk_boundaries:
        ax.axvline(boundary + 0.5, color="0.8", ls="--", lw=0.8)
    ax.set_xlabel("frame (frame 0 given)"); ax.set_ylabel("error / trajectory std (mean over fields)")
    ax.set_title(f"{comparison}, per frame (dashed: chunk boundaries)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", color="0.9")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ---------------------------------------------------------------- tables

# Harish's cfd_bench benchmark (Lars's Wan model is the reference baseline), plus Lars's relative H1/H2 seminorm
# errors (commit 648c03f on dev-lars/bench-seminorm, 2026-10-30).
BENCHMARK_METRICS = ("rmse", "vrmse", "ssim", "mlw", "h1", "h2")


def benchmark_metrics(pred, target, ssim_fn) -> dict:
    """Harish's benchmark, verbatim from scripts/eval_dit_vrmse.py ("same definitions as
    cfd_bench/windinet_benchmark_inference.py"): RMSE (pooled), VRMSE (per field, then mean), 1 - SSIM, MLW for
    one sim, [1, C, T, H, W] restricted to the scored frames; plus h1 / h2 = Lars's relative H1 / H2 seminorm
    errors |D(pred - target)| / |D target| per field, then mean (1 = as bad as a flat prediction). SSIM is main's
    fixed version (fields shifted to [0, 1], valid window; commit dd5302c). The caller clamps prediction and
    target to [-1, 1]."""
    return {
        "rmse": float(rmse_loss(pred, target).item()),
        "vrmse": float(vrms_per_channel(pred, target).mean().item()),
        "ssim": float(ssim_fn(pred, target).item()),
        "mlw": float(mlw_loss(pred, target).item()),
        "h1": float(h1_seminorm_per_channel(pred, target).mean().item()),
        "h2": float(h2_seminorm_per_channel(pred, target).mean().item()),
    }


def benchmark_lines(benchmark_rows: list[dict], score_frames) -> list[str]:
    """Markdown table: our rows (mean ± std over sims). No reference rows: other models' numbers are only
    comparable when scored on the same frames, so they are compared outside this report."""
    if not benchmark_rows:
        return []
    frames = f"{score_frames[0]}..{score_frames[-1]}" if list(score_frames) == list(
        range(score_frames[0], score_frames[-1] + 1)) else ", ".join(map(str, score_frames))
    lines = ["", f"## Benchmark (Harish's cfd_bench metrics, frames {frames})", "",
             "| model | sims | rmse | vrmse | 1-ssim | mlw | h1 | h2 |", "|---|---|---|---|---|---|---|---|"]
    for row in benchmark_rows:
        lines.append(f"| MAGI-1: {row['comparison']} | {row['num_sims']} | "
                     + " | ".join(f"{row[m + '_mean']:.4f} ± {row[m + '_std']:.4f}" for m in BENCHMARK_METRICS) + " |")
    lines += ["", "Normalized space, prediction and target clamped to [-1, 1]; each metric per sim over the scored "
                  "frames (frame k = k steps after the given frame 0), then the mean over sims."]
    return lines


def inference_summary(scores: list[dict], num_frames: int, sampler: str) -> dict | None:
    """Inference time per trajectory (frame 0 -> frames 1..F, evaluate.predict) over every sim that has it:
    mean, median, std, min, max, and the mean per generated frame. None if no sim was timed (VAE-only, or
    scored before 2026-11-01)."""
    timed = [s for s in scores if "inference_seconds" in s]
    if not timed:
        return None
    seconds = np.array([s["inference_seconds"] for s in timed])
    return {"sims": len(timed), "mean": seconds.mean(), "median": float(np.median(seconds)), "std": seconds.std(),
            "min": seconds.min(), "max": seconds.max(), "per_frame": seconds.mean() / num_frames,
            "num_frames": num_frames, "resolution": "x".join(map(str, timed[0]["resolution"])),
            "gpus": ", ".join(sorted({s["gpu"] for s in timed})), "sampler": sampler}


def inference_lines(inference: dict | None) -> list[str]:
    """Markdown table of inference_summary."""
    if not inference:
        return []
    return ["", "## Inference time per trajectory", "",
            f"Frame 0 -> frames 1..{inference['num_frames']} at {inference['resolution']} on one "
            f"{inference['gpus']}: encode frame 0, DiT rollout ({inference['sampler']}), decode. Model loading, "
            "the VAE round trip and scoring are not included.", "",
            "| sims | mean (s) | median (s) | std (s) | min (s) | max (s) | per generated frame, mean (s) |",
            "|---|---|---|---|---|---|---|",
            f"| {inference['sims']} | {inference['mean']:.2f} | {inference['median']:.2f} | {inference['std']:.2f} | "
            f"{inference['min']:.2f} | {inference['max']:.2f} | {inference['per_frame']:.3f} |"]


def runtime_lines(runtime: list[dict]) -> list[str]:
    """Markdown runtime table: one row per stage; '-' where a stage has no timing record."""
    if not runtime:
        return []
    fmt = lambda row, key, spec: format(row[key], spec) if row.get(key) is not None else "-"
    lines = ["", "## Runtime", "",
             "| stage | jobs | steps / sims | sec per step / sim | wall time (h) | GPU-hours | GPUs | setup per job (min) | peak GPU mem (GB) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for row in runtime:
        count = row.get("steps", row.get("sims"))
        per = row.get("sec_per_step", row.get("sec_per_sim"))
        lines.append(f"| {row['stage']} | {fmt(row, 'jobs', 'd')} | {count if count is not None else '-'} | "
                     f"{format(per, '.1f') if per is not None else '-'} | {fmt(row, 'wall_hours', '.2f')} | "
                     f"{fmt(row, 'gpu_hours', '.1f')} | {fmt(row, 'num_gpus', 'd')} | "
                     f"{fmt(row, 'setup_minutes_per_job', '.1f')} | {fmt(row, 'peak_gpu_mem_gb', '.1f')} |")
    lines.append("")
    lines.append("Step = one optimizer step (effective batch 32). GPU-hours count every GPU of every job, "
                 "setup included. Evaluation: time per test simulation on one GPU.")
    return lines


def write_tables(case_rows: list[dict], summary_rows: list[dict], output_dir: Path,
                 runtime: list[dict] | None = None, benchmark_rows=None, score_frames=None, sampler: str = "",
                 inference: dict | None = None) -> Path:
    """results.md (human) + results_cases.csv / results_summary.csv (machine)."""
    fields = CHANNEL_NAMES + ["mean"]
    lines = ["# MAGI-1 evaluation on test.h5"]
    lines += benchmark_lines(benchmark_rows or [], score_frames)
    lines += inference_lines(inference)
    lines += ["", f"## Over {summary_rows[0]['num_sims']} test simulations, all frames 1..F (VRMSE, mean ± std)", "",
             "| comparison | " + " | ".join(fields) + " |", "|---|" + "---|" * len(fields)]
    for row in summary_rows:
        lines.append(f"| {row['comparison']} | "
                     + " | ".join(f"{row[f + '_mean']:.4f} ± {row[f + '_std']:.4f}" for f in fields) + " |")
    lines += ["", "## Selected cases (VRMSE, normalized space)", "",
              "| case | gamma | comparison | " + " | ".join(fields) + " |",
              "|---|---|---|" + "---|" * len(fields)]
    for row in case_rows:
        lines.append(f"| {row['case']} | {row['gamma']:.3f} | {row['comparison']} | "
                     + " | ".join(f"{row[f]:.4f}" for f in fields) + " |")
    lines += runtime_lines(runtime or [])
    lines += ["",
              "- VRMSE = sqrt(mean((pred - truth)^2) / var(truth)) per channel, then the channel mean.",
              "- [normalized]: in the model's input space; [physical]: in dataset units.",
              "- Frame 0 is given; the per-field tables score every frame 1..F, the benchmark only its frames.",
              "- VAE round trip: encode -> decode of the true frames 1..F.",
              f"- DiT rollout: generated from frame 0 only (MAGI-1 image-to-video); sampler: {sampler}."]
    md = output_dir / "results.md"
    md.write_text("\n".join(lines) + "\n")

    for name, rows in (("results_cases.csv", case_rows), ("results_summary.csv", summary_rows)):
        if not rows:
            continue
        with open(output_dir / name, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    return md
