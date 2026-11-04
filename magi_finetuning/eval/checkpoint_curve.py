"""Learning curve over the saved DiT checkpoints: train vs test, one-step velocity loss and rollout.

Per checkpoint (one job each), on 30 test sims (those of the latent_sensitivity.py run) and 30 train sims (the
train split, spread over gamma the same way), with the F3 VAE:

1. One-step velocity loss, no rollout: every chunk noised to the same level (image latent clean, as in training),
   for noise fractions 0.1, 0.3, 0.5, 0.7, 0.9 (= 1 - MAGI's t). The noise of (sim, level) comes from its own
   torch.Generator, seeded by (split, sim index, level index), so every checkpoint sees exactly the same noisy
   inputs. Saved per level: dit.velocity_loss (raw MSE on the video latents, as in the training log) and that /
   mean(target^2); each also for chunk 1 (latents 1..5, frames 1..20) and chunks 2..5 (latents 6..25).
2. Rollout (64 Euler steps, shift 5, window 4 by default), initial noise seeded per sim as evaluate.py (the same
   for every checkpoint): benchmark metrics on frames 10, 20, ..., 90 and the relative latent error
   RMS((z_DiT - z) / std(z)) in total and per chunk.

Per-sim results go to <out>/scores/<split>_<sim>.json as they finish (a resubmitted job skips them).

    sbatch --time=01:30:00 jobs/jupiter/magi1/quick_test.sbatch -m magi_finetuning.eval.checkpoint_curve \
        --dit-ckpt <ckpt> --out <folder>/<step>
    sbatch --dependency=afterany:<jobs> jobs/jupiter/magi1/quick_test.sbatch -m magi_finetuning.eval.checkpoint_curve \
        --collect <folder>           (table + plots over all checkpoint subfolders: curve.md, curves.png, by_gamma.png)
"""

import argparse
import json
import re
import statistics
from pathlib import Path

import torch

from magi_finetuning.data import Normalization, build_latent_sequence, load_split, stratified_subset
from magi_finetuning.eval.dit_latent_error import chunk_slices, relative_error
from magi_finetuning.eval.evaluate import SEED, decode_body, load_physical_fields
from magi_finetuning.eval.latent_sensitivity import INK, MUTED, SURFACE, VAES, scored, style
from magi_finetuning.settings import CHUNK_WIDTH, OUTPUTS, SAMPLER_WINDOW, TEST_DATA, TRAIN_DATA
from windinet.losses import SSIMLoss
from windinet.training.shockwave_data import CHANNEL_NAMES, ShockWaveDataset, parse_gamma

import matplotlib.pyplot as plt  # noqa: E402  (after report: it sets the backend and MPLCONFIGDIR)

NOISE_FRACTIONS = (0.1, 0.3, 0.5, 0.7, 0.9)
PARTS = {"all": (1, 26), "chunk 1": (1, CHUNK_WIDTH), "chunks 2..5": (CHUNK_WIDTH, 26)}  # video latents [a, b)
SPLIT_OFFSET = {"test": 0, "train": 10_000_000}  # keeps the noise seeds of the two splits apart
SPLIT_COLORS = {"train": "#2a78d6", "test": "#eb6834"}  # fixed per split
GAMMA_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
              "#1c5cab", "#184f95", "#104281", "#0d366b"]  # one-hue sequential ramp: gamma is a magnitude


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dit-ckpt")
    p.add_argument("--vae-ckpt", default=str(VAES["F3"]))
    p.add_argument("--noise-dir", default=str(OUTPUTS / "diagnostics" / "latent_sensitivity"),
                   help="latent_sensitivity.py output: its sims are the test sims")
    p.add_argument("--sims", type=int, default=30, help="train sims, spread over gamma")
    p.add_argument("--num-steps", type=int, default=64)
    p.add_argument("--shift", type=float, default=5.0)
    p.add_argument("--window", type=int, default=SAMPLER_WINDOW)
    p.add_argument("--out")
    p.add_argument("--collect", help="instead: table and plots over every checkpoint subfolder of this folder")
    return p.parse_args()


def velocity_losses(model, latents, gamma, num_chunks, tokens_per_chunk, seed, use_autocast) -> dict:
    """{noise fraction: {part: {"raw", "normalized"}}} with the noise of level i from Generator(seed + i)."""
    from magi_finetuning.dit import add_noise, keep_image_clean

    result = {}
    for level, fraction in enumerate(NOISE_FRACTIONS):
        noise = torch.randn(latents.shape, device="cuda", generator=torch.Generator("cuda").manual_seed(seed + level))
        chunk_times = torch.full((1, num_chunks), 1 - fraction, device="cuda")
        noisy = keep_image_clean(add_noise(latents, chunk_times, noise), latents)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_autocast):
            velocity = model(noisy, chunk_times, gamma, num_chunks, tokens_per_chunk)
        target = latents - noise
        squared = (velocity.float() - target).pow(2)
        result[str(fraction)] = {part: {"raw": squared[:, :, a:b].mean().item(),
                                        "normalized": (squared[:, :, a:b].mean() / target[:, :, a:b].pow(2).mean()).item()}
                                 for part, (a, b) in PARTS.items()}
    return result


@torch.no_grad()
def run(args):
    from magi_finetuning.dit import load_trained_dit, rollout  # GPU imports only where needed
    from magi_finetuning.utils.runtime import init_magi_parallel_state
    from magi_finetuning.vae import load_vae

    out = Path(args.out)
    (out / "scores").mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args)))
    torch.cuda.set_device(0)
    init_magi_parallel_state()
    vae = load_vae(args.vae_ckpt)
    normalization = Normalization.of_vae_checkpoint(args.vae_ckpt)
    model, magi_config, is_full_finetune = load_trained_dit(args.dit_ckpt)
    scale = magi_config.runtime_config.scale_factor
    ssim_fn = SSIMLoss(channels=len(CHANNEL_NAMES)).cuda()
    datasets = {"test": ShockWaveDataset(TEST_DATA, num_sim_frames=101),
                "train": ShockWaveDataset(TRAIN_DATA, num_sim_frames=101)}
    sims = {"test": [r["sim_id"] for r in json.loads((Path(args.noise_dir) / "scores.json").read_text())],
            "train": stratified_subset(load_split()["train"], args.sims)}

    for split in ("test", "train"):
        dataset = datasets[split]
        for count, sim_id in enumerate(sims[split], 1):
            score_file = out / "scores" / f"{split}_{sim_id}.json"
            if score_file.is_file():
                continue
            index = dataset.ids.index(sim_id)
            truth = normalization.apply(load_physical_fields(dataset, sim_id))
            latents, num_body = build_latent_sequence(vae.encoder, truth, scale)
            num_chunks = latents.shape[2] // CHUNK_WIDTH
            tokens_per_chunk = CHUNK_WIDTH * (latents.shape[3] // 2) * (latents.shape[4] // 2)
            gamma = torch.tensor([[parse_gamma(sim_id)]], device="cuda")
            velocity = velocity_losses(model, latents, gamma, num_chunks, tokens_per_chunk,
                                       SEED + SPLIT_OFFSET[split] + 100 * index, is_full_finetune)

            torch.manual_seed(SEED + index)  # the rollout noise of evaluate.py: the same for every checkpoint
            generated = rollout(model, latents[:, :, :1], gamma, num_chunks, tokens_per_chunk, num_steps=args.num_steps,
                                shift=args.shift, clean_t=magi_config.runtime_config.clean_t, window=args.window,
                                use_autocast=is_full_finetune)
            fields = decode_body(vae, generated / scale, num_body, truth.shape[2] - 1)
            body = slice(1, 1 + num_body)
            std = latents[:, :, body].std(dim=(0, 2, 3, 4), keepdim=True)
            error = generated - latents
            scores = {"split": split, "sim_id": sim_id, "gamma": parse_gamma(sim_id), "velocity": velocity,
                      "rollout": scored(fields, truth, ssim_fn),
                      "latent_error": relative_error(error[:, :, body], std),
                      "latent_error_per_chunk": [relative_error(error[:, :, c], std) for c in chunk_slices(num_body)]}
            score_file.write_text(json.dumps(scores))
            print(f"[{split} {count}/{len(sims[split])}] {sim_id}: velocity loss (raw, mean over levels) "
                  f"{statistics.fmean(v['all']['raw'] for v in velocity.values()):.4f}, rollout vrmse "
                  f"{scores['rollout']['vrmse']:.3f}, latent error {scores['latent_error']:.3f}", flush=True)


# ---------------------------------------------------------------- collecting the checkpoints


def load_curve(folder: Path) -> dict:
    """{step: [per-sim scores]} for every subfolder whose args.json names a full_stepNNNNNN checkpoint."""
    curve = {}
    for args_file in folder.glob("*/args.json"):
        step = int(re.search(r"step0*(\d+)", json.loads(args_file.read_text())["dit_ckpt"]).group(1))
        curve[step] = [json.loads(f.read_text()) for f in sorted((args_file.parent / "scores").glob("*.json"))]
    return dict(sorted(curve.items()))


def mean_of(rows, get, **match) -> float:
    picked = [get(r) for r in rows if all(r[k] == v for k, v in match.items())]
    return statistics.fmean(picked) if picked else float("nan")


def level_mean(r, part="all", kind="raw") -> float:
    return statistics.fmean(r["velocity"][str(f)][part][kind] for f in NOISE_FRACTIONS)


def curve_lines(curve) -> list[str]:
    steps = list(curve)
    sims = {s: len([r for r in curve[steps[-1]] if r["split"] == s]) for s in ("train", "test")}
    columns = [("velocity loss, raw (mean over the 5 levels)", lambda r: level_mean(r)),
               ("velocity loss, chunk 1", lambda r: level_mean(r, "chunk 1")),
               ("velocity loss, chunks 2..5", lambda r: level_mean(r, "chunks 2..5")),
               ("rollout VRMSE", lambda r: r["rollout"]["vrmse"]),
               ("rollout 1 - SSIM", lambda r: r["rollout"]["ssim"]),
               ("rollout MLW", lambda r: r["rollout"]["mlw"]),
               ("latent error", lambda r: r["latent_error"])]
    lines = [f"# DiT learning curve: train ({sims['train']} sims) vs test ({sims['test']} sims)", "",
             "Each cell: train / test, mean over sims. Velocity loss: one step, every chunk at the same noise level, "
             "fixed noise per (sim, level), raw MSE on the video latents (as in the training log), mean over the "
             "noise fractions 0.1 .. 0.9. Rollout: benchmark frames 10, 20, ..., 90. Plots: curves.png, by_gamma.png.",
             "", "| step | " + " | ".join(name for name, _ in columns) + " |", "|---|" + "---|" * len(columns)]
    for step, rows in curve.items():
        lines.append(f"| {step} | " + " | ".join(
            f"{mean_of(rows, get, split='train'):.4g} / {mean_of(rows, get, split='test'):.4g}" for _, get in columns) + " |")
    lines += ["", "Velocity loss / mean(target^2) per noise fraction (train / test):", "",
              "| step | " + " | ".join(f"noise {f}" for f in NOISE_FRACTIONS) + " |", "|---|" + "---|" * len(NOISE_FRACTIONS)]
    for step, rows in curve.items():
        get = lambda f: (lambda r: r["velocity"][str(f)]["all"]["normalized"])
        lines.append(f"| {step} | " + " | ".join(
            f"{mean_of(rows, get(f), split='train'):.4f} / {mean_of(rows, get(f), split='test'):.4f}"
            for f in NOISE_FRACTIONS) + " |")
    last = curve[steps[-1]]
    gammas = sorted({r["gamma"] for r in last})
    lines += ["", f"Per gamma at step {steps[-1]} (train / test):", "",
              "| gamma | velocity loss, raw | rollout VRMSE | latent error, chunk 1 |", "|---|---|---|---|"]
    for g in gammas:
        cells = [f"{mean_of(last, get, split='train', gamma=g):.4g} / {mean_of(last, get, split='test', gamma=g):.4g}"
                 for get in (lambda r: level_mean(r), lambda r: r["rollout"]["vrmse"],
                             lambda r: r["latent_error_per_chunk"][0])]
        lines.append(f"| {g:.3f} | " + " | ".join(cells) + " |")
    return lines


def plot_curves(curve, path):
    steps = list(curve)
    panels = [("velocity loss, raw (mean over levels)", lambda r: level_mean(r), "-"),
              ("velocity loss: chunk 1 (solid) vs chunks 2..5 (dashed)", lambda r: level_mean(r, "chunk 1"), "-"),
              ("rollout VRMSE", lambda r: r["rollout"]["vrmse"], "-"),
              ("rollout latent error RMS((z_DiT - z) / std z)", lambda r: r["latent_error"], "-")]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), facecolor=SURFACE)
    flat = list(axes.flat)
    for ax, (title, get, _) in zip(flat[:4], panels):
        style(ax)
        for split, color in SPLIT_COLORS.items():
            ax.plot(steps, [mean_of(curve[s], get, split=split) for s in steps], color=color, lw=2, marker="o",
                    ms=6, label=split)
            if "chunk 1" in title:
                ax.plot(steps, [mean_of(curve[s], lambda r: level_mean(r, "chunks 2..5"), split=split) for s in steps],
                        color=color, lw=2, ls="--", marker="o", ms=6, mfc=SURFACE, label=f"{split}, chunks 2..5")
        ax.set_title(title, color=INK, fontsize=10, loc="left")
    for ax, split in zip(flat[4:], ("train", "test")):  # normalized loss per noise level, darker = noisier
        style(ax)
        for level, fraction in enumerate(NOISE_FRACTIONS):
            ax.plot(steps, [mean_of(curve[s], lambda r: r["velocity"][str(fraction)]["all"]["normalized"], split=split)
                            for s in steps], color=GAMMA_RAMP[3 + 2 * level], lw=2, marker="o", ms=5,
                    label=f"noise {fraction}")
        ax.set_yscale("log")
        ax.set_title(f"{split}: velocity loss / mean(target^2) per noise level", color=INK, fontsize=10, loc="left")
    for ax in flat:
        ax.legend(frameon=False, fontsize=8, labelcolor=INK)
        ax.set_xlabel("training step", color=MUTED, fontsize=9)
    fig.suptitle("DiT learning curve, train vs test (30 sims each)", color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_by_gamma(curve, path):
    steps = list(curve)
    gammas = sorted({r["gamma"] for r in curve[steps[-1]]})
    ramp = [GAMMA_RAMP[round(3 + i * (len(GAMMA_RAMP) - 4) / max(1, len(gammas) - 1))] for i in range(len(gammas))]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), facecolor=SURFACE, sharex=True)
    for row, (title, get) in enumerate((("velocity loss, raw", lambda r: level_mean(r)),
                                        ("rollout VRMSE", lambda r: r["rollout"]["vrmse"]))):
        for col, split in enumerate(("train", "test")):
            ax = axes[row, col]
            style(ax)
            for g, color in zip(gammas, ramp):
                ax.plot(steps, [mean_of(curve[s], get, split=split, gamma=g) for s in steps], color=color, lw=2,
                        label=f"gamma {g:.3f}")
            ax.set_title(f"{split}: {title} per gamma (darker = higher gamma)", color=INK, fontsize=10, loc="left")
    axes[0, 1].legend(frameon=False, fontsize=8, labelcolor=INK, ncol=2)
    for ax in axes[1]:
        ax.set_xlabel("training step", color=MUTED, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def collect(folder: Path):
    curve = load_curve(folder)
    lines = curve_lines(curve)
    (folder / "curve.md").write_text("\n".join(lines) + "\n")
    plot_curves(curve, folder / "curves.png")
    plot_by_gamma(curve, folder / "by_gamma.png")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    arguments = parse_args()
    collect(Path(arguments.collect)) if arguments.collect else run(arguments)
