"""Two quick diagnostics, decoder fragility vs latent shift, on ~N test sims (spread over gammas):

1. Decoder noise-sensitivity curve (off-manifold fragility): clean latents z = E(x) of each VAE (official, E, F3),
   perturbed z(sigma) = z + sigma * std(z) * eps with eps ~ N(0, I) (std per latent channel over the sim's video
   latents; one eps per sim, shared by every VAE and sigma), decoded by the same VAE and scored with the benchmark
   metrics on frames 10, 20, ..., 90. Plotted: VRMSE, 1 - SSIM, MLW; the table has all six.
2. Latent distribution and spectrum shift (stock vs official vs F3): per latent channel the mean, std, excess
   kurtosis and the 2D spatial radially averaged power spectrum (RAPS) of the video latents z_1..25 (frames
   1..100); and the frame-0 image latent z_img (frame 0 alone, repeated over the 4-frame temporal patch, as the
   DiT is conditioned) against z_1..25, per channel and per gamma.

Latents are in the DiT's space (x scale_factor); the noise curve and the kurtosis do not depend on the scale.
Each VAE sees the fields normalized the way it was trained.

    sbatch --time=00:40:00 jobs/jupiter/magi1/quick_test.sbatch -m magi_finetuning.eval.latent_sensitivity --out <dir>
"""

import argparse
import json
from pathlib import Path

import torch

from magi_finetuning.data import Normalization, build_latent_sequence, stratified_subset
from magi_finetuning.eval.evaluate import SEED, decode_body, load_physical_fields
from magi_finetuning.eval.report import BENCHMARK_METRICS, benchmark_metrics
from magi_finetuning.settings import SCRATCH, TEST_DATA
from windinet.losses import SSIMLoss
from windinet.training.shockwave_data import CHANNEL_NAMES, ShockWaveDataset, parse_gamma

import matplotlib.pyplot as plt  # noqa: E402  (after report: it sets the backend and MPLCONFIGDIR)

VAES = {  # name -> checkpoint (None: the released weights, pressure filters = mean of RGB; progress.md's stock row)
    "official": SCRATCH / "magi1_official" / "vae" / "vae_best.safetensors",
    "E": SCRATCH / "magi1_vae_E_vrmse_loss_io_lr20_wide16_refine64" / "vae_best.safetensors",
    "F3": SCRATCH / "magi1_vae_F3_vrmse_loss_io_lr20_wide16_refine64_40epochs" / "vae_best.safetensors",
    "stock": None,
}
CURVE_VAES = ("official", "E", "F3")  # the stock decoder has no pressure output
STATS_VAES = ("stock", "official", "F3")
COLORS = {"official": "#2a78d6", "E": "#eb6834", "F3": "#1baf7a", "stock": "#eda100"}  # fixed per VAE
SIGMAS = (0.0, 0.02, 0.05, 0.10, 0.15, 0.20)
SCORE_FRAMES = list(range(10, 91, 10))  # the comparison of record (Harish)
PLOTTED = ("vrmse", "ssim", "mlw")
LABELS = {"rmse": "RMSE", "vrmse": "VRMSE", "ssim": "1 - SSIM", "mlw": "MLW", "h1": "H1", "h2": "H2"}
HIGH_FREQUENCY = 8  # RAPS rings k >= 8 of 16 (the upper half of the band up to Nyquist)
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#1f1f1e", "#8a897f", "#e4e3dd"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--test-data", default=TEST_DATA)
    p.add_argument("--sims", type=int, default=30, help="spread over gammas (3 per gamma)")
    p.add_argument("--out", required=True)
    return p.parse_args()


def scored(prediction, truth, ssim_fn) -> dict:
    """prediction: frames 1..F; truth: frames 0..F (normalized). Benchmark metrics on SCORE_FRAMES."""
    index = torch.tensor(SCORE_FRAMES, device=truth.device)
    return benchmark_metrics(prediction.index_select(2, index - 1).clamp(-1, 1),
                             truth.index_select(2, index).clamp(-1, 1), ssim_fn)


def moments(z) -> dict:
    """Per channel of latents (N, C, T, h, w): mean, std and excess kurtosis over sims, time and space."""
    x = z.transpose(0, 1).flatten(1).double()
    mean = x.mean(dim=1)
    centred = x - mean[:, None]
    var = centred.pow(2).mean(dim=1)
    return {"mean": mean.tolist(), "std": var.sqrt().tolist(), "kurtosis": (centred.pow(4).mean(dim=1) / var ** 2 - 3).tolist()}


def spectrum(z) -> dict:
    """Per channel of latents (N, C, T, h, w): radially averaged power spectrum of the h x w latent frames
    (|FFT2|^2, orthonormal, averaged over sims and frames, then over rings of integer radius k = 0 .. h/2; k = 0 is
    the frame mean), the total power per ring and its number of modes, and the fraction of the non-DC power in
    rings k >= HIGH_FREQUENCY."""
    power = torch.fft.fft2(z.double(), norm="ortho").abs().pow(2).mean(dim=(0, 2))  # (C, h, w)
    h, w = power.shape[-2:]
    radius = torch.sqrt((torch.fft.fftfreq(h) * h)[:, None] ** 2 + (torch.fft.fftfreq(w) * w)[None, :] ** 2).round().long()
    rings = h // 2 + 1
    inside = radius < rings  # the corners beyond Nyquist are left out
    totals = torch.zeros(power.shape[0], rings, dtype=power.dtype).index_add_(1, radius[inside], power[:, inside])
    modes = torch.bincount(radius[inside], minlength=rings)
    return {"raps": (totals / modes).tolist(), "ring_power": totals.tolist(), "ring_modes": modes.tolist(),
            "high_frequency": (totals[:, HIGH_FREQUENCY:].sum(1) / totals[:, 1:].sum(1)).tolist()}


def image_vs_video(image, video) -> dict:
    """Per channel: image-latent mean shift in units of the video latents' std, and the std ratio."""
    i, v = moments(image), moments(video)
    return {"shift": [(a - b) / s for a, b, s in zip(i["mean"], v["mean"], v["std"])],
            "std_ratio": [a / b for a, b in zip(i["std"], v["std"])], "kurtosis_image": i["kurtosis"]}


def mean_over(rows, get) -> float:
    return sum(get(r) for r in rows) / len(rows)


def triple(values, fmt) -> str:
    return " / ".join(format(v, fmt) for v in values)


def curve_lines(rows) -> list[str]:
    lines = ["", "## 1. Decoder noise sensitivity: D(z + sigma std(z) eps), mean over sims", "",
             "| VAE | sigma | " + " | ".join(LABELS[m] for m in BENCHMARK_METRICS) + " |",
             "|---|---|" + "---|" * len(BENCHMARK_METRICS)]
    for name in CURVE_VAES:
        for sigma in SIGMAS:
            lines.append(f"| {name} | {sigma:.2f} | " + " | ".join(
                f"{mean_over(rows, lambda r: r['noise'][name][str(sigma)][m]):.4g}" for m in BENCHMARK_METRICS) + " |")
    lines += ["", "Growth factors (mean metric at the larger sigma / at the smaller):", "",
              "| VAE | " + " | ".join(f"{LABELS[m]} 0.05 -> 0.10" for m in PLOTTED) + " | "
              + " | ".join(f"{LABELS[m]} 0 -> 0.20" for m in PLOTTED) + " |", "|---|" + "---|" * 2 * len(PLOTTED)]
    for name in CURVE_VAES:
        at = lambda sigma, m: mean_over(rows, lambda r: r["noise"][name][str(sigma)][m])
        lines.append(f"| {name} | " + " | ".join(f"{at(0.10, m) / at(0.05, m):.2f}x" for m in PLOTTED) + " | "
                     + " | ".join(f"{at(0.20, m) / at(0.0, m):.2f}x" for m in PLOTTED) + " |")
    return lines


def stats_lines(stats, frame0, by_gamma) -> list[str]:
    names = " / ".join(STATS_VAES)
    lines = ["", "## 2. Latent distribution and spectrum of the video latents z_1..25 (frames 1..100)", "",
             f"Each cell: {names}. Over sims, latent frames and the 32 x 32 positions. High-frequency power: share of "
             f"the non-DC spatial power in RAPS rings k >= {HIGH_FREQUENCY} (of 16). Spectra: latent_spectra.png.", "",
             "| channel | mean | std | excess kurtosis | high-frequency power |", "|---|---|---|---|---|"]
    for c in range(len(stats[STATS_VAES[0]]["moments"]["mean"])):
        cells = [triple([stats[n]["moments"][key][c] for n in STATS_VAES], fmt)
                 for key, fmt in (("mean", "+.3f"), ("std", ".3f"), ("kurtosis", "+.2f"))]
        cells.append(triple([stats[n]["spectrum"]["high_frequency"][c] for n in STATS_VAES], ".3f"))
        lines.append(f"| {c} | " + " | ".join(cells) + " |")
    lines += ["", "## 3. Frame-0 image latent z_img against the video latents z_1..25", "",
              f"Each cell: {names}. Mean shift = (mean z_img - mean z_1..25) / std z_1..25, per channel.", "",
              "| channel | mean shift | std z_img / std z_1..25 | excess kurtosis z_img | excess kurtosis z_1..25 |",
              "|---|---|---|---|---|"]
    for c in range(len(frame0[STATS_VAES[0]]["shift"])):
        lines.append(f"| {c} | " + " | ".join([
            triple([frame0[n]["shift"][c] for n in STATS_VAES], "+.2f"),
            triple([frame0[n]["std_ratio"][c] for n in STATS_VAES], ".2f"),
            triple([frame0[n]["kurtosis_image"][c] for n in STATS_VAES], "+.2f"),
            triple([stats[n]["moments"]["kurtosis"][c] for n in STATS_VAES], "+.2f")]) + " |")
    lines += ["", "Per gamma (mean over the 16 channels; sims of that gamma only):", "",
              f"| gamma | sims | mean abs(mean shift) ({names}) | mean std z_img / std z_1..25 ({names}) |",
              "|---|---|---|---|"]
    for gamma, entry in by_gamma.items():
        lines.append(f"| {gamma:.3f} | {entry['sims']} | "
                     + triple([sum(abs(s) for s in entry[n]["shift"]) / len(entry[n]["shift"]) for n in STATS_VAES], ".2f")
                     + " | " + triple([sum(entry[n]["std_ratio"]) / len(entry[n]["std_ratio"]) for n in STATS_VAES], ".2f")
                     + " |")
    return lines


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, lw=0.8)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)


def plot_curve(rows, path, names=CURVE_VAES, points=()):
    """points: (label, sigma, {metric: value}, marker style) drawn on top of the curves (e.g. the DiT rollout)."""
    fig, axes = plt.subplots(1, len(PLOTTED), figsize=(13, 4.2), facecolor=SURFACE)
    for ax, metric in zip(axes, PLOTTED):
        style(ax)
        for name in names:
            ax.plot(SIGMAS, [mean_over(rows, lambda r: r["noise"][name][str(s)][metric]) for s in SIGMAS],
                    color=COLORS[name], lw=2, marker="o", ms=6, label=name)
        for label, sigma, values, marker in points:
            ax.plot([sigma], [values[metric]], ls="none", label=label, **marker)
        ax.set_title(LABELS[metric], color=INK, fontsize=11, loc="left")
        if not points:
            ax.set_xticks(SIGMAS)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK)
    fig.supxlabel("sigma  (z + sigma std(z) eps)", color=INK, fontsize=10)
    fig.suptitle(f"Decoder noise sensitivity, frames 10, 20, ..., 90, mean over {len(rows)} test sims",
                 color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_spectra(stats, path, extra=None):
    """extra: (label, RAPS per channel, color), drawn dashed on top (e.g. the DiT's latent error)."""
    channels = len(stats[STATS_VAES[0]]["spectrum"]["raps"])
    fig, axes = plt.subplots(4, channels // 4, figsize=(13, 11), sharex=True, facecolor=SURFACE)
    for c, ax in enumerate(axes.flat):
        style(ax)
        for name in STATS_VAES:
            raps = stats[name]["spectrum"]["raps"][c]
            ax.plot(range(1, len(raps)), raps[1:], color=COLORS[name], lw=2, label=name)
        if extra:
            label, raps, color = extra
            ax.plot(range(1, len(raps[c])), raps[c][1:], color=color, lw=2, ls="--", label=label)
        ax.set_yscale("log")
        ax.set_title(f"channel {c}", color=INK, fontsize=10, loc="left")
    axes[0, 0].legend(frameon=False, fontsize=9, labelcolor=INK)
    fig.supxlabel("spatial frequency k (ring radius in the 32 x 32 latent frame; 16 = Nyquist)", color=INK, fontsize=10)
    fig.supylabel("power (radial average)", color=INK, fontsize=10)
    fig.suptitle("Radially averaged power spectrum of the video latents z_1..25 per channel"
                 + (f", with the {extra[0]}" if extra else ""), color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


@torch.no_grad()
def main():
    args = parse_args()
    from magi_finetuning.dit import load_magi_config  # GPU imports only where needed
    from magi_finetuning.utils.runtime import init_magi_parallel_state
    from magi_finetuning.vae import load_vae, load_vae_encoder

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(0)
    init_magi_parallel_state()
    scale = load_magi_config().runtime_config.scale_factor
    vaes = {name: load_vae(str(VAES[name])) for name in CURVE_VAES}
    encoders = {name: vaes[name].encoder if name in vaes else load_vae_encoder(VAES[name]) for name in VAES}
    normalizations = {name: Normalization.of_vae_checkpoint(ckpt and str(ckpt)) for name, ckpt in VAES.items()}
    ssim_fn = SSIMLoss(channels=len(CHANNEL_NAMES)).cuda()
    test_set = ShockWaveDataset(args.test_data, num_sim_frames=101)
    sim_ids = stratified_subset(test_set.ids, args.sims)
    image_latents = {name: [] for name in STATS_VAES}
    video_latents = {name: [] for name in STATS_VAES}

    rows = []
    for count, sim_id in enumerate(sim_ids, 1):
        index = test_set.ids.index(sim_id)
        physical = load_physical_fields(test_set, sim_id)
        eps = None
        noise = {}
        for name in VAES:
            truth = normalizations[name].apply(physical)
            z, num_body = build_latent_sequence(encoders[name], truth, scale)  # [image | body 1..25 | padding]
            body = slice(1, 1 + num_body)
            if name in STATS_VAES:
                image_latents[name].append(z[:, :, :1].cpu())
                video_latents[name].append(z[:, :, body].cpu())
            if name not in CURVE_VAES:
                continue
            if eps is None:  # one draw per sim, shared by every VAE and sigma; the image latent is not decoded
                eps = torch.randn(z.shape, device="cuda", generator=torch.Generator("cuda").manual_seed(SEED + index))
            std = z[:, :, body].std(dim=(0, 2, 3, 4), keepdim=True)  # per channel, over this sim's video latents
            noise[name] = {str(sigma): scored(decode_body(vaes[name], (z + sigma * std * eps) / scale, num_body,
                                                          truth.shape[2] - 1), truth, ssim_fn) for sigma in SIGMAS}
        rows.append({"sim_id": sim_id, "gamma": parse_gamma(sim_id), "noise": noise})
        print(f"[{count}/{len(sim_ids)}] {sim_id}: vrmse at sigma 0 / 0.05 / 0.10 / 0.20: " + "; ".join(
            f"{name} " + " / ".join(f"{noise[name][str(s)]['vrmse']:.3f}" for s in (0.0, 0.05, 0.10, 0.20))
            for name in CURVE_VAES), flush=True)

    gammas = torch.tensor([r["gamma"] for r in rows])
    images = {name: torch.cat(image_latents[name]) for name in STATS_VAES}
    videos = {name: torch.cat(video_latents[name]) for name in STATS_VAES}
    stats = {name: {"moments": moments(videos[name]), "spectrum": spectrum(videos[name])} for name in STATS_VAES}
    frame0 = {name: image_vs_video(images[name], videos[name]) for name in STATS_VAES}
    by_gamma = {}
    for gamma in sorted(set(gammas.tolist())):
        same = gammas == gamma
        by_gamma[gamma] = {"sims": int(same.sum()),
                           **{name: image_vs_video(images[name][same], videos[name][same]) for name in STATS_VAES}}

    (out / "scores.json").write_text(json.dumps(rows))
    (out / "latent_stats.json").write_text(json.dumps({"video": stats, "image_vs_video": frame0,
                                                       "image_vs_video_by_gamma": {str(g): e for g, e in by_gamma.items()}}))
    plot_curve(rows, out / "noise_curve.png")
    plot_spectra(stats, out / "latent_spectra.png")
    lines = ["# Decoder noise sensitivity and latent distribution shift", "",
             f"{len(rows)} test sims spread over gamma ({args.test_data}). VAEs: "
             + "; ".join(f"{name} = {ckpt or 'released weights (pressure = mean of RGB filters)'}" for name, ckpt in VAES.items())
             + f". Latents in the DiT's space (x {scale}). Benchmark metrics on frames 10, 20, ..., 90 (normalized, "
             "clamped to [-1, 1], 1 - SSIM fixed, H1 / H2 relative). Curve: noise_curve.png; per sim: scores.json; "
             "all statistics: latent_stats.json."]
    lines += curve_lines(rows)
    lines += stats_lines(stats, frame0, by_gamma)
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
