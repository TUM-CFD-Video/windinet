"""The DiT on the sims of a latent_sensitivity.py run: where in latent space it makes its errors.

On exactly the sims of --noise-dir (a latent_sensitivity.py output), with the F3 VAE the DiT was trained on, in the
DiT's latent space (VAE latents x scale_factor); z = F3's latents of the true frames, z_DiT = the rollout's:

1. DiT rollout (same seeds as evaluate.py; the sampler flags make this a quick benchmark for sampler sweeps):
   benchmark metrics on frames 10, 20, ..., 90 and the inference time per trajectory (as evaluate.py).
2. Placement on the noise curve: the relative latent error RMS((z_DiT - z) / std(z)) over the video latents, std
   per channel over the sim's video latents, i.e. the curve's sigma; and F3's decode of z + sigma_DiT std(z) eps
   with the sim's own sigma_DiT and the curve's eps ("matched noise": white noise of the DiT's size).
3. Relative latent error per MAGI chunk (6 latents; chunk 1 = latents 0..5, latent 0 being the frame-0 image, so
   frames 1..20; then 21..44, 45..68, 69..92, 93..100), and per latent in scores.json.
4. Latent error spectrum: RAPS of z_DiT - z per channel, overlaid on latent_sensitivity's spectra, and the share of
   the error power at k = 0 (frame mean), 1..3, 4..7, 8..16.
5. One-step velocity error, no rollout: every chunk noised to the same level (the image latent clean, as in
   training), MSE of the predicted velocity against z - noise on the video latents, for noise fractions 0.1, 0.3,
   0.5, 0.7, 0.9 (= 1 - MAGI's t, where t = 0 is noise and 1 clean; 0.9 = almost pure noise).

    sbatch --time=00:50:00 jobs/jupiter/magi1/quick_test.sbatch -m magi_finetuning.eval.dit_latent_error \
        --out <dir> [--num-steps 64 --shift 3.0 --window 4]
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from magi_finetuning.data import Normalization, build_latent_sequence, encode_image
from magi_finetuning.eval.evaluate import SEED, decode_body, load_physical_fields
from magi_finetuning.eval.latent_sensitivity import (COLORS, LABELS, SURFACE, VAES, mean_over, plot_curve,
                                                     plot_spectra, scored, spectrum)
from magi_finetuning.eval.report import BENCHMARK_METRICS
from magi_finetuning.settings import CHUNK_WIDTH, OUTPUTS, SAMPLER_WINDOW, SCRATCH, TEST_DATA
from windinet.losses import SSIMLoss
from windinet.training.shockwave_data import CHANNEL_NAMES, ShockWaveDataset, parse_gamma

NOISE_FRACTIONS = (0.1, 0.3, 0.5, 0.7, 0.9)
BANDS = {"k = 0 (frame mean)": (0, 1), "k = 1..3": (1, 4), "k = 4..7": (4, 8), "k = 8..16": (8, 17)}
DIT_COLOR = "#e87ba4"  # the next categorical slot after latent_sensitivity.COLORS
DIT_MARKER = dict(marker="D", ms=9, color=DIT_COLOR, mec=SURFACE, mew=1.5)
MATCHED_MARKER = dict(marker="o", ms=10, mfc=SURFACE, mec=COLORS["F3"], mew=2)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dit-ckpt", default=str(SCRATCH / "magi1_dit_full_8000steps_on_F3" / "full_step008000.safetensors"))
    p.add_argument("--vae-ckpt", default=str(VAES["F3"]))
    p.add_argument("--noise-dir", default=str(OUTPUTS / "diagnostics" / "latent_sensitivity"),
                   help="latent_sensitivity.py output: its sims, noise curve and latent spectra")
    p.add_argument("--test-data", default=TEST_DATA)
    p.add_argument("--num-steps", type=int, default=64)
    p.add_argument("--shift", type=float, default=3.0)
    p.add_argument("--window", type=int, default=SAMPLER_WINDOW)
    p.add_argument("--out", required=True)
    return p.parse_args()


def relative_error(error, std) -> float:
    """RMS of a latent error in units of the per-channel std (the noise curve's sigma)."""
    return (error / std).pow(2).mean().sqrt().item()


def chunk_slices(num_body: int) -> list[slice]:
    """The video latents 1..num_body of each MAGI chunk of CHUNK_WIDTH latents (chunk 1 also holds latent 0)."""
    return [slice(max(1, start), min(num_body + 1, start + CHUNK_WIDTH)) for start in range(0, num_body + 1, CHUNK_WIDTH)]


def frames_of(chunk: slice) -> str:
    return f"{4 * chunk.start - 3}..{4 * (chunk.stop - 1)}"  # latent j covers frames 4j-3..4j


def velocity_errors(model, latents, gamma, num_chunks, tokens_per_chunk, num_body, generator, use_autocast) -> dict:
    """One forward per noise fraction f: every chunk at MAGI's t = 1 - f, the image latent clean (as in training).
    MSE of the predicted velocity against the target z - noise on the video latents, and that / mean(target^2)."""
    from magi_finetuning.dit import add_noise, keep_image_clean

    body = slice(1, 1 + num_body)
    result = {}
    for fraction in NOISE_FRACTIONS:
        chunk_times = torch.full((1, num_chunks), 1 - fraction, device="cuda")
        noise = torch.randn(latents.shape, device="cuda", generator=generator)
        noisy = keep_image_clean(add_noise(latents, chunk_times, noise), latents)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_autocast):
            velocity = model(noisy, chunk_times, gamma, num_chunks, tokens_per_chunk)
        target = (latents - noise)[:, :, body]
        mse = (velocity.float()[:, :, body] - target).pow(2).mean().item()
        result[str(fraction)] = {"mse": mse, "relative": mse / target.pow(2).mean().item()}
    return result


def band_shares(spec) -> dict:
    """Share of the spatial power (DC included) in each of BANDS: per channel, over all channels, and for white
    noise (power proportional to the number of modes)."""
    power, modes = torch.tensor(spec["ring_power"]), torch.tensor(spec["ring_modes"], dtype=torch.float64)
    return {"per_channel": {band: (power[:, a:b].sum(1) / power.sum(1)).tolist() for band, (a, b) in BANDS.items()},
            "all": {band: (power[:, a:b].sum() / power.sum()).item() for band, (a, b) in BANDS.items()},
            "white_noise": {band: (modes[a:b].sum() / modes.sum()).item() for band, (a, b) in BANDS.items()}}


def benchmark_lines(rows, noise_rows, sampler) -> list[str]:
    seconds = [r["inference_seconds"] for r in rows]
    stat = lambda values: f"{statistics.fmean(values):.4g} ± {statistics.pstdev(values):.4g}"
    lines = ["", f"## 1. Benchmark on these {len(rows)} sims (frames 10, 20, ..., 90; mean ± std over sims)", "",
             "| | " + " | ".join(LABELS[m] for m in BENCHMARK_METRICS) + " |", "|---|" + "---|" * len(BENCHMARK_METRICS),
             "| DiT rollout | " + " | ".join(stat([r["dit"][m] for r in rows]) for m in BENCHMARK_METRICS) + " |",
             "| F3 VAE round trip | " + " | ".join(stat([r["noise"]["F3"]["0.0"][m] for r in noise_rows])
                                                for m in BENCHMARK_METRICS) + " |",
             "", f"Sampler: {sampler}. Inference time per trajectory (encode frame 0, rollout, decode; one GPU): mean "
                 f"{statistics.fmean(seconds):.2f} s, median {statistics.median(seconds):.2f} s, std "
                 f"{statistics.pstdev(seconds):.2f} s."]
    return lines


def placement_lines(rows, noise_rows) -> list[str]:
    sigmas = [r["relative_latent_error"] for r in rows]
    lines = ["", "## 2. The DiT on the noise curve", "",
             f"Relative latent error RMS((z_DiT - z) / std(z)) over the video latents: mean {statistics.fmean(sigmas):.3f}, "
             f"median {statistics.median(sigmas):.3f}, min {min(sigmas):.3f}, max {max(sigmas):.3f} "
             "(the curve's sigma; the curve stops at 0.20). Plot: dit_on_noise_curve.png.", "",
             "| F3 decode of | sigma | " + " | ".join(LABELS[m] for m in BENCHMARK_METRICS) + " |",
             "|---|---|" + "---|" * len(BENCHMARK_METRICS)]
    for name, sigma, get in (
            ("z + 0.20 std(z) eps (curve end)", "0.20", lambda r, m: r["noise"]["F3"]["0.2"][m]),
            ("z + sigma_DiT std(z) eps (matched noise)", f"{statistics.fmean(sigmas):.3f}", lambda r, m: r["matched_noise"][m]),
            ("**z_DiT (the rollout)**", f"{statistics.fmean(sigmas):.3f}", lambda r, m: r["dit"][m])):
        source = noise_rows if "curve end" in name else rows
        lines.append(f"| {name} | {sigma} | " + " | ".join(f"{mean_over(source, lambda r: get(r, m)):.4g}"
                                                          for m in BENCHMARK_METRICS) + " |")
    return lines


def chunk_lines(rows, chunks) -> list[str]:
    lines = ["", "## 3. Relative latent error per chunk (mean over sims)", "",
             "| chunk | latents | frames | RMS((z_DiT - z) / std(z)) |", "|---|---|---|---|"]
    for k, chunk in enumerate(chunks):
        lines.append(f"| {k + 1} | {chunk.start}..{chunk.stop - 1} | {frames_of(chunk)} | "
                     f"{mean_over(rows, lambda r: r['per_chunk'][k]):.3f} |")
    per_latent = [mean_over(rows, lambda r: r["per_latent"][j]) for j in range(len(rows[0]["per_latent"]))]
    lines += ["", "Per latent 1..25: " + ", ".join(f"{v:.3f}" for v in per_latent) + "."]
    return lines


def spectrum_lines(error_shares, latent_shares) -> list[str]:
    bands = list(BANDS)
    lines = ["", "## 4. Where the latent error sits in spatial frequency (share of the power, DC included)", "",
             "RAPS rings of the 32 x 32 latent frames, k = 16 is Nyquist. Spectra per channel: "
             "latent_spectra_with_dit_error.png.", "",
             "| | " + " | ".join(bands) + " |", "|---|" + "---|" * len(bands)]
    for name, values in (("DiT error z_DiT - z, all channels", error_shares["all"]),
                         ("F3 latents z, all channels", latent_shares["all"]),
                         ("white noise (share of the modes)", error_shares["white_noise"])):
        lines.append(f"| {name} | " + " | ".join(f"{values[b]:.3f}" for b in bands) + " |")
    for c in range(len(error_shares["per_channel"][bands[0]])):
        lines.append(f"| DiT error, channel {c} | " + " | ".join(f"{error_shares['per_channel'][b][c]:.3f}"
                                                                for b in bands) + " |")
    return lines


def velocity_lines(rows) -> list[str]:
    lines = ["", "## 5. One-step velocity error (no rollout; every chunk at the same noise level), mean over sims", "",
             "| noise fraction (1 - t) | MAGI t | MSE vs z - noise | MSE / mean(target^2) |", "|---|---|---|---|"]
    for fraction in NOISE_FRACTIONS:
        key = str(fraction)
        lines.append(f"| {fraction} | {1 - fraction:.1f} | {mean_over(rows, lambda r: r['velocity'][key]['mse']):.4f} | "
                     f"{mean_over(rows, lambda r: r['velocity'][key]['relative']):.4f} |")
    return lines


@torch.no_grad()
def main():
    args = parse_args()
    from magi_finetuning.dit import load_trained_dit, rollout  # GPU imports only where needed
    from magi_finetuning.utils.runtime import init_magi_parallel_state
    from magi_finetuning.vae import load_vae

    out, noise_dir = Path(args.out), Path(args.noise_dir)
    out.mkdir(parents=True, exist_ok=True)
    noise_rows = json.loads((noise_dir / "scores.json").read_text())
    torch.cuda.set_device(0)
    init_magi_parallel_state()
    vae = load_vae(args.vae_ckpt)
    normalization = Normalization.of_vae_checkpoint(args.vae_ckpt)
    model, magi_config, is_full_finetune = load_trained_dit(args.dit_ckpt)
    scale = magi_config.runtime_config.scale_factor
    ssim_fn = SSIMLoss(channels=len(CHANNEL_NAMES)).cuda()
    test_set = ShockWaveDataset(args.test_data, num_sim_frames=101)
    sampler = f"{args.num_steps} Euler steps per chunk, shift {args.shift}, {args.window} chunk(s) at once"

    rows, latents_seen, errors_seen = [], [], []
    for count, sim_id in enumerate([r["sim_id"] for r in noise_rows], 1):
        index = test_set.ids.index(sim_id)
        torch.manual_seed(SEED + index)  # the same rollout as evaluate.py
        truth = normalization.apply(load_physical_fields(test_set, sim_id))
        latents, num_body = build_latent_sequence(vae.encoder, truth, scale)
        num_frames = truth.shape[2] - 1
        num_chunks = latents.shape[2] // CHUNK_WIDTH
        tokens_per_chunk = CHUNK_WIDTH * (latents.shape[3] // 2) * (latents.shape[4] // 2)
        gamma = torch.tensor([[parse_gamma(sim_id)]], device="cuda")
        torch.cuda.synchronize()
        started = time.perf_counter()  # what evaluate.predict times: frame 0 -> frames 1..F
        image_latent = encode_image(vae.encoder, truth[:, :, :1]) * scale  # == latents[:, :, :1]
        generated = rollout(model, image_latent, gamma, num_chunks, tokens_per_chunk, num_steps=args.num_steps,
                            shift=args.shift, clean_t=magi_config.runtime_config.clean_t, window=args.window,
                            use_autocast=is_full_finetune)
        fields = decode_body(vae, generated / scale, num_body, num_frames)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started

        body = slice(1, 1 + num_body)
        std = latents[:, :, body].std(dim=(0, 2, 3, 4), keepdim=True)  # per channel, as the noise curve
        error = generated - latents
        sigma = relative_error(error[:, :, body], std)
        generator = torch.Generator("cuda").manual_seed(SEED + index)
        eps = torch.randn(latents.shape, device="cuda", generator=generator)  # the noise curve's draw for this sim
        matched = scored(decode_body(vae, (latents + sigma * std * eps) / scale, num_body, num_frames), truth, ssim_fn)
        latents_seen.append(latents[:, :, body].cpu())
        errors_seen.append(error[:, :, body].cpu())
        rows.append({
            "sim_id": sim_id, "gamma": parse_gamma(sim_id), "inference_seconds": round(seconds, 3),
            "dit": scored(fields, truth, ssim_fn), "matched_noise": matched, "relative_latent_error": sigma,
            "per_chunk": [relative_error(error[:, :, chunk], std) for chunk in chunk_slices(num_body)],
            "per_latent": (error[:, :, body] / std).pow(2).mean(dim=(0, 1, 3, 4)).sqrt().tolist(),
            "velocity": velocity_errors(model, latents, gamma, num_chunks, tokens_per_chunk, num_body, generator,
                                        is_full_finetune),
        })
        r = rows[-1]
        print(f"[{count}/{len(noise_rows)}] {sim_id}: relative latent error {sigma:.3f} (per chunk "
              + " / ".join(f"{v:.2f}" for v in r["per_chunk"]) + f"); vrmse DiT {r['dit']['vrmse']:.3f}, matched noise "
              f"{matched['vrmse']:.3f}; {seconds:.1f} s", flush=True)

    error_spectrum = spectrum(torch.cat(errors_seen))
    error_shares, latent_shares = band_shares(error_spectrum), band_shares(spectrum(torch.cat(latents_seen)))
    (out / "scores.json").write_text(json.dumps(rows))
    (out / "error_spectrum.json").write_text(json.dumps({"error": error_spectrum, "error_shares": error_shares,
                                                         "latent_shares": latent_shares}))
    mean_sigma = mean_over(rows, lambda r: r["relative_latent_error"])
    plot_curve(noise_rows, out / "dit_on_noise_curve.png", names=("F3",), points=[
        ("F3 decode of matched noise", mean_sigma, {m: mean_over(rows, lambda r: r["matched_noise"][m]) for m in BENCHMARK_METRICS}, MATCHED_MARKER),
        ("DiT rollout", mean_sigma, {m: mean_over(rows, lambda r: r["dit"][m]) for m in BENCHMARK_METRICS}, DIT_MARKER)])
    plot_spectra(json.loads((noise_dir / "latent_stats.json").read_text())["video"],
                 out / "latent_spectra_with_dit_error.png", extra=("DiT error", error_spectrum["raps"], DIT_COLOR))
    lines = ["# The DiT's latent error on the diagnostic sims", "",
             f"DiT {args.dit_ckpt}; VAE {args.vae_ckpt}; the {len(rows)} sims of {noise_dir} (its noise curve and "
             f"spectra are reused). Latents in the DiT's space (x {scale}); z = F3's latents of the true frames, "
             "z_DiT = the rollout's. Per sim: scores.json; spectra: error_spectrum.json."]
    lines += benchmark_lines(rows, noise_rows, sampler)
    lines += placement_lines(rows, noise_rows)
    lines += chunk_lines(rows, chunk_slices(len(rows[0]["per_latent"])))
    lines += spectrum_lines(error_shares, latent_shares)
    lines += velocity_lines(rows)
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
