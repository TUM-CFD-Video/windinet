"""Sharpness of saved rollouts next to their accuracy, and which latent channels the decoder is sensitive to.

VRMSE is a pointwise squared error: its optimum is the mean over possible futures, which is blurry. The measures
here compare *statistics* of prediction and truth instead of pixels, so a blurred field cannot score well and a
sharp but slightly misplaced feature is not punished:

    power ratio per band     radially averaged power of the prediction / of the truth in a wavenumber band
                             (256 px: k = 1 is the domain, k = 128 Nyquist). 1 = right, < 1 = blurred.
    gradient ratio           mean |grad prediction| / mean |grad truth| (forward differences).
    gradient p99 ratio       the same for the 99th percentile of |grad|: the steepness of the fronts.
    error per band           power of (prediction - truth) / power of the truth per band: where the error sits.
    distance to reference    RMS(prediction - reference rollout) / std(truth): how far two samplers' outputs
                             are from each other, to compare with their distance to the truth (the VRMSE).

Part 1 scores every run folder that evaluate.py --save-rollouts wrote (<run>/rollouts/<sim>.pt: normalized frames
1..F), plus the VAE round trip of the truth as the ceiling a DiT on these latents could reach. Normalized space,
all frames 1..F, per channel, then the mean over channels and sims.

Part 2, decoder sensitivity per latent channel: the same absolute Gaussian noise (--noise, in the DiT's latent
space) is added to one latent channel at a time and decoded. The DiT's absolute latent error is about equal in
every channel (the unweighted velocity MSE makes it so), so the decoded effect of equal noise tells which
channels its pixel error comes from, and whether weighting the loss per channel can help.

    sbatch jobs/jupiter/magi1/quick_test.sbatch -m magi_finetuning.eval.sharpness --runs <run dirs> \
        --reference <the 64-step run> --out <dir>
"""

import argparse
import json
from pathlib import Path

import torch
import yaml

from magi_finetuning.data import Normalization, build_latent_sequence
from magi_finetuning.eval.evaluate import decode_body, load_physical_fields
from magi_finetuning.settings import OUTPUTS
from windinet.training.shockwave_data import ShockWaveDataset

BANDS = ((1, 3), (4, 7), (8, 15), (16, 31), (32, 63), (64, 128))  # wavenumbers at 256 px, both ends included
BAND_NAMES = [f"k {a}..{b}" for a, b in BANDS]
SCALE = 0.18215  # the DiT's latent space = VAE latents x this (MAGI-1 runtime_config.scale_factor)
SEED = 0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True, help="evaluation folders with rollouts/ (evaluate.py --save-rollouts)")
    p.add_argument("--reference", default=None, help="one of --runs: every run's distance to its rollouts")
    p.add_argument("--out", default=str(OUTPUTS / "diagnostics" / "sharpness"))
    p.add_argument("--sensitivity-sims", type=int, default=10, help="sims for the decoder sensitivity (0 = skip)")
    p.add_argument("--noise", type=float, default=0.17,
                   help="absolute noise per latent channel in the DiT's space (its error is 0.15..0.20 in every channel)")
    return p.parse_args()


def band_power(x) -> torch.Tensor:
    """(C, bands): power of the frames of x (1, C, T, H, W) in each wavenumber band, mean over frames.
    |FFT2|^2 (orthonormal) summed over rings of integer radius k, as latent_sensitivity.spectrum, on the GPU."""
    power = torch.fft.fft2(x.double(), norm="ortho").abs().pow(2).mean(dim=(0, 2))  # (C, H, W)
    h, w = power.shape[-2:]
    ky, kx = torch.fft.fftfreq(h, device=x.device) * h, torch.fft.fftfreq(w, device=x.device) * w
    radius = torch.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2).round().long()
    inside = radius <= h // 2  # the corners beyond Nyquist are left out
    rings = torch.zeros(power.shape[0], h // 2 + 1, dtype=power.dtype, device=x.device)
    rings.index_add_(1, radius[inside], power[:, inside])  # ring k at index k
    return torch.stack([rings[:, a:b + 1].sum(dim=1) for a, b in BANDS], dim=1)


def gradient_magnitude(x) -> torch.Tensor:
    """|grad| of every frame by forward differences, on the common (H - 1, W - 1) grid."""
    dy = x[..., 1:, :-1] - x[..., :-1, :-1]
    dx = x[..., :-1, 1:] - x[..., :-1, :-1]
    return torch.sqrt(dx ** 2 + dy ** 2)


def gradient_stats(x) -> tuple[torch.Tensor, torch.Tensor]:
    """Per channel of x (1, C, T, H, W): mean and 99th percentile of |grad| over frames and pixels."""
    g = gradient_magnitude(x)[0].flatten(1)  # (C, T * (H - 1) * (W - 1))
    k = int(0.99 * g.shape[1])
    return g.mean(dim=1), torch.stack([channel.kthvalue(k).values for channel in g])


def relative_rms(a, b, truth) -> float:
    """RMS(a - b) / std(truth) per channel, then the channel mean: the VRMSE when b is the truth."""
    mse = (a - b).pow(2).mean(dim=(0, 2, 3, 4))
    return (mse / truth.var(dim=(0, 2, 3, 4))).sqrt().mean().item()


def describe(prediction, truth, truth_stats) -> dict:
    """The sharpness measures of one prediction (1, C, T, H, W) against the truth."""
    truth_power, truth_grad_mean, truth_grad_p99 = truth_stats
    grad_mean, grad_p99 = gradient_stats(prediction)
    return {
        "vrmse": relative_rms(prediction, truth, truth),
        "power_ratio": (band_power(prediction) / truth_power).mean(dim=0).tolist(),
        "error_per_band": (band_power(prediction - truth) / truth_power).mean(dim=0).tolist(),
        "grad_ratio": (grad_mean / truth_grad_mean).mean().item(),
        "grad_p99_ratio": (grad_p99 / truth_grad_p99).mean().item(),
    }


def mean_of(rows, key):
    values = [row[key] for row in rows]
    if isinstance(values[0], list):
        return [sum(column) / len(column) for column in zip(*values)]
    return sum(values) / len(values)


@torch.no_grad()
def score_runs(runs, reference, vae, normalization, test_set, sim_ids) -> dict:
    """{label: [per-sim measures]} for the VAE round trip and every run."""
    rows = {"VAE round trip": [], **{run.name: [] for run in runs}}
    for count, sim_id in enumerate(sim_ids, 1):
        fields = normalization.apply(load_physical_fields(test_set, sim_id))  # frames 0..F
        truth = fields[:, :, 1:]
        truth_stats = (band_power(truth), *gradient_stats(truth))
        latents, num_body = build_latent_sequence(vae.encoder, fields, 1.0)
        round_trip = decode_body(vae, latents, num_body, truth.shape[2])
        rows["VAE round trip"].append({"sim_id": sim_id, **describe(round_trip, truth, truth_stats)})
        rollouts = {run.name: torch.load(run / "rollouts" / f"{sim_id}.pt").cuda().float() for run in runs}
        for run in runs:
            row = {"sim_id": sim_id, **describe(rollouts[run.name], truth, truth_stats)}
            if reference is not None:
                row["to_reference"] = relative_rms(rollouts[run.name], rollouts[reference.name], truth)
            rows[run.name].append(row)
        print(f"[{count}/{len(sim_ids)}] {sim_id}", flush=True)
    return rows


@torch.no_grad()
def decoder_sensitivity(vae, normalization, test_set, sim_ids, noise: float) -> list[dict]:
    """Per latent channel (and for all 16 at once): decode(z + noise in that channel) against decode(z), as
    RMS / std(truth) over the fields, and the share of that change in k >= 16. Mean over sims. Also the
    channel's std over these sims in the DiT's space."""
    generator = torch.Generator(device="cuda").manual_seed(SEED)
    channels = None
    for sim_id in sim_ids:
        fields = normalization.apply(load_physical_fields(test_set, sim_id))
        truth = fields[:, :, 1:]
        latents, num_body = build_latent_sequence(vae.encoder, fields, 1.0)  # unscaled
        base = decode_body(vae, latents, num_body, truth.shape[2])
        eps = torch.randn(latents.shape, device="cuda", generator=generator) * noise / SCALE  # independent per channel
        eps[:, :, :1] = 0  # the image latent stays as it is
        if channels is None:
            channels = [{"channel": c, "std": [], "effect": [], "fine_share": []} for c in [*range(latents.shape[1]), "all"]]
        for entry in channels:
            noisy = latents.clone()
            if entry["channel"] == "all":
                noisy += eps
            else:
                noisy[:, entry["channel"]] += eps[:, entry["channel"]]
            change = decode_body(vae, noisy, num_body, truth.shape[2]) - base
            power = band_power(change).sum(dim=0)  # (bands,), summed over the 4 fields
            entry["effect"].append((change.pow(2).mean(dim=(0, 2, 3, 4)) / truth.var(dim=(0, 2, 3, 4))).sqrt().mean().item())
            entry["fine_share"].append((power[3:].sum() / power.sum()).item())
            if entry["channel"] != "all":
                entry["std"].append((latents[:, entry["channel"], 1:1 + num_body] * SCALE).std().item())
        print(f"[sensitivity] {sim_id}", flush=True)
    mean = lambda values: sum(values) / len(values) if values else None
    return [{"channel": e["channel"], "std": mean(e["std"]), "effect": mean(e["effect"]), "fine_share": mean(e["fine_share"])}
            for e in channels]


def report_lines(rows, reference, sensitivity, noise: float) -> list[str]:
    lines = ["# Sharpness of the rollouts", "",
             f"{len(next(iter(rows.values())))} test sims, normalized space, all frames 1..F, mean over the 4 fields and the "
             "sims. Ratios are prediction / truth: 1 = as sharp as the truth, < 1 = blurred. Definitions: "
             "magi_finetuning/eval/sharpness.py.", "",
             "## 1. Accuracy and sharpness", "",
             "| run | VRMSE | to reference | gradient mean | gradient p99 | " + " | ".join(f"power {b}" for b in BAND_NAMES) + " |",
             "|---|---|---|---|---|" + "---|" * len(BANDS)]
    for label, sims in rows.items():
        to_reference = f"{mean_of(sims, 'to_reference'):.4f}" if "to_reference" in sims[0] else "-"
        lines.append(f"| {label} | {mean_of(sims, 'vrmse'):.4f} | {to_reference} | {mean_of(sims, 'grad_ratio'):.3f} | "
                     f"{mean_of(sims, 'grad_p99_ratio'):.3f} | " + " | ".join(f"{v:.3f}" for v in mean_of(sims, "power_ratio")) + " |")
    if reference is not None:
        lines += ["", f"to reference: RMS(rollout - rollout of {reference.name}) / std(truth), the VRMSE's units."]
    lines += ["", "## 2. Where the error sits: power of (prediction - truth) / power of the truth, per band", "",
              "| run | " + " | ".join(BAND_NAMES) + " |", "|---|" + "---|" * len(BANDS)]
    for label, sims in rows.items():
        lines.append(f"| {label} | " + " | ".join(f"{v:.3f}" for v in mean_of(sims, "error_per_band")) + " |")
    lines += ["", "2 = an uncorrelated prediction with the truth's power (0 + its own power + the truth's); 1 = no "
              "power at all in that band (fully blurred)."]
    if sensitivity:
        lines += ["", "## 3. Decoder sensitivity per latent channel", "",
                  f"Gaussian noise of std {noise} (DiT latent space, the same absolute size in every channel) added to "
                  "one channel of the video latents, decoded, compared with the clean decode. effect = RMS(change) / "
                  "std(truth) over the fields; fine share = share of that change in k >= 16. Sorted by the channel's std.",
                  "", "| latent channel | std | noise / std | effect (VRMSE units) | effect^2 share | fine share |",
                  "|---|---|---|---|---|---|"]
        single = [e for e in sensitivity if e["channel"] != "all"]
        total = sum(e["effect"] ** 2 for e in single)
        for e in sorted(single, key=lambda e: e["std"]):
            lines.append(f"| {e['channel']} | {e['std']:.3f} | {noise / e['std']:.2f} | {e['effect']:.4f} | "
                         f"{e['effect'] ** 2 / total:.3f} | {e['fine_share']:.3f} |")
        everything = next(e for e in sensitivity if e["channel"] == "all")
        lines += [f"| all 16 | | | {everything['effect']:.4f} | (sum of the singles: {total ** 0.5:.4f}) | "
                  f"{everything['fine_share']:.3f} |", "",
                  "Equal effect in every channel = the unweighted loss already matches the decoder (weighting per "
                  "channel would only move error around). A larger effect in the low-std channels = the decoder "
                  "amplifies them, and weighting the loss by 1 / std^2 puts the DiT's effort where the pixels need it."]
    return lines


def main():
    args = parse_args()
    runs = [Path(run) for run in args.runs]
    reference = Path(args.reference) if args.reference else None
    if reference is not None and reference.name not in {run.name for run in runs}:
        raise ValueError("--reference must be one of --runs")
    if len({run.name for run in runs}) != len(runs):
        raise ValueError("run folders must have different names")
    settings = [yaml.safe_load((run / "eval_args.yaml").read_text()) for run in runs]
    for key in ("vae_ckpt", "test_data", "frames"):
        if len({str(s[key]) for s in settings}) != 1:
            raise ValueError(f"the runs differ in {key}")
    shared = set.intersection(*({p.stem for p in (run / "rollouts").glob("*.pt")} for run in runs))
    sim_ids = sorted(shared)
    if not sim_ids:
        raise ValueError("the runs share no rollouts")

    from magi_finetuning.utils.runtime import init_magi_parallel_state
    from magi_finetuning.vae import load_vae

    torch.cuda.set_device(0)
    init_magi_parallel_state()
    vae = load_vae(settings[0]["vae_ckpt"])
    normalization = Normalization.of_vae_checkpoint(settings[0]["vae_ckpt"])
    test_set = ShockWaveDataset(settings[0]["test_data"], num_sim_frames=1 + settings[0]["frames"])

    rows = score_runs(runs, reference, vae, normalization, test_set, sim_ids)
    step = max(1, len(sim_ids) // args.sensitivity_sims) if args.sensitivity_sims else 1
    sensitivity = (decoder_sensitivity(vae, normalization, test_set, sim_ids[::step][:args.sensitivity_sims], args.noise)
                   if args.sensitivity_sims else [])

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "sharpness.json").write_text(json.dumps({"bands": BANDS, "runs": rows, "sensitivity": sensitivity,
                                                    "noise": args.noise, "reference": reference and reference.name}))
    report = "\n".join(report_lines(rows, reference, sensitivity, args.noise))
    (out / "sharpness.md").write_text(report + "\n")
    print("\n" + report + f"\n\nwritten to {out}", flush=True)


if __name__ == "__main__":
    main()
