#!/usr/bin/env python3
"""
Full-validation-set VRMSE: VAE-only reconstruction vs VAE+DiT rollout, in
BOTH decoded pixel space and raw latent space.

Answers "how much extra error does the DiT add on top of the VAE's own
reconstruction floor" -- using the SAME held-out sims the VAE was validated
on (read from <preprocessed_data_root>/split_manifest.json, written by
preprocess_dataset.py -- see that script's own comment: recorded so the val
set can be "reproduced ... without re-deriving the permutation") and the
SAME variance-normalized-RMSE formula/space VaeTrainer's own val_vrmse_chmean
uses (windinet.losses.vrms: per channel, then the mean over channels, computed
on NORMALIZED fields, matching vae_trainer.py's _validate -- NOT denormalized
physical units). The pooled VRMSE (one variance over all channels together)
was removed 2026-10-24 and is not reported any more.

Two passes per sample, same ground truth, same metric:
  - vae_only:  encode the full ground-truth sequence with the VAE, decode,
               compare -- pure autoencoding fidelity, no forecasting.
  - vae_dit:   encode only frame 0 (the initial condition), let the DiT
               roll the rest out in latent space, decode, compare -- the
               full inference-time pipeline inference_shockwave.py runs.

Both passes are reported in decoded pixel space (comparable to the VAE's
own val_vrmse_chmean) AND in raw latent space (the vae_dit pass's rollout compared
directly against the vae_only pass's own ground-truth latent, no decode in
between -- 2026-08-31: merged in from the former standalone
eval_dit_latent_vrmse.py, which existed only to avoid a second, separate
DiT rollout for this. The DiT's own training loss (flow-matching loss at
random noise levels) does not report this directly, and the pixel-space
number alone conflates DiT rollout error with the VAE decoder's own
reconstruction fidelity -- the latent-space number isolates DiT forecast
quality from that). The vae_dit pass now pulls the rollout with
output_type="latent" and decodes it explicitly (one extra decode, zero
extra transformer forward passes) instead of letting the pipeline decode
internally, purely so this same predicted latent is available for both
comparisons.

The DiT checkpoint can be a mid-training checkpoint (this run may not have
finished) -- this script does not require or check a "final" checkpoint.

Usage:
    python scripts/eval_dit_vrmse.py configs/dit/inference_dit.yaml \\
        --preprocessed_data_root finetune_vae_outputs/sng_pvc/dit_preprocessed/finetune_vae_ch6_loss_rmse_h1_256res \\
        --checkpoint /path/to/checkpoints/model_weights_step_09177.safetensors \\
        --scalar_checkpoint /path/to/checkpoints/model_weights_step_09177.state.pt \\
        --vae_checkpoint /path/to/finetune_vae_outputs/.../checkpoints/vae_shockwave_best.safetensors \\
        --num_samples 675 \\
        --save_npz_samples 3 \\
        --out_dir eval_dit_vrmse_out/

--num_samples defaults to all val_ids in the manifest (675 for the 256res
runs) -- pass a smaller number for a quick check, since each sample costs
one VAE encode + one DiT rollout + two VAE decodes.

--save_vis_samples N additionally renders GT/Prediction/Residual panels, at
--frame_numbers (default 25/50/75/100, matching the VAE/DiT trainers' own
periodic-visualization convention -- windinet.training.vae_visualization
.save_reconstruction_panels), for N samples chosen to span the evaluated
set's full gamma range (min/median/max for N=3, evenly-spaced quantiles
otherwise -- same selection vae_trainer.py's own visualization uses, per
the same PhD-advisor request that figures show the full physical regime
rather than whichever sims happen to land first in split order), for BOTH
passes -- vae_only and vae_dit -- under
<out_dir>/visualizations/{vae_only,vae_dit}/<sample_id>/frame_XXXX.png --
so the before/after gap from adding the DiT can be read off visually, same
layout the training-time panels use.

Benchmark block = the headline numbers (cfd_bench/windinet_benchmark_inference.py,
ported from Harish's dev-hr branch, 2026-10-19; made the default 2026-10-24 on
his instruction): RMSE, VRMSE (per field, then mean), 1-SSIM and MLW, each once
per sim over the scored frames, then averaged over sims, with prediction AND
target clamped to [-1, 1]. The scored frames default to every 10th frame from
40 on (40, 50, ..., 90 for a 97-frame model; 40, ..., 100 for a 101-frame
one) -- his autoregressive baseline takes 4 frames at stride 10 and predicts
every 10th frame from 40, so both models are scored on the same frames.
--score_frames overrides the frames, --no-clamp_prediction the clamp. The
*_vrmse_chmean / *_vrmse_per_channel keys keep their all-frames, unclamped
definition for comparison with older evals.
"""

import argparse
import json
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import yaml
from safetensors.torch import load_file
from torch.amp import autocast

from windinet.checkpoints import ensure_checkpoint
from windinet.config import ScalarConditioningConfig
from windinet.inference.model_loader import load_ltxv_components, select_vae_env
from windinet.inference.pipeline import LTXConditionPipeline
from windinet.losses import SSIMLoss, mlw_loss, rmse_loss, vrms_per_channel
from windinet.scalar_embeddings import ScalarEmbedding
from windinet.training.shockwave_data import (
    ShockWaveDataset,
    build_shockwave_video,
    load_channel_normalization,
    normalize_fields,
    pick_spread_ids,
    run_scalar_conditioning,
)
from windinet.training.vae_visualization import (
    denormalize_fields,
    save_reconstruction_panels,
    save_reconstruction_video,
)
from windinet.utils import get_default_device
from windinet.vae_adapter import latent_space_fingerprint

DTYPE = torch.bfloat16


# ----------------------------------------------------------------------
# Reused verbatim from scripts/inference_shockwave.py (that script is not
# an importable package -- scripts/ has no __init__.py -- so the small,
# already-tested pipeline-setup functions are copied rather than imported;
# any change there should be mirrored here).
# ----------------------------------------------------------------------

def verify_latent_space(vae_checkpoint, dit_checkpoint, stats) -> None:
    provenance_path = Path(dit_checkpoint).parent.parent / "latent_provenance.json"
    if not provenance_path.is_file():
        print(
            f"WARNING: no {provenance_path} -- cannot verify that {vae_checkpoint} is the VAE "
            "the DiT's latents were built with. A mismatch here produces wrong physics silently."
        )
        return
    expected = json.loads(provenance_path.read_text()).get("latent_fingerprint")
    if not expected:
        print(f"WARNING: {provenance_path} records no latent_fingerprint; skipping the check.")
        return
    actual = latent_space_fingerprint(
        Path(vae_checkpoint), stats["channel_mean"], stats["channel_std"], stats["normalization_clip"],
    )
    if actual != expected:
        raise SystemExit(
            f"VAE / DiT latent space mismatch.\n"
            f"  DiT was trained on latents with fingerprint : {expected}\n"
            f"  this VAE + normalization fingerprints as    : {actual}\n"
            f"  vae_checkpoint : {vae_checkpoint}\n"
            f"  provenance     : {provenance_path}\n"
            f"Point vae_checkpoint at the VAE recorded in {provenance_path}."
        )
    print(f"Latent space verified: fingerprint {actual} matches the DiT's training latents")


def make_pipe(model_source, device):
    c = load_ltxv_components(model_source=model_source, transformer_dtype=DTYPE, vae_dtype=DTYPE)
    pipe = LTXConditionPipeline(
        scheduler=deepcopy(c.scheduler), vae=c.vae, text_encoder=None, tokenizer=None, transformer=c.transformer,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    pipe.vae.requires_grad_(False)
    pipe.transformer.requires_grad_(False)
    # Our fields are already in [-1, 1] via the CFD channel stats -- see
    # inference_shockwave.py's own comment on why do_normalize is disabled.
    pipe.video_processor.register_to_config(do_normalize=False)
    return pipe


def load_transformer_weights(pipe, checkpoint):
    print(f"Loading transformer: {checkpoint}")
    sd = load_file(str(checkpoint))
    if any(k.startswith("module.") for k in sd):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    if any(k.startswith("transformer.") for k in sd):
        sd = {k.replace("transformer.", "", 1): v for k, v in sd.items() if k.startswith("transformer.")}
    pipe.transformer.load_state_dict(sd, strict=False)


def load_scalar_embedding(checkpoint, scalar_cfg, device):
    print(f"Loading scalar embedding: {checkpoint}")
    emb = ScalarEmbedding(scalar_cfg)
    emb.load_state_dict(load_file(str(checkpoint)))
    return emb.to(device=device, dtype=DTYPE).eval()


def build_initial_condition(sample, stats, device):
    fields = torch.stack([sample[name][0] for name in stats["channel_names"]]).unsqueeze(0)  # [1, C, H, W]
    fields = normalize_fields(fields, stats["channel_mean"], stats["channel_std"], stats["normalization_clip"])
    return fields.unsqueeze(1).to(device=device, dtype=DTYPE)  # [B, F, C, H, W]


# ----------------------------------------------------------------------
# VAE-only encode/decode, copied from VaeTrainer._encode/_decode
# (windinet/training/vae_trainer.py) so the "VAE-only" pass matches the
# exact rescaling (latents_mean/std, scaling_factor, default_temb) that
# produced the already-committed VAE val numbers -- reimplementing this
# math independently would risk a silent mismatch against those numbers.
# ----------------------------------------------------------------------

def vae_encode(vae, video: torch.Tensor) -> torch.Tensor:
    """video: [B, C, F, H, W], normalized. Returns rescaled latents."""
    out = vae.encode(video)
    posterior_mean = out.latent_dist.mean
    norm_mean = vae.latents_mean.view(1, -1, 1, 1, 1).to(posterior_mean.device, posterior_mean.dtype)
    norm_std = vae.latents_std.view(1, -1, 1, 1, 1).to(posterior_mean.device, posterior_mean.dtype)
    sf = float(getattr(vae.config, "scaling_factor", 1.0))
    return (posterior_mean - norm_mean) * sf / norm_std


def vae_decode(vae, latents: torch.Tensor, default_temb: float) -> torch.Tensor:
    mean = vae.latents_mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    std = vae.latents_std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    sf = float(getattr(vae.config, "scaling_factor", 1.0))
    z = latents * std / sf + mean
    temb = torch.full((z.shape[0],), default_temb, device=z.device, dtype=z.dtype)
    return vae.decode(z, temb=temb, return_dict=True).sample


def pad_to(t: torch.Tensor, n_frames: int) -> torch.Tensor:
    """Trim a [B,C,F,H,W] tensor's frame dim to n_frames (both passes may
    return a few more frames than the ground truth due to 8k+1 padding)."""
    return t[:, :, :n_frames]


def trim_latent_frames(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """[B, C, F, H, W] pair -> both trimmed to the shorter F. The DiT
    rollout's latent and the ground-truth encode's latent are not
    guaranteed to land on the exact same latent frame count -- same
    alignment issue pad_to() handles in pixel space, here in latent space."""
    n = min(a.shape[2], b.shape[2])
    return a[:, :, :n], b[:, :, :n]


BENCHMARK_METRICS = ("rmse", "vrmse", "ssim", "mlw")
# Per-frame curve: MLW has a temporal wavelet term, so it is undefined on one frame.
PER_FRAME_METRICS = ("rmse", "vrmse", "ssim")


def benchmark_metrics(pred: torch.Tensor, target: torch.Tensor, ssim_fn: SSIMLoss) -> dict:
    """RMSE, VRMSE (per field, then mean), 1-SSIM, MLW for one sim, [1, C, T, H, W] restricted to the scored frames.
    Same definitions as cfd_bench/windinet_benchmark_inference.py; pred and target arrive clamped to [-1, 1]
    unless --no-clamp_prediction."""
    return {
        "rmse": float(rmse_loss(pred, target).item()),
        "vrmse": float(vrms_per_channel(pred, target).mean().item()),  # per field, then mean (Li eq. 5.11, The Well)
        "ssim": float(ssim_fn(pred, target).item()),
        "mlw": float(mlw_loss(pred, target).item()),
    }


def sync(device: torch.device) -> None:
    """Wait for queued GPU work, so wall-clock timings cover it."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "xpu":
        torch.xpu.synchronize(device)


def per_frame_benchmark_metrics(pred: torch.Tensor, target: torch.Tensor, ssim_fn: SSIMLoss) -> dict:
    """PER_FRAME_METRICS for each scored frame on its own, {metric: [one value per scored frame]}; the VRMSE
    variance is that frame's spatial variance (The Well's per-step convention)."""
    out = {m: [] for m in PER_FRAME_METRICS}
    for t in range(pred.shape[2]):
        p, g = pred[:, :, t:t + 1], target[:, :, t:t + 1]
        out["rmse"].append(float(rmse_loss(p, g).item()))
        out["vrmse"].append(float(vrms_per_channel(p, g).mean().item()))
        out["ssim"].append(float(ssim_fn(p, g).item()))
    return out


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path, help="Path to inference YAML config")
    ap.add_argument("--preprocessed_data_root", type=Path, required=True,
                     help="Same preprocessed_data_root the DiT training config used -- gives the "
                          "h5 path and the exact held-out val_ids via split_manifest.json")
    ap.add_argument("--num_samples", type=int, default=None,
                     help="How many val sims to evaluate (default: all in the manifest)")
    ap.add_argument("--checkpoint", type=Path, default=None, help="Override transformer checkpoint")
    ap.add_argument("--scalar_checkpoint", type=Path, default=None, help="Override scalar embedding checkpoint")
    ap.add_argument("--untrained_dit", action="store_true",
                     help="'DiT never finetuned' control: skip --checkpoint/--scalar_checkpoint entirely "
                          "and use the stock pretrained LTX-Video transformer as-is (no finetuning on "
                          "shockwave latents) plus a freshly random-initialized ScalarEmbedding (there is "
                          "no pretrained scalar-conditioning module to fall back to -- LTX-Video has no "
                          "notion of gamma conditioning). Answers the DiT-side analog of the un-finetuned-"
                          "VAE control: how much does DiT training actually add on top of just running the "
                          "untouched pretrained transformer on these latents.")
    ap.add_argument("--vae_checkpoint", type=Path, default=None, help="Override inflate-mode VAE checkpoint")
    ap.add_argument("--normalization", type=Path, default=None,
                     help="Override normalization source (a training_config.yaml or stats yaml) -- "
                          "point this at the VAE run's own training_config.yaml for that run's exact stats")
    ap.add_argument("--num_inference_steps", type=int, default=None, help="Override denoising steps")
    ap.add_argument("--guidance_scale", type=float, default=None, help="Override guidance scale")
    ap.add_argument("--default_temb", type=float, default=0.0,
                     help="temb passed to the VAE-only decode pass (matches adapter.default_temb "
                          "in every VAE training config seen in this repo so far)")
    ap.add_argument("--save_vis_samples", type=int, default=3,
                     help="Render GT/Prediction/Residual panels (both passes) for the first N samples")
    ap.add_argument("--frame_numbers", type=int, nargs="+", default=[25, 50, 75, 97],
                     help="1-indexed frame numbers to render panels for (default matches "
                          "DitVisualizationConfig's own default)")
    ap.add_argument("--dpi", type=int, default=150, help="Panel image DPI")
    ap.add_argument("--save_video", action="store_true",
                     help="Also write a GT/Prediction/Residual MP4 over every frame for each visualized sample "
                          "(<out_dir>/visualizations/{vae_only,vae_dit}/<sample_id>/video.mp4)")
    ap.add_argument("--video_fps", type=int, default=8, help="--save_video frame rate")
    ap.add_argument("--out_dir", type=Path, default=Path("outputs/eval_dit_vrmse"))
    ap.add_argument("--test_h5", type=Path, default=None,
                     help="Evaluate every sim of this standalone test.h5 instead of the manifest's val_ids "
                          "(Chapter 6: the same 256x256_ds/test.h5 scripts/eval_vae_test.py uses). The "
                          "manifest is still read, for provenance only.")
    ap.add_argument("--save_npz_samples", type=int, default=0,
                     help="Save fields (gt_raw float32; gt_norm, vae_only_norm, vae_dit_norm float16) of N "
                          "gamma-spread sims at --npz_frames for thesis figures; same picks as "
                          "eval_vae_test.py on the same id list")
    ap.add_argument("--npz_frames", type=int, nargs="+", default=[0, 25, 50, 75, 96],
                     help="0-indexed frames saved per npz sample (frame 0 = the conditioning IC)")
    ap.add_argument("--metric_frames", type=int, default=None,
                     help="Score only the first N frames (pixel space; latent space keeps every latent "
                          "frame). For comparing a 101-frame model against a 97-frame one on the same "
                          "frames: the model still runs on its own full length, only the metric is cropped.")
    ap.add_argument("--npz_dir", type=Path, default=None, help="Where the npz samples go (default: <out_dir>/samples)")
    ap.add_argument("--sim_ids", type=str, nargs="+", default=None,
                     help="Evaluate exactly these sim ids of the h5 (the manifest's data_root, or --test_h5) "
                          "and save an npz for each, e.g. a hand-picked sim for a thesis figure. They may "
                          "come from any split; the split of each id is printed. The averages then cover "
                          "only these sims, not a held-out set.")
    ap.add_argument("--score_frames", type=int, nargs="+", default=None,
                     help="0-indexed stored frames the benchmark metrics are computed on (default: every 10th "
                          "frame from 40, as cfd_bench scores them: 40 50 ... 90 for 97 frames, ... 100 for 101; "
                          "with frame_stride > 1 they must be multiples of it)")
    ap.add_argument("--clamp_prediction", action=argparse.BooleanOptionalAction, default=True,
                     help="Clamp prediction and target to [-1, 1] before the benchmark metrics (default on, "
                          "as in cfd_bench); --no-clamp_prediction turns it off")
    return ap.parse_args()


def load_config(args):
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    for key in ("checkpoint", "scalar_checkpoint", "vae_checkpoint", "normalization",
                "num_inference_steps", "guidance_scale"):
        value = getattr(args, key, None)
        if value is not None:
            cfg[key] = str(value) if isinstance(value, Path) else value
    return cfg


@torch.no_grad()
def main():
    args = parse_args()
    cfg = load_config(args)
    device = get_default_device()

    manifest_path = args.preprocessed_data_root / "split_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    norm_payload = json.loads((args.preprocessed_data_root / "normalization.json").read_text())
    # Which dataset the latents were encoded from (preprocess_dataset.py records
    # the VAE run's data.dataset; older latents predate it and are euler_mq).
    dataset_name = norm_payload.get("dataset") or "euler_mq"
    # Temporal subsampling the latents were built with (data.frame_stride);
    # sequence index k is stored frame k * frame_stride.
    frame_stride = int(norm_payload.get("frame_stride") or 1)
    print(f"Dataset: {dataset_name}, frame_stride {frame_stride}")
    if args.test_h5 is not None:
        # val_ids keeps its name below but holds the test ids from here on.
        h5_path = str(args.test_h5)
        val_ids = list(ShockWaveDataset(h5_path, dataset=dataset_name, frame_stride=frame_stride).ids)
        print(f"Standalone test set: {len(val_ids)} sims from {h5_path} (manifest {manifest_path} not used for ids)")
    else:
        h5_path = manifest["data_root"]
        val_ids = manifest["val_ids"]
        print(f"Held-out split: {len(val_ids)} sims (seed={manifest['split_seed']}) from {manifest_path}")
    print(f"h5: {h5_path}")

    if args.sim_ids:
        for sid in args.sim_ids:
            split = next((k for k in ("train_ids", "val_ids") if sid in manifest.get(k, [])), "not in manifest")
            print(f"--sim_ids: {sid} ({split})")
        val_ids = list(args.sim_ids)
    if args.num_samples is not None:
        val_ids = val_ids[: args.num_samples]
    print(f"Evaluating {len(val_ids)} sim(s)")

    vis_ids = pick_spread_ids(val_ids, args.save_vis_samples, dataset_name)
    if vis_ids:
        print(f"Visualizing {len(vis_ids)} sample(s) spanning the gamma/Mach range: {sorted(vis_ids)}")
    npz_ids = set(val_ids) if args.sim_ids else pick_spread_ids(val_ids, args.save_npz_samples, dataset_name)
    npz_dir = args.npz_dir or args.out_dir / "samples"
    if npz_ids:
        print(f"Saving npz fields for {sorted(npz_ids)} (frames {args.npz_frames}) to {npz_dir}")

    model_source = cfg.get("model_source", "LTXV_2B_0.9.6_DEV")
    checkpoint = None if args.untrained_dit else ensure_checkpoint(cfg["checkpoint"])
    scalar_checkpoint = None if args.untrained_dit else ensure_checkpoint(cfg["scalar_checkpoint"])
    num_inference_steps = cfg.get("num_inference_steps", 2)
    guidance_scale = cfg.get("guidance_scale", 1.0)
    # Latents encoded from truncated sims (preprocess_dataset.py --num-sim-frames)
    # record that length; the eval must use the same one, and the DiT then
    # generates the matching 8k+1 length instead of the config's 105.
    num_sim_frames = norm_payload.get("num_sim_frames")
    if num_sim_frames is not None:
        num_frames = ((num_sim_frames - 1 + 7) // 8) * 8 + 1
        print(f"Latents built from {num_sim_frames}-frame sims -> generating {num_frames} frames")
    else:
        num_frames = cfg.get("num_frames", 105)
    if args.metric_frames is not None:
        print(f"Scoring only the first {args.metric_frames} frames")
    image_cond_noise_scale = cfg.get("image_cond_noise_scale", 0.0)
    seed = cfg.get("seed", 42)

    vae_ckpt = cfg.get("vae_checkpoint")
    if not vae_ckpt:
        raise SystemExit("vae_checkpoint is required (inflate-mode finetuned checkpoint)")
    select_vae_env(ensure_checkpoint(vae_ckpt))  # must run BEFORE make_pipe: sets which VAE it loads

    stats = load_channel_normalization(cfg["normalization"])
    channel_names = stats["channel_names"]
    n_ch = len(channel_names)
    print(f"Normalization from {cfg['normalization']}: channels={channel_names} clip={stats['normalization_clip']}")
    if args.untrained_dit:
        print("--untrained_dit: skipping latent-space provenance check (no DiT training happened to verify against)")
    else:
        verify_latent_space(ensure_checkpoint(vae_ckpt), checkpoint, stats)

    sc = run_scalar_conditioning(checkpoint, cfg.get("scalar_conditioning", {}))
    print(f"Scalar conditioning: {sc.get('scalar_names')} ranges {sc.get('scalar_ranges')}")
    scalar_cfg = ScalarConditioningConfig(
        enabled=True,
        scalar_names=sc.get("scalar_names", ["gamma"]),
        scalar_ranges={k: tuple(v) for k, v in sc.get("scalar_ranges", {"gamma": [1.0, 2.0]}).items()},
        embedding_dim=sc.get("embedding_dim", 4096),
        num_tokens_per_scalar=sc.get("num_tokens_per_scalar", 4),
    )

    dataset = ShockWaveDataset(h5_path, num_sim_frames=num_sim_frames, dataset=dataset_name, frame_stride=frame_stride)
    missing = [s for s in val_ids if s not in dataset.ids]
    if missing:
        raise SystemExit(f"val_ids not found in {h5_path}: {missing[:5]}...")

    pipe = make_pipe(model_source, device)  # pipe.vae is now the finetuned VAE from vae_ckpt
    if args.untrained_dit:
        print("--untrained_dit: leaving transformer at its stock pretrained weights (no checkpoint loaded)")
        scalar_emb = ScalarEmbedding(scalar_cfg).to(device=device, dtype=DTYPE).eval()
        print("--untrained_dit: using a freshly random-initialized ScalarEmbedding (no pretrained equivalent exists)")
    else:
        load_transformer_weights(pipe, checkpoint)
        scalar_emb = load_scalar_embedding(scalar_checkpoint, scalar_cfg, device)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    per_sample = []
    ssim_fn = SSIMLoss(channels=n_ch).to(device)
    bench_values = {"vae_only": {m: [] for m in BENCHMARK_METRICS}, "vae_dit": {m: [] for m in BENCHMARK_METRICS}}
    # [sim][scored frame] per label and metric
    per_frame_values = {"vae_only": {m: [] for m in PER_FRAME_METRICS}, "vae_dit": {m: [] for m in PER_FRAME_METRICS}}
    sum_channel = {"vae_only": [0.0] * n_ch, "vae_dit": [0.0] * n_ch}
    sum_lat_rmse, sum_lat_mse = 0.0, 0.0
    sum_lat_channel = None  # lazily sized to the latent's own channel count on first sample
    sum_lat_channel_mse = None
    gen_seconds = []  # per sim: frame-0 condition -> DiT sampling -> decoded trajectory, metrics excluded

    for i, sid in enumerate(val_ids):
        idx = dataset.ids.index(sid)
        sample = dataset[idx]
        H, W = sample[channel_names[0]].shape[-2:]
        orig_F = sample[channel_names[0]].shape[0]
        score_F = min(orig_F, args.metric_frames or orig_F)

        # Ground truth, normalized, [1, C, orig_F, H, W] -- same call
        # preprocess_dataset.py used to build the VAE's own training/eval input.
        gt_video = build_shockwave_video(
            sample, device=device, channel_mean=stats["channel_mean"],
            channel_std=stats["channel_std"], normalization_clip=stats["normalization_clip"],
            channel_order=channel_names,
        )
        target = pad_to(gt_video, orig_F).float()

        # --- VAE-only pass: encode + decode the full ground-truth sequence ---
        # gt_latent doubles as the "correct answer" the vae_dit pass's own
        # rollout latent is compared against below -- the VAE encoder's
        # direct output, zero DiT involvement.
        gt_latent = vae_encode(pipe.vae, gt_video.to(DTYPE))
        vae_recon = vae_decode(pipe.vae, gt_latent, args.default_temb).float()
        vae_recon = pad_to(vae_recon, orig_F)

        vo_channel = vrms_per_channel(vae_recon[:, :, :score_F], target[:, :, :score_F]).tolist()

        # --- VAE+DiT pass: condition on frame 0 only, roll the rest out ---
        sync(device)
        t_gen = time.perf_counter()
        cond_video = build_initial_condition(sample, stats, device)
        scalar_values = [sample["meta"][n] for n in scalar_cfg.scalar_names]
        scalars = torch.tensor([scalar_values], device=device, dtype=DTYPE)
        prompt_embeds = scalar_emb(scalars)
        prompt_mask = torch.ones(1, prompt_embeds.shape[1], device=device, dtype=torch.long)

        g = torch.Generator(device=device).manual_seed(seed + i)
        with autocast(device.type, dtype=DTYPE, enabled=(device.type in ("cuda", "xpu"))):
            out = pipe(
                prompt=None, negative_prompt=None,
                video=cond_video, frame_index=0, strength=1.0,
                width=W, height=H,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                image_cond_noise_scale=image_cond_noise_scale,
                generator=g,
                output_reference_comparison=False,
                prompt_embeds=prompt_embeds,
                prompt_attention_mask=prompt_mask,
                negative_prompt_embeds=torch.zeros_like(prompt_embeds),
                negative_prompt_attention_mask=prompt_mask.clone(),
                output_type="latent",
            )
        # out.frames is the raw predicted latent, [B, C, F_lat, H_lat, W_lat],
        # in the SAME rescaled space vae_encode() returns (LTXConditionPipeline's
        # output_type=="latent" branch returns the unpacked latents BEFORE
        # _denormalize_latents, which is exactly the inverse of vae_encode's
        # own rescale) -- directly comparable to gt_latent with no further
        # conversion, and decodable with the same vae_decode() helper used
        # for the vae_only pass above.
        dit_pred_latent = out.frames.float()
        sync(device)
        gen_s = time.perf_counter() - t_gen  # decode below is added to it
        pred_lat, cmp_gt_lat = trim_latent_frames(dit_pred_latent, gt_latent.float())

        lat_rmse = float(rmse_loss(pred_lat, cmp_gt_lat).item())
        lat_sq_err = (pred_lat - cmp_gt_lat) ** 2
        lat_mse = float(lat_sq_err.mean().item())
        lat_channel = vrms_per_channel(pred_lat, cmp_gt_lat)
        lat_vrmse = float(lat_channel.mean().item())
        lat_channel_mse = lat_sq_err.mean(dim=(0, 2, 3, 4))
        if sum_lat_channel is None:
            sum_lat_channel = torch.zeros(lat_channel.shape[0])
            sum_lat_channel_mse = torch.zeros(lat_channel_mse.shape[0])
        sum_lat_channel += lat_channel.cpu()
        sum_lat_channel_mse += lat_channel_mse.cpu()

        # Decoded pixel space -- same predicted latent, one extra decode
        # (zero extra transformer forward passes), directly comparable to
        # vae_recon/target above. out.frames: [B, F, C, H, W] in the SAME
        # normalized [-1,1] space as `target` (do_normalize=False on the
        # video processor) -- do not denormalize here, VRMSE is a
        # normalized-space metric.
        sync(device)
        t_dec = time.perf_counter()
        dit_full = pad_to(vae_decode(pipe.vae, pred_lat.to(DTYPE), args.default_temb).float(), orig_F)
        sync(device)
        gen_seconds.append(gen_s + time.perf_counter() - t_dec)
        n_px = min(dit_full.shape[2], target.shape[2], score_F)
        dit_pred, target_cmp = dit_full[:, :, :n_px], target[:, :, :n_px]

        vd_channel = vrms_per_channel(dit_pred, target_cmp).tolist()
        vo_chmean, vd_chmean = sum(vo_channel) / n_ch, sum(vd_channel) / n_ch

        # Benchmark block: all four metrics on the scored frames only (not
        # cropped by --metric_frames: the scored frames pick their own).
        # Scored frames are stored-frame numbers (the physical times cfd_bench
        # scores), mapped to sequence indices through frame_stride, so a
        # subsampled model is scored at the same times as a stride-1 one.
        n_avail = min(dit_full.shape[2], vae_recon.shape[2], target.shape[2])
        last_stored = (n_avail - 1) * frame_stride
        score_frames = (args.score_frames if args.score_frames is not None
                        else list(range(40, last_stored + 1, 10)))
        if max(score_frames) > last_stored:
            raise SystemExit(f"--score_frames {score_frames} exceeds the last available frame {last_stored} for {sid}")
        if any(f % frame_stride for f in score_frames):
            raise SystemExit(f"--score_frames {score_frames} are not all multiples of frame_stride {frame_stride}")
        frame_idx = torch.as_tensor([f // frame_stride for f in score_frames], device=target.device)
        target_scored = target.index_select(2, frame_idx)
        vae_only_scored = vae_recon.index_select(2, frame_idx)
        vae_dit_scored = dit_full.index_select(2, frame_idx)
        if args.clamp_prediction:
            # The target is already in [-1, 1] (build_shockwave_video clamps);
            # clamped again so the block does not depend on that.
            target_scored = target_scored.clamp(-1.0, 1.0)
            vae_only_scored = vae_only_scored.clamp(-1.0, 1.0)
            vae_dit_scored = vae_dit_scored.clamp(-1.0, 1.0)
        bench = {
            "vae_only": benchmark_metrics(vae_only_scored, target_scored, ssim_fn),
            "vae_dit": benchmark_metrics(vae_dit_scored, target_scored, ssim_fn),
        }
        for label, values in bench.items():
            for m in BENCHMARK_METRICS:
                bench_values[label][m].append(values[m])
        per_frame = {
            "vae_only": per_frame_benchmark_metrics(vae_only_scored, target_scored, ssim_fn),
            "vae_dit": per_frame_benchmark_metrics(vae_dit_scored, target_scored, ssim_fn),
        }
        for label, values in per_frame.items():
            for m in PER_FRAME_METRICS:
                per_frame_values[label][m].append(values[m])

        per_sample.append({
            "id": sid, **{k: float(v) for k, v in sample["meta"].items()},
            "vae_only_vrmse_chmean": vo_chmean, "vae_dit_vrmse_chmean": vd_chmean,
            "vae_only_per_channel": dict(zip(channel_names, vo_channel)),
            "vae_dit_per_channel": dict(zip(channel_names, vd_channel)),
            "latent_vrmse_chmean": lat_vrmse, "latent_rmse": lat_rmse, "latent_mse": lat_mse,
            "benchmark": bench,
            "benchmark_per_frame": per_frame,
            "generation_seconds": gen_seconds[-1],
        })
        for c in range(n_ch):
            sum_channel["vae_only"][c] += vo_channel[c]
            sum_channel["vae_dit"][c] += vd_channel[c]
        sum_lat_rmse += lat_rmse
        sum_lat_mse += lat_mse

        print(f"[{i+1}/{len(val_ids)}] {sid}: benchmark vrmse vae_only={bench['vae_only']['vrmse']:.5f}  "
              f"vae+dit={bench['vae_dit']['vrmse']:.5f}  (all frames chmean {vo_chmean:.5f} / {vd_chmean:.5f})")

        if sid in npz_ids:
            frames = [f for f in args.npz_frames if f < n_px]
            gt_raw = torch.stack([sample[name] for name in channel_names])  # [C, F, H, W]
            npz_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                npz_dir / f"{sid}.npz",
                frames=np.array(frames),
                channel_order=np.array(channel_names),
                **{k: float(v) for k, v in sample["meta"].items()},
                gt_raw=gt_raw[:, frames].numpy().astype(np.float32),  # physical units overflow float16 (SRBA pressure ~1e5-1e6)
                gt_norm=target_cmp[0][:, frames].cpu().numpy().astype(np.float16),
                vae_only_norm=vae_recon[0][:, frames].cpu().numpy().astype(np.float16),
                vae_dit_norm=dit_pred[0][:, frames].cpu().numpy().astype(np.float16),
                channel_mean=np.array(stats["channel_mean"]),
                channel_std=np.array(stats["channel_std"]),
                normalization_clip=float(stats["normalization_clip"]),
                vae_only_vrmse_chmean=vo_chmean,
                vae_dit_vrmse_chmean=vd_chmean,
            )

        if sid in vis_ids:
            # Physical-units ground truth, straight from the dataset -- same
            # approach dit_visualization.py's own periodic panels use (not
            # `target`, which is the normalized-space vrmse comparand above).
            gt_physical = torch.stack([sample[name] for name in channel_names]).unsqueeze(0)  # [1,C,F,H,W]
            gt_physical = pad_to(gt_physical, orig_F)

            vae_recon_physical = denormalize_fields(
                vae_recon, stats["channel_mean"], stats["channel_std"], stats["normalization_clip"],
            )
            dit_pred_physical = denormalize_fields(
                dit_pred, stats["channel_mean"], stats["channel_std"], stats["normalization_clip"],
            )

            for label, pred_physical in (("vae_only", vae_recon_physical), ("vae_dit", dit_pred_physical)):
                save_reconstruction_panels(
                    prediction=pred_physical[0],
                    target=gt_physical[0],
                    sample_id=sid,
                    label=label,
                    frame_numbers=args.frame_numbers,
                    channel_names=channel_names,
                    output_dir=args.out_dir,
                    dpi=args.dpi,
                )
                if args.save_video:
                    save_reconstruction_video(
                        prediction=pred_physical[0],
                        target=gt_physical[0],
                        sample_id=sid,
                        label=label,
                        channel_names=channel_names,
                        output_dir=args.out_dir,
                        fps=args.video_fps,
                    )

        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "xpu":
            torch.xpu.empty_cache()

    n = len(val_ids)
    mean_lat_channel = (sum_lat_channel / n).tolist()
    ranked = sorted(range(len(mean_lat_channel)), key=lambda c: mean_lat_channel[c], reverse=True)
    summary = {
        "n_samples": n,
        "split": "test" if args.test_h5 is not None else "val",
        "h5": h5_path,
        "checkpoint": str(checkpoint) if checkpoint else "untrained (stock pretrained transformer + random ScalarEmbedding)",
        "vae_checkpoint": str(vae_ckpt),
        "channel_names": channel_names,
        "num_sim_frames": num_sim_frames,
        "num_frames_generated": num_frames,
        "metric_frames": args.metric_frames,
        "num_inference_steps": num_inference_steps,
        "vae_only_vrmse_per_channel": {name: sum_channel["vae_only"][c] / n for c, name in enumerate(channel_names)},
        "vae_dit_vrmse_per_channel": {name: sum_channel["vae_dit"][c] / n for c, name in enumerate(channel_names)},
        # All frames, unclamped: mean of the per-channel VRMSEs (The Well's
        # convention), same as the VAE trainer's val_vrmse_chmean. Kept for
        # comparison with evals from before 2026-10-24; the headline numbers
        # are the benchmark block's.
        "vae_only_vrmse_chmean": sum(sum_channel["vae_only"]) / (n_ch * n),
        "vae_dit_vrmse_chmean": sum(sum_channel["vae_dit"]) / (n_ch * n),
        # Mean over the latent channels of the per-channel latent VRMSE.
        "latent_vrmse_chmean": sum(mean_lat_channel) / len(mean_lat_channel),
        "latent_rmse_mean": sum_lat_rmse / n,
        # Raw (unnormalized) latent MSE, per sim then averaged: the quantity the
        # DiT's flow-matching objective is closest to. Like latent_vrmse it is
        # in this VAE's own latent space -- compare DiTs on the same VAE only.
        "latent_mse_mean": sum_lat_mse / n,
        "latent_vrmse_per_channel": mean_lat_channel,
        "latent_mse_per_channel": (sum_lat_channel_mse / n).tolist(),
        "worst_5_latent_channels": [{"channel": c, "vrmse": mean_lat_channel[c]} for c in ranked[:5]],
        "best_5_latent_channels": [{"channel": c, "vrmse": mean_lat_channel[c]} for c in ranked[-5:]],
        "benchmark": {
            "scored_frames": score_frames,  # stored-frame numbers, same for every sim
            "frame_stride": frame_stride,
            "clamp_prediction": bool(args.clamp_prediction),
            "notes": "each metric once per sim over the scored frames and all channels, then the plain "
                     "mean over sims; vrmse is per field then mean; ssim is 1 - SSIM; prediction and "
                     "target clamped to [-1, 1] if clamp_prediction",
            **{
                label: {
                    m: {"mean": float(torch.tensor(v, dtype=torch.float64).mean()),
                        "std_over_sims": float(torch.tensor(v, dtype=torch.float64).std(unbiased=False))}
                    for m, v in values.items()
                }
                for label, values in bench_values.items()
            },
            # Same scored frames, each scored on its own (list index = scored_frames index), mean and
            # std over sims. No mlw: it needs more than one frame.
            "per_frame": {
                label: {
                    m: {"mean": torch.tensor(v, dtype=torch.float64).mean(0).tolist(),
                        "std_over_sims": torch.tensor(v, dtype=torch.float64).std(0, unbiased=False).tolist()}
                    for m, v in values.items()
                }
                for label, values in per_frame_values.items()
            },
        },
        # Wall clock for one trajectory: frame-0 conditioning, DiT sampling and VAE decode, one GPU,
        # batch 1, metrics and the VAE-only pass excluded. The first sim includes warm-up.
        "generation_time": {
            "device": (torch.cuda.get_device_name(device) if device.type == "cuda" else device.type),
            "num_inference_steps": num_inference_steps,
            "total_seconds": sum(gen_seconds),
            "seconds_per_trajectory": sum(gen_seconds) / n,
            "median_seconds_per_trajectory": float(np.median(gen_seconds)),
            "min_seconds": min(gen_seconds),
            "max_seconds": max(gen_seconds),
            "seconds_per_trajectory_without_first": (sum(gen_seconds[1:]) / (n - 1)) if n > 1 else None,
            "first_seconds": gen_seconds[0],
        },
        "per_sample": per_sample,
    }
    (args.out_dir / "vrmse_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 60)
    print(f"n={n} sims")
    print(f"Benchmark metrics (cfd_bench definitions), frames {summary['benchmark']['scored_frames']}, "
          f"clamp={summary['benchmark']['clamp_prediction']}:")
    for label in ("vae_only", "vae_dit"):
        print(f"  {label:9s} " + "  ".join(
            f"{m}={summary['benchmark'][label][m]['mean']:.5f}" for m in BENCHMARK_METRICS))
    print("  per frame (vae_dit): " + "  ".join(
        f"f{f}: vrmse={v:.4f}" for f, v in zip(score_frames, summary["benchmark"]["per_frame"]["vae_dit"]["vrmse"]["mean"])))
    gt = summary["generation_time"]
    print(f"Generation time ({gt['device']}, {gt['num_inference_steps']} steps): {gt['total_seconds']:.1f} s for "
          f"{n} trajectories = mean {gt['seconds_per_trajectory']:.2f} s/trajectory, median "
          f"{gt['median_seconds_per_trajectory']:.2f} s (range {gt['min_seconds']:.2f}-{gt['max_seconds']:.2f})")
    print(f"\nAll frames, unclamped:")
    print(f"  VAE-only  vrmse_chmean : {summary['vae_only_vrmse_chmean']:.5f}")
    print(f"  VAE+DiT   vrmse_chmean : {summary['vae_dit_vrmse_chmean']:.5f}")
    for name in channel_names:
        vo = summary["vae_only_vrmse_per_channel"][name]
        vd = summary["vae_dit_vrmse_per_channel"][name]
        print(f"    {name:12s} vae_only={vo:.5f}  vae+dit={vd:.5f}  delta={vd - vo:+.5f}")
    print(f"\nDiT rollout vs VAE-encoder ground truth, LATENT space (isolated from VAE decode):")
    print(f"  latent_vrmse_chmean : {summary['latent_vrmse_chmean']:.5f}")
    print(f"  latent_rmse         : {summary['latent_rmse_mean']:.5f}")
    print(f"  latent_mse          : {summary['latent_mse_mean']:.5f}")
    print(f"  worst 5 latent channels: {summary['worst_5_latent_channels']}")
    print(f"\nSaved: {args.out_dir / 'vrmse_summary.json'}")
    if vis_ids:
        print(f"Saved GT/Prediction/Residual panels for {len(vis_ids)} sample(s) spanning the gamma "
              f"range, frames {args.frame_numbers}, under:")
        print(f"  {args.out_dir / 'visualizations' / 'vae_only'}/<sample_id>/frame_XXXX.png")
        print(f"  {args.out_dir / 'visualizations' / 'vae_dit'}/<sample_id>/frame_XXXX.png")


if __name__ == "__main__":
    main()
