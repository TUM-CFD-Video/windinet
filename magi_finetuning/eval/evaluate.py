"""Evaluate a trained VAE + DiT pair on test.h5: rollouts, figures, videos, VRMSE tables.

Conditioning: frame 0 is given, frames 1..F are generated (F = 100 by default).

Benchmark (first table in results.md): Harish's cfd_bench metrics -- RMSE, VRMSE (per field, then mean),
1 - SSIM, MLW -- each once per sim over the trajectory frames --score-frames (default every 10th frame 10..90:
10, 20, ..., 90; Harish, 2026-10-28), then the mean over sims; prediction and target clamped to [-1, 1].

For 3 test cases chosen by gamma (lowest, closest to the mean, highest):
    visualizations/panels/<case>/   GT | prediction | residual at selected frames (physical units)
    videos/<case>_<channel>.mp4     frame 0 (given) + frames 1..F, one video per channel in the standardized
                                    format shared with LTX / Wan (rollout_video.py): GT | prediction | |error| / sigma
                                    with the training-set colour limits of --video-stats; <video>.json per-frame numbers
    videos/<case>_vae_<channel>.mp4 the same for the VAE round trip (encode -> decode the true frames)
    error_over_time.png             error per generated frame
For --metric-sims test simulations (default: all 500), VRMSE mean ± std over frames 1..F of
    VAE round trip vs dataset      (encode -> decode the ground truth: the VAE's own error)
    DiT rollout vs dataset         (from frame 0 only)
    DiT rollout vs VAE round trip  (the DiT's error alone)
VRMSE = per channel in NORMALIZED space, then channel mean; results.md also lists
the physical-space version (identical except where the +-1 normalization clip bites).

Large evaluations are sharded: run --shard i --num-shards N on N GPUs (each writes scores/<sim>.json and
skips sims already scored, so it can resume), then once with --report-only to merge into the tables.
jobs/jupiter/magi1/evaluate_magi1.sbatch does all of that.

    python -m magi_finetuning.eval.evaluate --dit-ckpt <ckpt> [--vae-ckpt <vae>] [--frames 100]
    python -m magi_finetuning.eval.evaluate --vae-only --vae-ckpt <vae> --out <dir>     (VAE round trip only)
--vae-ckpt / --frames default to the DiT run's args.yaml. Fields are normalized with the VAE's own
normalization (stored in its checkpoint, data.Normalization).

When every requested sim is scored, the report step copies the VAE + DiT checkpoints and the results to
settings.FINAL_MODELS/<name> on SCRATCH (archive_run; --no-archive to skip).

Read `run()` first; every helper below it does one thing.
"""

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from magi_finetuning.data import Normalization, build_latent_sequence, encode_image, stratified_subset
from magi_finetuning.eval.report import (BENCHMARK_METRICS, benchmark_metrics, inference_summary, save_case_panels,
                                         save_error_over_time, vrmse, vrmse_per_frame, write_tables)
from magi_finetuning.eval.rollout_video import STATS_FILE, load_stats, save_rollout_videos
from magi_finetuning.settings import CHUNK_WIDTH, DEFAULT_FRAMES, FINAL_MODELS, OUTPUTS, SAMPLER_WINDOW, TEST_DATA
from magi_finetuning.utils.runtime import summarize
from windinet.losses import SSIMLoss
from windinet.training.shockwave_data import CHANNEL_NAMES, ShockWaveDataset, parse_gamma

COMPARISONS = ("VAE round trip vs dataset", "DiT rollout vs dataset", "DiT rollout vs VAE round trip")
SEED = 42  # sim i is sampled with seed 42 + i (Weihao's convention)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a MAGI-1 VAE + DiT pair on test.h5.")
    p.add_argument("--dit-ckpt", default=None, help="LoRA or full-FT checkpoint (type detected automatically)")
    p.add_argument("--vae-only", action="store_true", help="no DiT: score and show the VAE round trip only")
    p.add_argument("--vae-ckpt", default=None, help="default: whatever the DiT run used (its args.yaml)")
    p.add_argument("--frames", type=int, default=None, help="frames 1..F to generate; default: the DiT run's --frames")
    p.add_argument("--test-data", default=TEST_DATA, help="test.h5: inference only, never used for training")
    p.add_argument("--metric-sims", type=int, default=500, help="test sims for the mean ± std table (spread over gammas)")
    p.add_argument("--video-sims", nargs="+", default=None, metavar="SIM_ID",
                   help="only these test sims, each with videos + panels (e.g. 0456_gamma1.7599999905), "
                        "instead of the 3 gamma cases and --metric-sims")
    p.add_argument("--num-steps", type=int, default=64, help="Euler steps per chunk (MAGI-1 uses 64)")
    p.add_argument("--shift", type=float, default=3.0, help="time-schedule shift, same as training")
    p.add_argument("--window", type=int, default=SAMPLER_WINDOW,
                   help="chunks denoised at once (MAGI-1's pipeline: 4; 1 = one chunk at a time, as before 2026-10-28)")
    p.add_argument("--video-fps", type=int, default=8)
    p.add_argument("--video-stats", default=str(STATS_FILE),
                   help="training-set channel statistics fixing the video colour limits (utils/rollout_video_stats.py)")
    p.add_argument("--score-frames", type=int, nargs="+", default=list(range(10, 91, 10)),
                   help="trajectory frames of the benchmark table (Harish's cfd_bench, dev-hr); default 10, 20, ..., 90")
    p.add_argument("--out", default=None, help="default: <DiT run dir>/eval_<checkpoint name>")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--report-only", action="store_true", help="only merge scores/*.json into tables + plots")
    p.add_argument("--no-archive", action="store_true", help="don't copy models + results to FINAL_MODELS")
    p.add_argument("--save-rollouts", action="store_true",
                   help="also keep every DiT rollout (normalized frames 1..F, fp16) in <out>/rollouts/<sim>.pt "
                        "(~52 MB per sim), so new metrics or frame sets can be scored without generating again")
    p.add_argument("--archive-name", default=None, help="folder name under FINAL_MODELS (default: from --out)")
    args = p.parse_args()
    if args.vae_only == bool(args.dit_ckpt):
        p.error("give --dit-ckpt, or --vae-only (then --out, and --vae-ckpt unless the released VAE)")
    return args


def run(args):
    fill_defaults_from_training_run(args)
    output_dir = Path(args.out)
    (output_dir / "scores").mkdir(parents=True, exist_ok=True)
    (output_dir / "videos").mkdir(parents=True, exist_ok=True)
    if not args.report_only:
        load_stats(args.video_stats)  # fail now, not after the first rollout, if the video statistics are missing
    if args.save_rollouts:
        (output_dir / "rollouts").mkdir(parents=True, exist_ok=True)

    # 1. Which test simulations: 3 showcase cases + an even spread over gammas for the table
    test_set = ShockWaveDataset(args.test_data, num_sim_frames=1 + args.frames)  # frame 0 + frames 1..F
    if args.video_sims:
        unknown = [sim_id for sim_id in args.video_sims if sim_id not in test_set.ids]
        if unknown:
            raise ValueError(f"not in {args.test_data}: {unknown}")
        cases = {sim_id: sim_id for sim_id in args.video_sims}  # each its own case, named after the sim
        metric_ids = sorted(args.video_sims)
    else:
        cases = pick_cases_by_gamma(test_set.ids)
        # Sorted by id (fully deterministic): every shard process must build the SAME list, or the shards
        # overlap and miss sims. (Sorting by gamma alone left ties in set order, which differs per process.)
        metric_ids = sorted(set(stratified_subset(test_set.ids, args.metric_sims)) | set(cases.values()))

    if not args.report_only:
        (output_dir / "eval_args.yaml").write_text(yaml.safe_dump(vars(args)))
        score_shard(args, test_set, cases, metric_ids, output_dir)
    if args.report_only or args.num_shards == 1:
        complete = write_report(cases, metric_ids, output_dir, args)
        if complete and not args.no_archive:
            archive_run(args, output_dir)


# ---------------------------------------------------------------- scoring (one shard, one GPU)

def score_shard(args, test_set, cases, metric_ids, output_dir: Path) -> None:
    """Rolls out this shard's sims (every num_shards-th), writes scores/<sim>.json, figures for the cases."""
    from magi_finetuning.dit import load_magi_config, load_trained_dit, rollout  # GPU imports only where needed
    from magi_finetuning.utils.runtime import init_magi_parallel_state
    from magi_finetuning.vae import load_vae

    torch.cuda.set_device(0)  # each shard sees exactly one GPU (CUDA_VISIBLE_DEVICES)
    init_magi_parallel_state()
    vae = load_vae(args.vae_ckpt)
    normalization = Normalization.of_vae_checkpoint(args.vae_ckpt)  # the one the VAE was trained with
    print(f"normalization: {normalization.describe()}", flush=True)
    if args.vae_only:
        model, is_full_finetune = None, False
        magi_config = load_magi_config()  # only for the latent scale factor
    else:
        model, magi_config, is_full_finetune = load_trained_dit(args.dit_ckpt)
    case_names = {sim_id: name for name, sim_id in cases.items()}
    ssim_fn = SSIMLoss(channels=len(CHANNEL_NAMES)).cuda()  # for the benchmark (1 - SSIM)

    my_ids = metric_ids[args.shard::args.num_shards]
    for count, sim_id in enumerate(my_ids, 1):
        score_file = output_dir / "scores" / f"{sim_id}.json"
        if score_file.is_file():
            continue  # already scored (resumed evaluation)
        started = time.time()
        torch.manual_seed(SEED + test_set.ids.index(sim_id))
        truth_physical = load_physical_fields(test_set, sim_id)  # raw dataset values, frames 0..F
        truth = normalization.apply(truth_physical)
        vae_round_trip, generated, inference_seconds = predict(vae, model, magi_config, truth, parse_gamma(sim_id),
                                                               args, is_full_finetune, rollout)  # frames 1..F
        if args.save_rollouts and generated is not None:
            torch.save(generated.half().cpu(), output_dir / "rollouts" / f"{sim_id}.pt")  # normalized frames 1..F
        to_physical = normalization.invert
        scores = {
            "sim_id": sim_id, "gamma": parse_gamma(sim_id),
            "normalized": score(truth[:, :, 1:], vae_round_trip, generated),
            "physical": score(truth_physical[:, :, 1:], to_physical(vae_round_trip),
                              None if generated is None else to_physical(generated)),
            "benchmark": benchmark(truth, vae_round_trip, generated, args.score_frames, ssim_fn),
        }
        if sim_id in case_names:
            name = case_names[sim_id]
            scores["case"] = name
            scores["vae_error_per_frame"] = vrmse_per_frame(vae_round_trip, truth[:, :, 1:]).tolist()  # frames 1..F
            make_case_figures(to_physical(vae_round_trip), truth_physical, f"{name}_vae", "VAE round trip",
                              sim_id, output_dir, args)
            if generated is not None:
                scores["error_per_frame"] = vrmse_per_frame(generated, truth[:, :, 1:]).tolist()
                make_case_figures(to_physical(generated), truth_physical, name, model_name(args, output_dir),
                                  sim_id, output_dir, args)
        scores["seconds"] = round(time.time() - started, 1)  # rollout + scoring (+ figures for the cases)
        if inference_seconds is not None:  # frame 0 -> frames 1..F only (see predict); for the timing table
            scores["inference_seconds"] = round(inference_seconds, 3)
            scores["gpu"] = torch.cuda.get_device_name(0)
            scores["resolution"] = list(truth.shape[-2:])
        score_file.write_text(json.dumps(scores))
        headline = COMPARISONS[0] if generated is None else COMPARISONS[1]
        print(f"[shard {args.shard}: {count}/{len(my_ids)}] {sim_id}  {headline} VRMSE "
              f"{np.mean(scores['normalized'][headline]):.4f}", flush=True)


def predict(vae, model, magi_config, truth, gamma, args, is_full_finetune, rollout):
    """truth: normalized frames 0..F. Returns (VAE round trip, DiT rollout, inference seconds): both fields
    normalized frames 1..F; the rollout and its time are None without a DiT (--vae-only).

    Same latent layout as training (data.build_latent_sequence). The rollout only sees the image latent of
    frame 0; the round trip encodes and decodes the true frames 1..F. The inference time covers exactly what a
    user of the model runs, frame 0 -> frames 1..F: encoding frame 0, the DiT rollout and decoding it (GPU
    synchronized at both ends; the round trip, scoring and figures are not included)."""
    scale = magi_config.runtime_config.scale_factor
    latents, num_body = build_latent_sequence(vae.encoder, truth, scale)
    num_frames = truth.shape[2] - 1
    round_trip = decode_body(vae, latents / scale, num_body, num_frames)
    if model is None:
        return round_trip, None, None
    num_chunks = latents.shape[2] // CHUNK_WIDTH
    tokens_per_chunk = CHUNK_WIDTH * (latents.shape[3] // 2) * (latents.shape[4] // 2)
    torch.cuda.synchronize()
    started = time.perf_counter()
    image_latent = encode_image(vae.encoder, truth[:, :, :1]) * scale  # == latents[:, :, :1]
    generated = rollout(
        model, image_latent, torch.tensor([[gamma]], device="cuda"), num_chunks, tokens_per_chunk,
        num_steps=args.num_steps, shift=args.shift, clean_t=magi_config.runtime_config.clean_t,
        window=args.window, use_autocast=is_full_finetune,
    )
    fields = decode_body(vae, generated / scale, num_body, num_frames)
    torch.cuda.synchronize()
    return round_trip, fields, time.perf_counter() - started


def decode_body(vae, latents, num_body: int, num_frames: int) -> torch.Tensor:
    """Body latents (after the image latent) -> frames 1..F. Decodes whole 4-latent tiles (incl. the
    padding the encoder saw) so every tile matches how it was encoded, then drops the padding frames."""
    num_tiles = -(-num_body // 4)
    return decode(vae, latents[:, :, 1:1 + 4 * num_tiles])[:, :, :num_frames]


@torch.no_grad()
def decode(vae, latents) -> torch.Tensor:
    """Unscaled latents -> normalized fields, in tiles of 4 latent frames (= the 16-frame clips it encoded)."""
    tiles = [vae.decoder(latents[:, :, i:i + 4].bfloat16()) for i in range(0, latents.shape[2], 4)]
    return torch.cat(tiles, dim=2).float()


def score(truth, vae_round_trip, rollout_fields) -> dict:
    """Per-channel VRMSE for the three comparisons (only the first without a rollout), all on frames 1..F."""
    scores = {COMPARISONS[0]: vrmse(vae_round_trip, truth)}
    if rollout_fields is not None:
        scores[COMPARISONS[1]] = vrmse(rollout_fields, truth)
        scores[COMPARISONS[2]] = vrmse(rollout_fields, vae_round_trip)
    return scores


@torch.no_grad()
def benchmark(truth, vae_round_trip, rollout_fields, score_frames, ssim_fn) -> dict:
    """Harish's benchmark metrics on the scored trajectory frames (frame k = k steps after the given frame 0),
    in normalized space, prediction and target clamped to [-1, 1] (cfd_bench's clamp_prediction).
    vae_round_trip / rollout_fields hold frames 1..F, so frame k is at index k - 1."""
    frames = torch.tensor(score_frames, device=truth.device)
    target = truth.index_select(2, frames).clamp(-1, 1)
    scored = lambda frames_1_to_f: frames_1_to_f.index_select(2, frames - 1).clamp(-1, 1)
    result = {COMPARISONS[0]: benchmark_metrics(scored(vae_round_trip), target, ssim_fn)}
    if rollout_fields is not None:
        result[COMPARISONS[1]] = benchmark_metrics(scored(rollout_fields), target, ssim_fn)
    return result


def make_case_figures(rollout_physical, truth_physical, case_name, model, sim_id, output_dir, args) -> None:
    """rollout (or VAE round trip): frames 1..F; truth: frames 0..F (physical units).
    Panels at frames 1, 25, 50, 75 and F; per channel a standardized video of frame 0 (given, same in both)
    followed by frames 1..F."""
    num_frames = rollout_physical.shape[2]
    panel_frames = sorted({1, *range(25, num_frames, 25), num_frames})
    save_case_panels(rollout_physical, truth_physical[:, :, 1:], case_name, panel_frames, output_dir)
    prediction_with_given = torch.cat([truth_physical[:, :, :1], rollout_physical], dim=2)
    save_rollout_videos(prediction_with_given, truth_physical, CHANNEL_NAMES, load_stats(args.video_stats),
                        output_dir / "videos", trajectory=sim_id, model=model, name=case_name, fps=args.video_fps,
                        dataset=args.test_data)


def model_name(args, output_dir: Path) -> str:
    """Model id in the videos: the archive name (e.g. dit_full_8000steps_on_F3) or the evaluation folder's."""
    return "MAGI-1 " + (args.archive_name or archive_name_from(output_dir))


# ---------------------------------------------------------------- report (after all shards)

def write_report(cases, metric_ids, output_dir: Path, args) -> bool:
    """Merges scores/*.json into results.md / CSVs and error_over_time.png, plus the runtime table.
    Returns True if every requested sim was scored (the report is final)."""
    scores = [json.loads(p.read_text()) for p in sorted((output_dir / "scores").glob("*.json"))]
    if not scores:
        print(f"No sims scored in {output_dir / 'scores'} -- check the shard logs; no report written.", flush=True)
        return False
    missing = set(metric_ids) - {s["sim_id"] for s in scores}
    if missing:
        print(f"WARNING: {len(missing)} of {len(metric_ids)} sims not scored yet; report covers the rest", flush=True)
    comparisons = [c for c in COMPARISONS if c in scores[0]["normalized"]]  # only the VAE one with --vae-only

    case_rows, summary_rows, curves = [], [], {}
    for space in ("normalized", "physical"):
        for comparison in comparisons:
            summary_rows.append(summary_row(f"{comparison} [{space}]", [s[space][comparison] for s in scores]))
    curve_key = "error_per_frame" if COMPARISONS[1] in comparisons else "vae_error_per_frame"
    for s in sorted((s for s in scores if "case" in s), key=lambda s: s["gamma"]):
        for comparison in comparisons:
            case_rows.append(table_row(s["normalized"][comparison], case=s["case"], gamma=s["gamma"],
                                       comparison=comparison))
        if curve_key in s:
            curves[f"{s['case']} (gamma {s['gamma']:.3f})"] = np.array(s[curve_key])

    if curves:
        title = ("DiT rollout vs dataset" if curve_key == "error_per_frame" else "VAE round trip vs dataset")
        save_error_over_time(curves, chunk_boundaries(args.frames), output_dir / "error_over_time.png", title)
    sampler = f"{args.num_steps} Euler steps per chunk, shift {args.shift}, {args.window} chunk(s) at once"
    results = write_tables(case_rows, summary_rows, output_dir, runtime_rows(args, scores),
                           benchmark_summary(scores), args.score_frames, sampler,
                           inference_summary(scores, args.frames, sampler))
    print(results.read_text(), flush=True)
    print(f"everything written to {output_dir}", flush=True)
    return not missing


def benchmark_summary(scores) -> list[dict]:
    """Mean ± std over sims of each benchmark metric, per comparison (sims scored before the benchmark existed
    have no "benchmark" entry and are left out)."""
    with_benchmark = [s["benchmark"] for s in scores if "benchmark" in s]
    rows = []
    for comparison in COMPARISONS[:2]:
        values = [b[comparison] for b in with_benchmark if comparison in b]
        if not values:
            continue
        row = {"comparison": comparison, "num_sims": len(values)}
        for metric in BENCHMARK_METRICS:
            column = torch.tensor([v[metric] for v in values], dtype=torch.float64)
            row[f"{metric}_mean"], row[f"{metric}_std"] = column.mean().item(), column.std(unbiased=False).item()
        rows.append(row)
    return rows


def runtime_rows(args, scores) -> list[dict]:
    """Training time of the VAE and DiT runs (their timing.json next to the checkpoints) and this evaluation."""
    rows = []
    for stage, ckpt in (("VAE training", args.vae_ckpt), ("DiT training", args.dit_ckpt)):
        if stage == "DiT training" and args.vae_only:
            continue
        summary = summarize(Path(ckpt).parent / "timing.json") if ckpt else None
        rows.append({"stage": stage, **(summary or {})})
    timed = [s["seconds"] for s in scores if "seconds" in s]
    if timed:
        rows.append({"stage": "evaluation (rollout + scoring)", "sims": len(timed),
                     "sec_per_sim": sum(timed) / len(timed), "gpu_hours": sum(timed) / 3600})
    return rows


def table_row(channel_scores, **labels) -> dict:
    return {**labels, **dict(zip(CHANNEL_NAMES, channel_scores)), "mean": sum(channel_scores) / len(channel_scores)}


def summary_row(comparison, per_sim_scores) -> dict:
    """mean and std over simulations, per field and for the field average."""
    values = torch.tensor(per_sim_scores)  # (sims, 4)
    columns = {name: values[:, c] for c, name in enumerate(CHANNEL_NAMES)} | {"mean": values.mean(dim=1)}
    row = {"comparison": comparison, "num_sims": len(per_sim_scores)}
    for name, column in columns.items():
        row[f"{name}_mean"] = column.mean().item()
        row[f"{name}_std"] = column.std(unbiased=False).item()
    return row


# ---------------------------------------------------------------- setup

def fill_defaults_from_training_run(args) -> None:
    """--vae-ckpt / --frames / --out default to the DiT run that produced --dit-ckpt."""
    if args.vae_only:  # without --vae-ckpt: the released VAE, pressure channel = mean of the RGB filters
        if not args.out:
            raise ValueError("--vae-only needs --out")
        args.frames = args.frames or DEFAULT_FRAMES
        return
    ckpt = Path(args.dit_ckpt)
    run_dir = ckpt.parent.parent if ckpt.parent.name == "checkpoints" else ckpt.parent
    train_args = {}
    if (run_dir / "args.yaml").is_file():
        train_args = yaml.safe_load((run_dir / "args.yaml").read_text()) or {}
    if args.vae_ckpt is None:
        args.vae_ckpt = train_args.get("vae_ckpt")
    if args.frames is None:
        args.frames = train_args.get("frames", DEFAULT_FRAMES)
    if max(args.score_frames) > args.frames or min(args.score_frames) < 1:
        raise ValueError(f"--score-frames must lie in 1..{args.frames} (frame 0 is given, not generated)")
    if args.frames % 4 or not 4 <= args.frames <= 100:
        raise ValueError("--frames must be a multiple of 4 between 4 and 100")
    if args.out is None:
        args.out = str(run_dir / f"eval_{ckpt.stem}")


def archive_run(args, output_dir: Path) -> None:
    """Copies the evaluated models and their results to FINAL_MODELS/<name> (SCRATCH):
        vae/   the VAE checkpoint + its run records (timing, best, finished, args)
        dit/   the DiT checkpoint + its run records (not with --vae-only)
        eval/  everything in the evaluation folder (results.md, CSVs, plots, videos, scores)
    This is also the meta folder evaluate_experiments.sh works on. A failed copy (e.g. quota) only warns:
    the results are already written."""
    name = args.archive_name or archive_name_from(output_dir)
    destination = FINAL_MODELS / name
    try:
        for kind, ckpt in (("vae", args.vae_ckpt), ("dit", args.dit_ckpt)):
            if ckpt:
                copy_checkpoint_with_records(Path(ckpt), destination / kind)
        shutil.copytree(output_dir, destination / "eval", dirs_exist_ok=True)
        print(f"final models and results archived to {destination}", flush=True)
    except OSError as error:
        print(f"WARNING: archiving to {destination} failed ({error}); results stay in {output_dir}", flush=True)


def archive_name_from(output_dir: Path) -> str:
    """<OUTPUTS>/official/lora/eval -> official_lora (outside OUTPUTS: the folder's name)."""
    resolved = output_dir.resolve()
    parts = resolved.relative_to(OUTPUTS).parts if resolved.is_relative_to(OUTPUTS) else (resolved.name,)
    return "_".join(p for p in parts if p != "eval") or resolved.name


def copy_checkpoint_with_records(ckpt: Path, destination: Path) -> None:
    """The checkpoint and the small run records next to it; skipped if it is already the archived copy."""
    destination.mkdir(parents=True, exist_ok=True)
    for source in [ckpt] + [ckpt.parent / f for f in ("timing.json", "best.json", "finished.json", "args.yaml")]:
        target = destination / source.name
        if source.is_file() and not (target.exists() and target.samefile(source)):
            shutil.copy2(source, target)


def chunk_boundaries(num_frames: int) -> list[int]:
    """Last generated frame of each chunk except the final one. Chunk 0 = [image, latents 1-5] -> frames
    1-20; every later chunk adds 6 latents = 24 frames: boundaries after frames 20, 44, 68, 92."""
    return [f for f in range(4 * (CHUNK_WIDTH - 1), num_frames, 4 * CHUNK_WIDTH)]


def pick_cases_by_gamma(sim_ids) -> dict:
    """The two gamma tails and the gamma closest to the mean; first simulation of each gamma."""
    gammas = sorted({parse_gamma(i) for i in sim_ids})
    mean_gamma = sum(parse_gamma(i) for i in sim_ids) / len(sim_ids)
    chosen = {"low_gamma": gammas[0], "mid_gamma": min(gammas, key=lambda g: abs(g - mean_gamma)),
              "high_gamma": gammas[-1]}
    return {name: sorted(i for i in sim_ids if parse_gamma(i) == g)[0] for name, g in chosen.items()}


def load_physical_fields(test_set, sim_id) -> torch.Tensor:
    """(1, 4, 1 + F, H, W) raw physical fields (frames 0..F) of one test simulation, on the GPU."""
    sample = test_set[test_set.ids.index(sim_id)]
    return torch.stack([sample[name] for name in CHANNEL_NAMES], dim=0)[None].cuda().float()


if __name__ == "__main__":
    run(parse_args())
