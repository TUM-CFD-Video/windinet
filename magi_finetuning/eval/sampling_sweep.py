"""Sampling-side checks on one DiT checkpoint, no training: symmetry and seed ensembles, fewer steps, and the
rollout on train / val sims against test sims (2026-11-13).

Every rollout is the 4-step sampler of the checkpoint curves (dit.rollout: 4 Euler steps per chunk, shift 5, 4
chunks at once, seed 42 + sim index, exactly as evaluate.py) unless a variant says otherwise.

1. Train / val / test. --split-sims N sims of the train split and N of the val split (never trained on), evenly
   over the gammas, are rolled out and scored like the test sims. Scoring a second, earlier checkpoint on the same
   sims (--tag old) lets the report compare the CHANGE per sim: a model that memorizes its training sims improves
   more on them than on val and test sims. The change per sim is far less noisy than a difference of group means
   (the sims differ in difficulty: std 0.12 over sims, about 0.03 for a change).

2. On the test sims (--test-sims-from: the sims of an evaluate.py run), --test-variants:
   d4      the 8 symmetries of the square. The Euler equations and the open boundaries are unchanged by flips and
           by swapping x and y, so the rollout from a transformed frame 0, transformed back, is another prediction
           of the same simulation. Scored: every orientation alone, the mean of the 4 flips, the mean of all 8.
   seeds   3 more seeds; the mean of 2 and of 4 seeds (4 = as many samples as the 4 flips).
   steps   fewer denoising steps, (steps per chunk, chunks at once): (1, 1), (2, 1), (2, 2), (4, 1), (4, 2).

Axes: the fields are stored (..., x, y). momentum_x points along the second-to-last array axis, momentum_y along
the last, both positive toward increasing index (least squares of the mass equation on test.h5: R^2 0.37 to 0.51
for this pairing against 0.01 to 0.06 for the other; both coefficients negative). check_transforms repeats that fit
on every transformed truth before any model is loaded and stops the run if a transform breaks it.

Scores per sim and variant: evaluate.py's benchmark metrics on frames 10, 20, ..., 90, the all-frames VRMSE, and
the sharpness measures of sharpness.py on frames 1..100 (99th-percentile gradient and power per wavenumber band,
as ratios to the truth's: an averaged prediction is blurrier).

--video-sims (with d4) writes evaluate.py's panels and videos of those test sims twice: for one sample and for
the mean of all 8 symmetries. A run that was cut off continues where it stopped (the score file of the shard).

    python -m magi_finetuning.eval.sampling_sweep --dit-ckpt <ckpt> --out <dir> --tag new [--split-sims 60]
        [--roundtrip] [--test-sims-from <evaluate.py run>] [--test-variants d4,seeds,steps] [--shard i --num-shards n]
        [--video-sims low_gamma=0000_gamma1.1299999952 0456_gamma1.7599999905]
    then once: ... --out <dir> --report-only [--test-sims-from <run>] [--old-test-scores <evaluate.py run, old ckpt>]
"""

import argparse
import json
import time
from pathlib import Path

import torch

from magi_finetuning.settings import OUTPUTS, SAMPLER_WINDOW, SCRATCH, TEST_DATA, TRAIN_DATA
from windinet.training.shockwave_data import CHANNEL_NAMES

MX, MY = CHANNEL_NAMES.index("momentum_x"), CHANNEL_NAMES.index("momentum_y")
X_AXIS, Y_AXIS = -2, -1  # the fields are stored (..., x, y)
# (swap x and y, flip x, flip y): the 8 symmetries of the square; the first four are the flips, the first the identity
SYMMETRIES = [(swap, flip_x, flip_y) for swap in (False, True) for flip_x in (False, True) for flip_y in (False, True)]
EXTRA_SEEDS = (1042, 2042, 3042)  # as the seed-spread runs of 2026-11-11
STEP_VARIANTS = ((1, 1), (2, 1), (2, 2), (4, 1), (4, 2))  # (Euler steps per chunk, chunks at once)
SCORE_FRAMES = list(range(10, 91, 10))  # evaluate.py's default
BASE = "base"
ROLLOUT = "DiT rollout vs dataset"  # evaluate.py's name of the rollout scores


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dit-ckpt", default=str(SCRATCH / "magi1_dit_full_14000_plus10000_lr5e-6_fast_on_F3" /
                                             "full_step010000.safetensors"))
    p.add_argument("--vae-ckpt", default=None, help="default: VAE F3")
    p.add_argument("--tag", default="new", help="name of this checkpoint in the score files (new / old)")
    p.add_argument("--split-sims", type=int, default=0, help="this many train-split and val-split sims (0: none)")
    p.add_argument("--roundtrip", action="store_true", help="also score the VAE round trip of every sim")
    p.add_argument("--test-sims-from", default=None, help="evaluate.py run whose test sims are used")
    p.add_argument("--test-variants", default="", help="comma-separated: d4, seeds, steps")
    p.add_argument("--test-data", default=TEST_DATA)
    p.add_argument("--num-steps", type=int, default=4)
    p.add_argument("--shift", type=float, default=5.0)
    p.add_argument("--window", type=int, default=SAMPLER_WINDOW)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--save-sims", nargs="*", default=[],
                   help="test sims whose one-sample and symmetry-mean fields are saved to <out>/fields (with d4)")
    p.add_argument("--video-sims", nargs="*", default=[], metavar="[NAME=]SIM_ID",
                   help="test sims with panels and videos of one sample and of the mean of 8 symmetries (with d4); "
                        "they are done first")
    p.add_argument("--model-label", default="MAGI-1 +10,000 steps", help="model id shown in the videos")
    p.add_argument("--video-slowdown", type=float, default=8.0, help="video seconds per physical second")
    p.add_argument("--video-stats", default=str(OUTPUTS / "rollout_video_stats.json"),
                   help="training-set channel statistics fixing the video colour limits")
    p.add_argument("--max-minutes", type=float, default=None, help="start no new sim after this many minutes")
    p.add_argument("--report-only", action="store_true", help="only merge <out>/scores_*.json into summary.md")
    p.add_argument("--old-test-scores", default=None,
                   help="report: evaluate.py run of the old checkpoint on the test sims, same sampler")
    p.add_argument("--out", required=True)
    return p.parse_args()


# ---------------------------------------------------------------- symmetries

def transform(fields, symmetry, inverse: bool = False):
    """Physical fields (1, 4, T, X, Y) under one symmetry: x and y swapped first, then the flips (inverse: the
    flips first). A flip along an axis negates the momentum component along it; the swap exchanges the two."""
    swap, flip_x, flip_y = symmetry

    def flips(f):
        dims = [axis for axis, on in ((X_AXIS, flip_x), (Y_AXIS, flip_y)) if on]
        f = f.flip(dims) if dims else f.clone()
        if flip_x:
            f[:, MX] = -f[:, MX]
        if flip_y:
            f[:, MY] = -f[:, MY]
        return f

    def swapped(f):
        if not swap:
            return f
        f = f.transpose(X_AXIS, Y_AXIS).clone()
        f[:, [MX, MY]] = f[:, [MY, MX]]
        return f

    return swapped(flips(fields)) if inverse else flips(swapped(fields))


def mass_equation_fit(fields, frames=(20, 40, 60, 80)):
    """Least squares of d(density)/dt on d(momentum_x)/dx and d(momentum_y)/dy (central differences, interior
    points) on some frames of physical fields (1, 4, T, X, Y): the two coefficients and R^2. Mass conservation
    makes both coefficients negative; with the momenta paired with the wrong axes R^2 falls below 0.1."""
    f = fields[0].double()
    rows, targets = [], []
    for k in frames:
        targets.append((f[0, k + 1] - f[0, k - 1])[1:-1, 1:-1].flatten())
        rows.append(torch.stack([((f[MX, k, 2:, 1:-1] - f[MX, k, :-2, 1:-1]) / 2).flatten(),
                                 ((f[MY, k, 1:-1, 2:] - f[MY, k, 1:-1, :-2]) / 2).flatten()], dim=1))
    a, y = torch.cat(rows), torch.cat(targets)
    coef = torch.linalg.solve(a.T @ a, a.T @ y)
    r2 = 1 - (y - a @ coef).pow(2).sum() / (y - y.mean()).pow(2).sum()
    return coef.tolist(), r2.item()


def check_transforms(fields) -> None:
    """Raises if a symmetry is implemented wrongly. The data must satisfy the mass equation with the axes as
    assumed here, every transformed truth must satisfy it exactly as well (the same fit, the two coefficients
    possibly exchanged), and transforming back must return the data."""
    reference, r2 = mass_equation_fit(fields)
    print(f"mass equation on the data: coefficients {reference[0]:.3f}, {reference[1]:.3f}, R^2 {r2:.3f}", flush=True)
    if max(reference) >= 0 or r2 < 0.1:
        raise ValueError("the data do not fit the mass equation with momentum_x along the second-to-last axis")
    for symmetry in SYMMETRIES:
        moved = transform(fields, symmetry)
        coef, moved_r2 = mass_equation_fit(moved)
        back = (transform(moved, symmetry, inverse=True) - fields).abs().max().item()
        same_fit = abs(moved_r2 - r2) < 1e-6 and all(abs(a - b) < 1e-6 * abs(b) + 1e-9
                                                     for a, b in zip(sorted(coef), sorted(reference)))
        print(f"  symmetry {symmetry}: coefficients {coef[0]:.3f}, {coef[1]:.3f}, R^2 {moved_r2:.3f}, "
              f"back and forth error {back:.1e}", flush=True)
        if back != 0 or not same_fit:
            raise ValueError(f"symmetry {symmetry} does not keep the mass equation or is not inverted exactly")


def normalize(normalization, x):
    """Normalization.apply without the clamp to [-1, 1]: predictions are scored as the decoder returns them."""
    if normalization.log_fields:
        raise ValueError("the symmetry ensemble is written for a normalization without log fields")
    mean, std = normalization.mean_std
    shape = (1, len(CHANNEL_NAMES)) + (1,) * (x.ndim - 2)
    return (x - x.new_tensor(mean).view(shape)) / (x.new_tensor(std).view(shape) * normalization.clip)


# ---------------------------------------------------------------- scoring (one shard, one GPU)

def sims_of(args) -> tuple[list, list]:
    """([(group, sim id)] of the train and val splits, [("test", sim id)])."""
    from magi_finetuning.data import load_split, stratified_subset

    split_sims = []
    if args.split_sims:
        split = load_split()
        for group in ("train", "val"):
            split_sims += [(group, sim_id) for sim_id in stratified_subset(split[group], args.split_sims)]
    test_sims = []
    if args.test_sims_from:
        test_sims = [("test", path.stem) for path in sorted((Path(args.test_sims_from) / "scores").glob("*.json"))]
    return split_sims, test_sims


def measures(prediction, truth, ssim_fn, truth_stats) -> dict:
    """prediction: normalized frames 1..F; truth: normalized frames 0..F. The benchmark metrics on the scored
    frames (as evaluate.benchmark), the all-frames VRMSE (as evaluate.score), sharpness on frames 1..F."""
    from magi_finetuning.eval.report import benchmark_metrics, vrmse
    from magi_finetuning.eval.sharpness import describe

    frames = torch.tensor(SCORE_FRAMES, device=truth.device)
    result = benchmark_metrics(prediction.index_select(2, frames - 1).clamp(-1, 1),
                               truth.index_select(2, frames).clamp(-1, 1), ssim_fn)
    result["all_frames"] = sum(vrmse(prediction, truth[:, :, 1:])) / len(CHANNEL_NAMES)
    sharpness = describe(prediction, truth[:, :, 1:], truth_stats)
    result["grad_p99"] = sharpness["grad_p99_ratio"]
    result["power"] = sharpness["power_ratio"]
    return result


@torch.no_grad()
def score_shard(args, out: Path) -> None:
    from magi_finetuning.data import Normalization, build_latent_sequence, encode_image
    from magi_finetuning.dit import load_trained_dit, rollout
    from magi_finetuning.eval.evaluate import (SEED, decode_body, load_physical_fields, load_video_stats,
                                               make_case_figures)
    from magi_finetuning.eval.latent_sensitivity import VAES
    from magi_finetuning.eval.sharpness import BAND_NAMES, band_power, gradient_stats, relative_rms
    from magi_finetuning.settings import CHUNK_WIDTH
    from magi_finetuning.utils.runtime import init_magi_parallel_state
    from magi_finetuning.vae import load_vae
    from windinet.losses import SSIMLoss
    from windinet.training.shockwave_data import ShockWaveDataset, parse_gamma

    started = time.time()
    split_sims, test_sims = sims_of(args)
    todo = test_sims[args.shard::args.num_shards] + split_sims[args.shard::args.num_shards]  # the sweep first
    videos = {entry.split("=")[-1]: entry.split("=")[0] for entry in args.video_sims}  # sim id -> file name
    unknown = set(videos) - {sim_id for _, sim_id in test_sims}
    if unknown:
        raise ValueError(f"--video-sims not among the test sims: {sorted(unknown)}")
    todo.sort(key=lambda task: task[1] not in videos)  # the sims with videos first (stable: the rest keeps its order)
    if videos:
        load_video_stats(args.video_stats)  # fail now if they are missing or invalid
        (out / "videos").mkdir(exist_ok=True)
    test_variants = {v for v in args.test_variants.split(",") if v}
    unknown = test_variants - {"d4", "seeds", "steps"}
    if unknown:
        raise ValueError(f"unknown --test-variants {sorted(unknown)}")
    datasets = {}
    if test_sims:
        datasets["test"] = ShockWaveDataset(args.test_data, num_sim_frames=101)
    if split_sims:
        datasets["train"] = datasets["val"] = ShockWaveDataset(TRAIN_DATA, num_sim_frames=101)
    if "d4" in test_variants:  # before anything expensive is loaded
        sample = datasets["test"][datasets["test"].ids.index(todo[0][1])]
        check_transforms(torch.stack([sample[name] for name in CHANNEL_NAMES], dim=0)[None].float())

    vae_ckpt = args.vae_ckpt or str(VAES["F3"])
    torch.cuda.set_device(0)
    init_magi_parallel_state()
    vae = load_vae(vae_ckpt)
    normalization = Normalization.of_vae_checkpoint(vae_ckpt)
    model, magi_config, is_full_finetune = load_trained_dit(args.dit_ckpt)
    scale, clean_t = magi_config.runtime_config.scale_factor, magi_config.runtime_config.clean_t
    ssim_fn = SSIMLoss(channels=len(CHANNEL_NAMES)).cuda()
    base_sampler = dict(num_steps=args.num_steps, shift=args.shift, clean_t=clean_t, window=args.window,
                        use_autocast=is_full_finetune)
    print(f"loaded after {time.time() - started:.0f} s; {len(todo)} sims", flush=True)

    rows = []
    score_file = out / f"scores_{args.tag}_shard{args.shard}.json"
    settings = {"dit_ckpt": args.dit_ckpt, "vae_ckpt": vae_ckpt, "tag": args.tag, "num_steps": args.num_steps,
                "shift": args.shift, "window": args.window, "test_variants": sorted(test_variants),
                "bands": BAND_NAMES}
    if score_file.is_file():  # a run that was cut off: keep what it scored
        earlier = json.loads(score_file.read_text())
        if earlier["settings"] != settings:
            raise ValueError(f"{score_file} holds scores of other settings: {earlier['settings']}")
        rows = earlier["rows"]
        print(f"continuing: {len(rows)} sims already scored", flush=True)
    done = {(row["group"], row["sim_id"]) for row in rows}
    for count, (group, sim_id) in enumerate(todo, 1):
        if (group, sim_id) in done:
            continue
        if args.max_minutes and time.time() - started > 60 * args.max_minutes:
            print(f"stopped after {count - 1} of {len(todo)} sims: --max-minutes {args.max_minutes}", flush=True)
            break
        sim_started = time.time()
        dataset = datasets[group]
        index = dataset.ids.index(sim_id)
        physical = load_physical_fields(dataset, sim_id)  # frames 0..F
        truth = normalization.apply(physical)
        latents, num_body = build_latent_sequence(vae.encoder, truth, scale)
        num_frames = truth.shape[2] - 1
        num_chunks = latents.shape[2] // CHUNK_WIDTH
        tokens_per_chunk = CHUNK_WIDTH * (latents.shape[3] // 2) * (latents.shape[4] // 2)
        gamma = torch.tensor([[parse_gamma(sim_id)]], device="cuda")
        truth_stats = (band_power(truth[:, :, 1:]), *gradient_stats(truth[:, :, 1:]))
        score = lambda fields: measures(fields, truth, ssim_fn, truth_stats)

        def sample(symmetry=SYMMETRIES[0], seed=SEED, sampler=base_sampler):
            """Normalized frames 1..F in the simulation's own orientation, generated from frame 0 under symmetry."""
            identity = symmetry == SYMMETRIES[0]
            frame0 = truth[:, :, :1] if identity else normalization.apply(transform(physical[:, :, :1], symmetry))
            image_latent = encode_image(vae.encoder, frame0) * scale
            torch.manual_seed(seed + index)
            generated = rollout(model, image_latent, gamma, num_chunks, tokens_per_chunk, **sampler)
            fields = decode_body(vae, generated / scale, num_body, num_frames)
            if not identity:
                fields = normalize(normalization, transform(normalization.invert(fields), symmetry, inverse=True))
            return fields

        scores, spread = {}, {}
        if args.roundtrip:
            scores["roundtrip"] = score(decode_body(vae, latents / scale, num_body, num_frames))
        base = sample()
        scores[BASE] = score(base)
        variants = test_variants if group == "test" else set()
        distance = lambda members: sum(relative_rms(m, base, truth[:, :, 1:]) for m in members[1:]) / (len(members) - 1)
        if "d4" in variants:
            members = [base] + [sample(symmetry=symmetry) for symmetry in SYMMETRIES[1:]]
            for number, member in enumerate(members[1:], 1):
                scores[f"orientation{number}"] = score(member)
            scores["mean_4_flips"] = score(torch.stack(members[:4]).mean(dim=0))
            scores["mean_8_symmetries"] = score(torch.stack(members).mean(dim=0))
            spread["flips_to_base"], spread["symmetries_to_base"] = distance(members[:4]), distance(members)
            if sim_id in videos:  # evaluate.py's panels and video, for one sample and for the mean
                sampler = f"{args.num_steps} steps"
                for name, fields in (("one_sample", base), ("mean_8_symmetries", torch.stack(members).mean(dim=0))):
                    make_case_figures(normalization.invert(fields), physical, f"{videos[sim_id]}_{name}",
                                      f"{args.model_label}, {name.replace('_', ' ')}, {sampler}", sim_id, out, args)
            if sim_id in args.save_sims:  # normalized frames 1..F, as evaluate.py --save-rollouts
                (out / "fields").mkdir(exist_ok=True)
                torch.save({"one_sample": base.half().cpu(), "mean_4_flips": torch.stack(members[:4]).mean(dim=0).half().cpu(),
                            "mean_8_symmetries": torch.stack(members).mean(dim=0).half().cpu()},
                           out / "fields" / f"{sim_id}.pt")
            del members
        if "seeds" in variants:
            members = [base] + [sample(seed=seed) for seed in EXTRA_SEEDS]
            for seed, member in zip(EXTRA_SEEDS, members[1:]):
                scores[f"seed{seed}"] = score(member)
            scores["mean_2_seeds"] = score(torch.stack(members[:2]).mean(dim=0))
            scores["mean_4_seeds"] = score(torch.stack(members).mean(dim=0))
            spread["seeds_to_base"] = distance(members)
            del members
        if "steps" in variants:
            for steps, window in STEP_VARIANTS:
                scores[f"steps{steps}_window{window}"] = score(sample(sampler={**base_sampler, "num_steps": steps,
                                                                               "window": window}))
        rows.append({"group": group, "sim_id": sim_id, "gamma": parse_gamma(sim_id), "scores": scores, "spread": spread})
        partial = score_file.with_suffix(".tmp")
        partial.write_text(json.dumps({"settings": settings, "rows": rows}))
        partial.replace(score_file)  # a complete file after every sim, also if the job is cut off
        print(f"[{args.tag} shard {args.shard}: {count}/{len(todo)}] {group} {sim_id}: VRMSE {scores[BASE]['vrmse']:.4f}"
              + (f", mean of 4 flips {scores['mean_4_flips']['vrmse']:.4f}" if "mean_4_flips" in scores else "")
              + f" ({time.time() - sim_started:.0f} s)", flush=True)
    print(f"done after {(time.time() - started) / 60:.1f} min", flush=True)


# ---------------------------------------------------------------- report

def mean_se(values) -> tuple[float, float]:
    n = len(values)
    mean = sum(values) / n
    return mean, ((sum((v - mean) ** 2 for v in values) / (n - 1)) ** 0.5 / n ** 0.5 if n > 1 else float("nan"))


def variant_order(names) -> list:
    first = ["roundtrip", BASE, "mean_2_seeds", "mean_4_seeds", "mean_4_flips", "mean_8_symmetries"]
    steps = [f"steps{s}_window{w}" for s, w in STEP_VARIANTS]
    singles = sorted(n for n in names if n.startswith(("orientation", "seed")))
    return [n for n in first + steps + singles if n in names]


LABELS = {"roundtrip": "VAE round trip (the floor)", BASE: "one sample: 4 steps, 4 chunks at once, seed 42",
          "mean_2_seeds": "mean of 2 seeds", "mean_4_seeds": "mean of 4 seeds",
          "mean_4_flips": "mean of the 4 flips", "mean_8_symmetries": "mean of all 8 symmetries"}


def label_of(name: str) -> str:
    if name in LABELS:
        return LABELS[name]
    if name.startswith("steps"):
        steps, window = name[5:].split("_window")
        return f"one sample: {steps} step(s) per chunk, {window} chunk(s) at once"
    if name.startswith("orientation"):
        swap, flip_x, flip_y = SYMMETRIES[int(name[11:])]
        parts = [text for text, on in (("x and y swapped", swap), ("x flipped", flip_x), ("y flipped", flip_y)) if on]
        return "one sample, frame 0 with " + ", ".join(parts)
    return f"one sample, seed {name[4:]}"


def report(args, out: Path) -> None:
    files = sorted(out.glob("scores_*_shard*.json"))
    if not files:
        raise FileNotFoundError(f"no scores_*_shard*.json in {out}")
    by_tag = {}
    for path in files:
        content = json.loads(path.read_text())
        by_tag.setdefault(content["settings"]["tag"], {"settings": content["settings"], "rows": []})["rows"] += content["rows"]
    new = by_tag.get("new") or next(iter(by_tag.values()))
    band = new["settings"]["bands"].index("k 16..31")
    lines = ["# Sampling sweep: symmetry and seed ensembles, fewer steps, train / val / test", "",
             f"DiT {new['settings']['dit_ckpt']}. Unless a row says otherwise: {new['settings']['num_steps']} Euler "
             f"steps per chunk, shift {new['settings']['shift']}, {new['settings']['window']} chunks at once, seed 42 + "
             "sim index. Definitions: magi_finetuning/eval/sampling_sweep.py. Per sim: scores_*.json.", ""]

    test = [row for row in new["rows"] if row["group"] == "test"]
    if test:
        names = variant_order(set.intersection(*(set(row["scores"]) for row in test)))
        lines += [f"## Test sims ({len(test)}): variants of the sampler", "",
                  "Benchmark metrics on frames 10, 20, ..., 90; \"against one sample\" is the paired difference of "
                  "the VRMSE per sim (mean ± standard error). Sharpness on frames 1..100, as ratios to the truth's.", "",
                  "| variant | VRMSE | against one sample | sims better | RMSE | 1-SSIM | MLW | all-frames VRMSE | "
                  "gradient p99 | power k 16..31 |", "|---|---|---|---|---|---|---|---|---|---|"]
        for name in names:
            get = lambda key: sum(row["scores"][name][key] for row in test) / len(test)
            diffs = [row["scores"][name]["vrmse"] - row["scores"][BASE]["vrmse"] for row in test]
            diff, se = mean_se(diffs)
            paired = "–" if name == BASE else f"{diff:+.4f} ± {se:.4f}"
            better = "–" if name == BASE else f"{sum(d < 0 for d in diffs)}/{len(diffs)}"
            power = sum(row["scores"][name]["power"][band] for row in test) / len(test)
            lines.append(f"| {label_of(name)} | {get('vrmse'):.4f} | {paired} | {better} | {get('rmse'):.4f} | "
                         f"{get('ssim'):.4f} | {get('mlw'):.2f} | {get('all_frames'):.4f} | {get('grad_p99'):.3f} | "
                         f"{power:.3f} |")
        lines.append("")
        spreads = sorted(set.intersection(*(set(row["spread"]) for row in test)))
        if spreads:
            lines += ["Distance of the other samples to the first, RMS difference / std of the truth (all frames): "
                      + "; ".join(f"{key.replace('_to_base', '').replace('_', ' ')} "
                                  f"{sum(row['spread'][key] for row in test) / len(test):.4f}" for key in spreads)
                      + ". Equal distances for flips and seeds mean the model treats the orientations alike.", ""]
        lines += ["In the format of the comparison of record (mean ± std over the sims, as evaluate.py's results.md):", "",
                  "| prediction | sims | RMSE | VRMSE | 1-SSIM | MLW |", "|---|---|---|---|---|---|"]
        for name in (n for n in ("roundtrip", BASE, "mean_4_seeds", "mean_4_flips", "mean_8_symmetries") if n in names):
            def spread(key, digits):
                values = sorted(row["scores"][name][key] for row in test)
                mean = sum(values) / len(values)
                std = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
                median = (values[(len(values) - 1) // 2] + values[len(values) // 2]) / 2
                return f"{mean:.{digits}f} ± {std:.{digits}f}" + (f" (median {median:.{digits}f})" if key == "vrmse" else "")
            lines.append(f"| {label_of(name)} | {len(test)} | {spread('rmse', 4)} | {spread('vrmse', 4)} | "
                         f"{spread('ssim', 4)} | {spread('mlw', 1)} |")
        lines.append("")
        if args.test_sims_from:
            import yaml

            run = Path(args.test_sims_from)
            reference = {p.stem: json.loads(p.read_text())["benchmark"][ROLLOUT]["vrmse"] for p in (run / "scores").glob("*.json")}
            both = [row for row in test if row["sim_id"] in reference]
            gaps = [row["scores"][BASE]["vrmse"] - reference[row["sim_id"]] for row in both]
            theirs = yaml.safe_load((run / "eval_args.yaml").read_text())
            ours = new["settings"]
            if all(theirs[key] == ours[key] for key in ("num_steps", "shift", "window")):
                lines += [f"Check: the one-sample VRMSE here against {run} on {len(gaps)} sims: largest difference "
                          f"{max(abs(g) for g in gaps):.1e}.", ""]
            else:
                diff, se = mean_se(gaps)
                lines += [f"One sample here against one sample of {run} ({theirs['num_steps']} steps, shift "
                          f"{theirs['shift']}, {theirs['window']} chunks at once) on {len(gaps)} sims: {diff:+.4f} ± {se:.4f} "
                          f"({sum(g < 0 for g in gaps)}/{len(gaps)} lower here); mean of all 8 symmetries against it: "
                          + ("{:+.4f} ± {:.4f}".format(*mean_se([row["scores"]["mean_8_symmetries"]["vrmse"] - reference[row["sim_id"]]
                                                                 for row in both])) if "mean_8_symmetries" in names else "–")
                          + ".", ""]

    groups = [g for g in ("train", "val", "test") if any(row["group"] == g for row in new["rows"])]
    if "train" in groups:
        old = {(row["group"], row["sim_id"]): row for row in by_tag.get("old", {"rows": []})["rows"]}
        if args.old_test_scores:  # the old checkpoint on the test sims, from an evaluate.py run
            for path in (Path(args.old_test_scores) / "scores").glob("*.json"):
                old.setdefault(("test", path.stem), {"scores": {BASE: json.loads(path.read_text())["benchmark"][ROLLOUT]}})
        lines += ["## Train, val and test sims", "",
                  "VRMSE on frames 10, 20, ..., 90, mean ± standard error over the sims of the group. \"Change\" is "
                  "new - old per sim, on the sims both checkpoints were scored on"
                  + (f" (old: {by_tag['old']['settings']['dit_ckpt']})." if "old" in by_tag else "."), "",
                  "| group | sims | VAE round trip | new checkpoint | old checkpoint | change per sim | sims better |",
                  "|---|---|---|---|---|---|---|"]
        changes = {}
        cell = lambda values: "–" if not values else "{:.4f} ± {:.4f}".format(*mean_se(values))
        for group in groups:
            rows = [row for row in new["rows"] if row["group"] == group]
            both = [row for row in rows if (group, row["sim_id"]) in old]
            before = [old[(group, row["sim_id"])]["scores"][BASE]["vrmse"] for row in both]
            changes[group] = [row["scores"][BASE]["vrmse"] - b for row, b in zip(both, before)]
            cells = [group, str(len(rows)),
                     cell([row["scores"]["roundtrip"]["vrmse"] for row in rows if "roundtrip" in row["scores"]]),
                     cell([row["scores"][BASE]["vrmse"] for row in rows]),
                     cell(before) + (f" ({len(both)} sims)" if both and len(both) != len(rows) else ""),
                     cell(changes[group]),
                     f"{sum(c < 0 for c in changes[group])}/{len(both)}" if both else "–"]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            if changes.get(a) and changes.get(b):
                (ma, sa), (mb, sb) = mean_se(changes[a]), mean_se(changes[b])
                lines.append(f"- Change on {a} sims minus change on {b} sims: {ma - mb:+.4f} ± {(sa ** 2 + sb ** 2) ** 0.5:.4f} "
                             "(negative: the model improved more on the first group).")
        lines.append("")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.report_only:
        report(args, out)
    else:
        score_shard(args, out)


if __name__ == "__main__":
    main()
