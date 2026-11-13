"""Why the rollout's latent error grows from chunk to chunk: is it error accumulated from the earlier chunks?

On the sims of --noise-dir (as dit_latent_error.py), each chunk is generated three ways with the same start noise
and the same schedule (MAGI-1's windowed sampler, dit.rollout):

  free rollout     the chunks before it are the model's own (what evaluate.py does)
  true context     the chunks before it are the truth at every step, noised to the time t_j the sampler has
                   them at: t_j z_j + (1 - t_j) eps_j, clean once finished. This is what training shows the model.
  clean context    (--clean-context) the chunks before it are the clean truth from the first step on
  chunk 1 on the   (--chunk1-path) a full free rollout, except that chunk 1 follows the truth's path
  true path        t z_1 + (1 - t) eps_1 for the first 1, 2, 3 or all 4 quarters of its steps and is then released:
                   how much of the rollout's error is decided in each part of chunk 1's denoising. The true path
                   carries information about the truth, so this is a bound, as the true context is.

Chunk 1 has no earlier chunk and is the same in all three. If the error of chunk k with true context falls back to
chunk 1's, the growth is accumulated error; what remains is later frames being harder for the model at that
chunk position. Reported: the relative latent error RMS((z_gen - z) / std(z)) per chunk (as dit_latent_error.py)
and the VRMSE of the decoded frames of each chunk (variance of the truth over all frames).

Two checks on the first sim of shard 0: the free rollout here equals dit.rollout, and chunk 1 generated alone
equals chunk 1 of the free rollout (later chunks do not influence earlier ones, so each chunk can be generated
without the chunks after it).

    python -m magi_finetuning.eval.teacher_forced --dit-ckpt <ckpt> --out <dir> [--num-steps 4 --shift 5]
        [--sims 12] [--shard 0 --num-shards 3] [--clean-context]        then once: ... --out <dir> --report-only
"""

import argparse
import json
from pathlib import Path

import torch

from magi_finetuning.data import Normalization, build_latent_sequence, encode_image
from magi_finetuning.eval.dit_latent_error import chunk_slices, frames_of, relative_error
from magi_finetuning.eval.evaluate import SEED, decode_body, load_physical_fields
from magi_finetuning.eval.latent_sensitivity import VAES
from magi_finetuning.settings import CHUNK_WIDTH, OUTPUTS, SAMPLER_WINDOW, SCRATCH, TEST_DATA
from windinet.training.shockwave_data import ShockWaveDataset, parse_gamma

VARIANTS = {"free": "free rollout", "forced": "true context", "clean": "clean context",
            "path1": "chunk 1 true for 1/4 of its steps", "path2": "for 1/2", "path3": "for 3/4", "path4": "chunk 1 true"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dit-ckpt", default=str(SCRATCH / "magi1_dit_full_8000steps_on_F3" / "full_step008000.safetensors"))
    p.add_argument("--vae-ckpt", default=str(VAES["F3"]))
    p.add_argument("--noise-dir", default=str(OUTPUTS / "diagnostics" / "latent_sensitivity"),
                   help="latent_sensitivity.py output: its sims are used")
    p.add_argument("--test-data", default=TEST_DATA)
    p.add_argument("--num-steps", type=int, default=64)
    p.add_argument("--shift", type=float, default=3.0)
    p.add_argument("--window", type=int, default=SAMPLER_WINDOW)
    p.add_argument("--sims", type=int, default=None, help="use this many of the sims, evenly spread (default: all)")
    p.add_argument("--clean-context", action="store_true", help="also the variant with clean true context")
    p.add_argument("--chunk1-path", action="store_true",
                   help="also free rollouts with chunk 1 on the truth's path for its first 1..window stages")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--report-only", action="store_true", help="only merge <out>/scores_shard*.json into summary.md")
    p.add_argument("--out", required=True)
    return p.parse_args()


def free_rollout(model, image_latent, gamma, eps, tokens_per_chunk, num_steps, shift, clean_t, window, use_autocast,
                 truth0=None, force_first: int = 0):
    """dit.rollout with the start noise of every chunk given (eps: list of chunks). Returns the list of chunks.
    truth0, force_first: chunk 1 (index 0) is held on the truth's path t z + (1 - t) eps for its first force_first
    stages (each a quarter of its steps with window 4) and released at the path's point there; force_first =
    window makes it the truth itself."""
    from magi_finetuning.dit import time_schedule

    steps_per_stage = num_steps // window
    times = time_schedule(num_steps, shift, image_latent.device)
    chunks = []
    for stage in range(len(eps) + window - 1):
        if stage < len(eps):
            chunks.append(eps[stage].clone())
        first_active = max(0, stage - window + 1)
        active = range(first_active, len(chunks))
        for i in range(steps_per_stage):
            if truth0 is not None and (stage < force_first or (stage == force_first and i == 0)):
                t = times[stage * steps_per_stage + i]  # where chunk 1 is on its schedule (1 once it is finished)
                chunks[0] = t * truth0 + (1 - t) * eps[0]
            chunks[0][:, :, :1] = image_latent
            steps = torch.tensor([(stage - k) * steps_per_stage + i for k in active], device=image_latent.device)
            chunk_times = torch.full((1, len(chunks)), clean_t, device=image_latent.device)
            chunk_times[0, first_active:] = times[steps]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_autocast):
                velocity = model(torch.cat(chunks, dim=2), chunk_times, gamma, len(chunks), tokens_per_chunk)
            for k, step in zip(active, steps):
                v = velocity[:, :, k * CHUNK_WIDTH:(k + 1) * CHUNK_WIDTH].float()
                chunks[k] = chunks[k] + v * (times[step + 1] - times[step])
    chunks[0][:, :, :1] = image_latent
    return chunks


def forced_chunk(model, image_latent, gamma, eps, truth, k, tokens_per_chunk, num_steps, shift, clean_t, window,
                 use_autocast, clean_context=False):
    """Chunk k from its start noise eps[k], with the chunks before it taken from the truth (list of true chunks).
    The chunks after it are left out: they do not influence it. At stage s chunk j < k is where the sampler has it:
    at step (s - j) * steps_per_stage + i while active, clean once finished (always clean with clean_context)."""
    from magi_finetuning.dit import time_schedule

    steps_per_stage = num_steps // window
    times = time_schedule(num_steps, shift, image_latent.device)
    x = eps[k].clone()
    for stage in range(k, k + window):
        for i in range(steps_per_stage):
            step = (stage - k) * steps_per_stage + i
            chunk_times = torch.full((1, k + 1), clean_t, device=image_latent.device)
            context = []
            for j in range(k):
                active = stage - j < window and not clean_context
                t = times[(stage - j) * steps_per_stage + i] if active else clean_t
                chunk_times[0, j] = t
                context.append(t * truth[j] + (1 - t) * eps[j])
            chunk_times[0, k] = times[step]
            if k == 0:
                x[:, :, :1] = image_latent
            else:
                context[0][:, :, :1] = image_latent
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_autocast):
                velocity = model(torch.cat(context + [x], dim=2), chunk_times, gamma, k + 1, tokens_per_chunk)
            x = x + velocity[:, :, k * CHUNK_WIDTH:(k + 1) * CHUNK_WIDTH].float() * (times[step + 1] - times[step])
    if k == 0:
        x[:, :, :1] = image_latent
    return x


def errors(generated, latents, std, chunks, vae, scale, num_body, truth_frames) -> dict:
    """Latent error per chunk and overall, VRMSE of the decoded frames per chunk and overall."""
    error = generated - latents
    fields = decode_body(vae, generated / scale, num_body, truth_frames.shape[2])
    variance = truth_frames.var(dim=(0, 2, 3, 4))
    squared = (fields - truth_frames).pow(2)
    vrmse = lambda frames: (squared[:, :, frames].mean(dim=(0, 2, 3, 4)) / variance).sqrt().mean().item()
    return {"latent": [relative_error(error[:, :, c], std) for c in chunks],
            "latent_all": relative_error(error[:, :, 1:1 + num_body], std),
            "vrmse": [vrmse(slice(4 * c.start - 4, 4 * (c.stop - 1))) for c in chunks],
            "vrmse_all": vrmse(slice(None))}


def report_lines(rows, settings) -> list[str]:
    variants = [v for v in VARIANTS if v in rows[0]]
    chunks = chunk_slices(settings["num_body"])
    mean = lambda get: sum(get(r) for r in rows) / len(rows)
    lines = ["# Latent error per chunk: free rollout against true context", "",
             f"DiT {settings['dit_ckpt']}; {len(rows)} sims; {settings['num_steps']} Euler steps per chunk, shift "
             f"{settings['shift']}, {settings['window']} chunks at once. Same start noise in every variant. Definitions: "
             "magi_finetuning/eval/teacher_forced.py. Per sim: scores_shard*.json.", ""]
    for key, title in (("latent", "Relative latent error RMS((z_gen - z) / std(z))"),
                       ("vrmse", "VRMSE of the decoded frames of the chunk")):
        lines += [f"## {title}", "",
                  "| chunk | frames | " + " | ".join(VARIANTS[v] for v in variants) + " | growth over chunk 1 removed by true context |",
                  "|---|---|" + "---|" * (len(variants) + 1)]
        first = mean(lambda r: r["free"][key][0])
        for k, chunk in enumerate(chunks):
            values = {v: mean(lambda r: r[v][key][k]) for v in variants}
            removed = "-" if k == 0 else f"{100 * (values['free'] - values['forced']) / (values['free'] - first):.0f}%"
            lines.append(f"| {k + 1} | {frames_of(chunk)} | " + " | ".join(f"{values[v]:.3f}" for v in variants) + f" | {removed} |")
        lines.append("| all | 1..100 | " + " | ".join(f"{mean(lambda r: r[v][key + '_all']):.3f}" for v in variants) + " | |")
        lines.append("")
    checks = [r["checks"] for r in rows if "checks" in r]
    if checks:
        lines.append(f"Checks on one sim (largest absolute difference of latents): free rollout here against dit.rollout "
                     f"{checks[0]['free_vs_dit_rollout']:.2e}; chunk 1 generated alone against chunk 1 of the free rollout "
                     f"{checks[0]['chunk1_alone_vs_free']:.2e}.")
    return lines


@torch.no_grad()
def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if not args.report_only:
        from magi_finetuning.dit import load_trained_dit, rollout
        from magi_finetuning.utils.runtime import init_magi_parallel_state
        from magi_finetuning.vae import load_vae

        sim_ids = [r["sim_id"] for r in json.loads((Path(args.noise_dir) / "scores.json").read_text())]
        if args.sims:
            sim_ids = [sim_ids[round(i * (len(sim_ids) - 1) / max(1, args.sims - 1))] for i in range(args.sims)]
        torch.cuda.set_device(0)
        init_magi_parallel_state()
        vae = load_vae(args.vae_ckpt)
        normalization = Normalization.of_vae_checkpoint(args.vae_ckpt)
        model, magi_config, is_full_finetune = load_trained_dit(args.dit_ckpt)
        scale, clean_t = magi_config.runtime_config.scale_factor, magi_config.runtime_config.clean_t
        test_set = ShockWaveDataset(args.test_data, num_sim_frames=101)
        sampler = dict(num_steps=args.num_steps, shift=args.shift, clean_t=clean_t, window=args.window,
                       use_autocast=is_full_finetune)

        rows = []
        my_ids = sim_ids[args.shard::args.num_shards]
        for count, sim_id in enumerate(my_ids, 1):
            index = test_set.ids.index(sim_id)
            truth = normalization.apply(load_physical_fields(test_set, sim_id))
            latents, num_body = build_latent_sequence(vae.encoder, truth, scale)
            num_chunks = latents.shape[2] // CHUNK_WIDTH
            tokens_per_chunk = CHUNK_WIDTH * (latents.shape[3] // 2) * (latents.shape[4] // 2)
            gamma = torch.tensor([[parse_gamma(sim_id)]], device="cuda")
            image_latent = encode_image(vae.encoder, truth[:, :, :1]) * scale
            true_chunks = list(latents.split(CHUNK_WIDTH, dim=2))
            chunk_shape = image_latent.shape[:2] + (CHUNK_WIDTH,) + image_latent.shape[3:]
            torch.manual_seed(SEED + index)  # the draws of dit.rollout, in its order
            eps = [torch.randn(chunk_shape, device="cuda") for _ in range(num_chunks)]

            free = free_rollout(model, image_latent, gamma, eps, tokens_per_chunk, **sampler)
            generated = {"free": torch.cat(free, dim=2)}
            for name, clean in (("forced", False), ("clean", True)):
                if clean and not args.clean_context:
                    continue
                generated[name] = torch.cat([free[0]] + [
                    forced_chunk(model, image_latent, gamma, eps, true_chunks, k, tokens_per_chunk, **sampler,
                                 clean_context=clean) for k in range(1, num_chunks)], dim=2)

            for stages in range(1, args.window + 1) if args.chunk1_path else ():
                generated[f"path{stages}"] = torch.cat(free_rollout(
                    model, image_latent, gamma, eps, tokens_per_chunk, **sampler, truth0=true_chunks[0],
                    force_first=stages), dim=2)

            std = latents[:, :, 1:1 + num_body].std(dim=(0, 2, 3, 4), keepdim=True)
            chunks = chunk_slices(num_body)
            row = {"sim_id": sim_id, "gamma": parse_gamma(sim_id), "num_body": num_body,
                   **{name: errors(x, latents, std, chunks, vae, scale, num_body, truth[:, :, 1:])
                      for name, x in generated.items()}}
            if count == 1 and args.shard == 0:
                torch.manual_seed(SEED + index)
                reference = rollout(model, image_latent, gamma, num_chunks, tokens_per_chunk, **sampler)
                alone = forced_chunk(model, image_latent, gamma, eps, true_chunks, 0, tokens_per_chunk, **sampler)
                row["checks"] = {"free_vs_dit_rollout": (generated["free"] - reference).abs().max().item(),
                                 "chunk1_alone_vs_free": (alone - free[0]).abs().max().item()}
                print(f"checks: {row['checks']}", flush=True)
            rows.append(row)
            print(f"[{count}/{len(my_ids)}] {sim_id}: latent error per chunk, free "
                  + " / ".join(f"{v:.2f}" for v in row["free"]["latent"]) + "; true context "
                  + " / ".join(f"{v:.2f}" for v in row["forced"]["latent"]), flush=True)
        (out / f"scores_shard{args.shard}.json").write_text(json.dumps({
            "settings": {"dit_ckpt": args.dit_ckpt, "num_steps": args.num_steps, "shift": args.shift,
                         "window": args.window, "num_body": rows[0]["num_body"]}, "rows": rows}))
        if args.num_shards > 1:
            return

    shards = [json.loads(path.read_text()) for path in sorted(out.glob("scores_shard*.json"))]
    rows = [row for shard in shards for row in shard["rows"]]
    lines = report_lines(rows, shards[0]["settings"])
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
