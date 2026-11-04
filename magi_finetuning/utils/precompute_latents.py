"""Pre-computed VAE latents for DiT training without the VAE in the loop, plus per-channel standardization stats.

Each sim of train.h5 (train + val split) and test.h5 is encoded once with the F3 encoder exactly as
data.build_latent_sequence does during training: [image latent of frame 0 | 25 video latents of frames 1..100 |
padding] = 30 latents (1, 16, 30, 32, 32). The encoder's posterior mode is deterministic and comes out of bf16, so
the latents are stored as bf16 *unscaled* (bit-exact): the DiT's current space is latents x scale_factor
(0.18215), the standardized one (latents - mean_c) / std_c with the train split's per-channel stats. Which one
to use is the training's choice (load_latents): a DiT resumed from a checkpoint needs the space it was trained in.

    <out>/<train|test>/<sim_id>.pt   {"latents": bf16 (16, 30, 32, 32), "num_body": 25, "gamma": float}
    <out>/stats.json                 per-channel mean / std of the train split's video latents 1..25 (unscaled)

    sbatch --time=01:00:00 jobs/jupiter/magi1/quick_test.sbatch -m magi_finetuning.utils.precompute_latents \
        --out <dir> --shard k --num-shards 4           (one job per shard; a resubmitted shard skips done sims)
    sbatch --dependency=afterok:<shards> jobs/jupiter/magi1/quick_test.sbatch \
        -m magi_finetuning.utils.precompute_latents --out <dir> --stats
"""

import argparse
import json
import time
from pathlib import Path

import torch

from magi_finetuning.data import Normalization, build_latent_sequence, load_split
from magi_finetuning.settings import TEST_DATA, TRAIN_DATA
from windinet.training.shockwave_data import CHANNEL_NAMES, ShockWaveDataset, parse_gamma

DATASETS = {"train": TRAIN_DATA, "test": TEST_DATA}  # file -> folder (train.h5 holds the train and val splits)
F3 = "/e/scratch/e-dev-2026d09-262/ig_work/magi1_vae_F3_vrmse_loss_io_lr20_wide16_refine64_40epochs/vae_best.safetensors"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--vae-ckpt", default=F3)
    p.add_argument("--out", required=True)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--stats", action="store_true", help="instead: per-channel stats of the train split's latents")
    return p.parse_args()


def load_latents(folder, split: str, sim_id: str, space: str = "dit", scale: float = 0.18215) -> dict:
    """One sim's latents, float32 (1, 16, 30, 32, 32), in the DiT's current space ("dit": x scale), standardized
    per channel ("standardized": train-split stats), or as stored ("raw")."""
    entry = torch.load(Path(folder) / split / f"{sim_id}.pt")
    z = entry["latents"].float()[None]
    if space == "dit":
        z = z * scale
    elif space == "standardized":
        stats = json.loads((Path(folder) / "stats.json").read_text())
        shape = (1, -1, 1, 1, 1)
        z = (z - torch.tensor(stats["mean"]).view(shape)) / torch.tensor(stats["std"]).view(shape)
    return {**entry, "latents": z}


@torch.no_grad()
def encode_all(args):
    from magi_finetuning.vae import load_vae_encoder

    out = Path(args.out)
    torch.cuda.set_device(0)
    encoder = load_vae_encoder(args.vae_ckpt)
    normalization = Normalization.of_vae_checkpoint(args.vae_ckpt)
    seconds = []
    for split, path in DATASETS.items():
        (out / split).mkdir(parents=True, exist_ok=True)
        dataset = ShockWaveDataset(path, num_sim_frames=101)
        for index in range(args.shard, len(dataset), args.num_shards):
            sim_id = dataset.ids[index]
            target = out / split / f"{sim_id}.pt"
            if target.is_file():
                continue
            sample = dataset[index]
            fields = torch.stack([sample[name] for name in CHANNEL_NAMES], dim=0)[None].cuda().float()
            torch.cuda.synchronize()
            started = time.perf_counter()
            latents, num_body = build_latent_sequence(encoder, normalization.apply(fields), 1.0)  # unscaled
            torch.cuda.synchronize()
            seconds.append(time.perf_counter() - started)
            torch.save({"latents": latents[0].bfloat16().cpu(), "num_body": num_body, "gamma": parse_gamma(sim_id)},
                       target)
            if len(seconds) % 100 == 1:
                print(f"[{split}] {index + 1}/{len(dataset)} {sim_id}: encode {seconds[-1]:.2f} s", flush=True)
    if seconds:
        print(f"shard {args.shard}: {len(seconds)} sims, encode mean {sum(seconds) / len(seconds):.3f} s per sim "
              f"(what each DiT training sample spends in the VAE)", flush=True)


def write_stats(args):
    """Mean / std per channel over the train split's video latents 1..25 (not val, not test), float64."""
    out = Path(args.out)
    total, total_sq, count = 0.0, 0.0, 0
    for sim_id in load_split()["train"]:
        entry = torch.load(out / "train" / f"{sim_id}.pt")
        z = entry["latents"][:, 1:1 + entry["num_body"]].double()  # (16, 25, 32, 32)
        total, total_sq, count = total + z.sum(dim=(1, 2, 3)), total_sq + z.pow(2).sum(dim=(1, 2, 3)), count + z[0].numel()
    mean = total / count
    std = (total_sq / count - mean ** 2).sqrt()
    stats = {"mean": mean.tolist(), "std": std.tolist(), "sims": len(load_split()["train"]), "split": "train",
             "latents": "video latents 1..25, unscaled", "vae_ckpt": args.vae_ckpt}
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    print("per channel (unscaled):\n" + "\n".join(f"  {c:2d}: mean {m:+.3f} std {s:.3f}"
                                                 for c, (m, s) in enumerate(zip(stats["mean"], stats["std"]))), flush=True)


if __name__ == "__main__":
    arguments = parse_args()
    write_stats(arguments) if arguments.stats else encode_all(arguments)
