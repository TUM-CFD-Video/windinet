"""DiT training entry point: LoRA (train_lora.py) or, with --full-finetune, every weight (train_full.py).
Run from the repo root as a module, env active (normally through jobs/jupiter/magi1/*.sbatch):

    accelerate launch --multi_gpu --num_processes 4 -m magi_finetuning.train.finetune --vae-ckpt <vae>       # LoRA
    accelerate launch ... -m magi_finetuning.train.finetune --full-finetune --ckpt-dir <scratch> ...       # full
"""

import argparse

from magi_finetuning.settings import DEFAULT_FRAMES, OUTPUTS, TRAIN_DATA


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=TRAIN_DATA, help="train.h5; training uses its fixed train split only")
    p.add_argument("--out", default=str(OUTPUTS / "finetune"), help="args and small files (on SCRATCH)")
    p.add_argument("--frames", type=int, default=DEFAULT_FRAMES,
                   help="frames to generate after the given frame 0: 1..F (100 = Weihao; 96 also works)")
    p.add_argument("--max-samples", type=int, default=None, help="train on ~N sims of the train split, spread over all gammas")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--shift", type=float, default=3.0, help="MAGI's sd3 timestep shift (t=0 noise, t=1 clean)")
    p.add_argument("--clean-chunks", action="store_true",
                   help="MAGI-1's video-continuation samples: some samples start with clean chunks (up to 5%% noise, "
                        "no loss), as often as the windowed sampler meets them (dit.sample_num_clean_chunks)")
    p.add_argument("--save-every", type=int, default=250)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--vae-ckpt", default=None, help="adapted VAE from magi_finetuning/train/train_vae.py (default: released VAE)")
    # LoRA only (full FT uses settings.FULL_FT instead)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--grad-accum", type=int, default=None, help="default: 32 / number of GPUs (effective batch 32)")
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--ckpt-dir", default=None,
                   help="checkpoints (SCRATCH). Full FT: required unless --no-save, never under home/IG_work. "
                        "LoRA: optional, default <out>/checkpoints")
    # full finetuning only
    p.add_argument("--full-finetune", action="store_true", help="train all DiT weights + gamma embedding")
    p.add_argument("--no-save", action="store_true", help="write no checkpoints (smoke runs)")
    p.add_argument("--save-optimizer", action="store_true", help="also save the 8-bit optimizer state")
    p.add_argument("--max-minutes", type=float, default=None, help="stop cleanly after this wall time")
    p.add_argument("--resume", action="store_true",
                   help="full FT: continue from <ckpt-dir>/resume.pt if it exists (otherwise start fresh)")
    p.add_argument("--inference-kv-range", choices=["block_causal", "noise2clean"], default="block_causal",
                   help="recorded for sampling: cross-chunk attention at inference (training is block-causal)")
    args = p.parse_args()
    if args.frames % 4 or not 4 <= args.frames <= 100:
        raise ValueError("--frames must be a multiple of 4 (the VAE's time patch) between 4 and 100")
    return args


def main():
    args = parse_args()
    if args.full_finetune:
        from magi_finetuning.train.train_full import run
    else:
        from magi_finetuning.train.train_lora import run
    run(args)


if __name__ == "__main__":
    main()
