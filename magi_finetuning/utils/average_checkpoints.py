"""Weight average ("checkpoint soup") of full-finetune DiT checkpoints: a plain or weighted mean of every tensor,
accumulated in float32, saved in the checkpoints' own dtype (bf16). The result loads like any
full_stepNNNNNN.safetensors (dit.load_trained_dit), e.g. into eval/checkpoint_curve.py or eval/evaluate.py.

    sbatch --time=00:20:00 jobs/jupiter/magi1/quick_test.sbatch -m magi_finetuning.utils.average_checkpoints \
        --out <soup>.safetensors <ckpt> <ckpt> [...] [--weights 0.5 0.5]
"""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoints", nargs="+")
    p.add_argument("--weights", type=float, nargs="+", default=None, help="default: equal; normalized to sum 1")
    p.add_argument("--out", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    weights = args.weights or [1.0] * len(args.checkpoints)
    if len(weights) != len(args.checkpoints):
        raise ValueError("one weight per checkpoint")
    weights = [w / sum(weights) for w in weights]
    total, dtypes = None, {}
    for checkpoint, weight in zip(args.checkpoints, weights):
        state = load_file(checkpoint)
        if total is None:
            total = {name: tensor.float() * weight for name, tensor in state.items()}
            dtypes = {name: tensor.dtype for name, tensor in state.items()}
        else:
            if state.keys() != total.keys():
                raise ValueError(f"{checkpoint}: different tensors than {args.checkpoints[0]}")
            for name, tensor in state.items():
                total[name] += tensor.float() * weight
        print(f"added {checkpoint} (weight {weight:.3f})", flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_file({name: tensor.to(dtypes[name]).contiguous() for name, tensor in total.items()}, str(out))
    out.with_suffix(".json").write_text(json.dumps({"checkpoints": args.checkpoints, "weights": weights}, indent=1))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
