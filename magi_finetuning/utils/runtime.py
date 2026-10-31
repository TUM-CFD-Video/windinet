"""What every training job needs around the training itself:

    multi-GPU setup   Accelerate (as in windinet's trainers) + the parallel_state MAGI-1's model code requires
    job timing        every job appends one record to <checkpoint dir>/timing.json, so a stage that spans
                      several resumed 12 h jobs adds up correctly; evaluate.py turns them into the runtime table
    pre-flight checks 8-bit AdamW really works on this GPU; the checkpoint folder is allowed and has room
"""

import json
import os
import shutil
import socket
import time
from pathlib import Path

import torch
import torch.nn as nn
from accelerate import Accelerator
from inference.infra.distributed import parallel_state as mpu

from magi_finetuning.compat import IG_WORK


# ---------------------------------------------------------------- multi-GPU setup


def init_distributed(grad_accum: int, kwargs_handlers=None) -> Accelerator:
    """Training: Accelerate for multi-GPU, plus the parallel_state MAGI-1's model code requires."""
    acc = Accelerator(gradient_accumulation_steps=grad_accum, mixed_precision="no",  # dtypes are MAGI's own
                      kwargs_handlers=kwargs_handlers)
    # acc.device is "cuda" (no index) in a plain single-process run, which set_device rejects.
    torch.cuda.set_device(acc.local_process_index)
    init_magi_parallel_state()
    return acc


def init_magi_parallel_state() -> None:
    """MAGI-1 builds its model only after its parallel_state exists, even on a single GPU. Starts a
    1-process group if none is running (single-GPU training, evaluation), then MAGI's groups with no
    model parallelism: every extra GPU is plain data parallel."""
    if not torch.distributed.is_initialized():
        s = socket.socket(); s.bind(("", 0)); port = s.getsockname()[1]; s.close()
        torch.distributed.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}", rank=0, world_size=1)
    if not mpu.model_parallel_is_initialized():
        mpu.initialize_model_parallel(cp_size=1, pp_size=1)


def shutdown(acc: Accelerator) -> None:
    acc.wait_for_everyone()
    acc.end_training()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


# ---------------------------------------------------------------- job timing


class JobTimer:
    """Create at the very start of a job; call .training_started() right before the training loop and
    .record(...) at the end (GPU 0 only)."""

    def __init__(self):
        self.job_start = time.time()
        self.loop_start = None

    def training_started(self):
        self.loop_start = time.time()

    def record(self, path: Path, stage: str, steps_from: int, steps_to: int, num_gpus: int) -> dict:
        end = time.time()
        steps = max(steps_to - steps_from, 0)
        train_seconds = end - (self.loop_start or self.job_start)
        entry = {
            "stage": stage,
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "start": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(self.job_start)),
            "end": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(end)),
            "steps_from": steps_from,
            "steps_to": steps_to,
            "setup_seconds": round((self.loop_start or end) - self.job_start, 1),
            "train_seconds": round(train_seconds, 1),
            "wall_seconds": round(end - self.job_start, 1),
            "sec_per_step": round(train_seconds / steps, 2) if steps else None,
            "num_gpus": num_gpus,
            "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1),
        }
        path = Path(path)
        records = json.loads(path.read_text()) if path.is_file() else []
        records.append(entry)
        path.write_text(json.dumps(records, indent=1))
        return entry


def summarize(path: Path) -> dict | None:
    """All jobs of one stage combined: steps, wall time, GPU-hours, mean seconds per step, peak memory."""
    path = Path(path)
    if not path.is_file():
        return None
    records = json.loads(path.read_text())
    if not records:
        return None
    steps = sum(r["steps_to"] - r["steps_from"] for r in records)
    train_seconds = sum(r["train_seconds"] for r in records)
    wall_seconds = sum(r["wall_seconds"] for r in records)
    return {
        "jobs": len(records),
        "steps": steps,
        "wall_hours": wall_seconds / 3600,
        "gpu_hours": sum(r["wall_seconds"] * r["num_gpus"] for r in records) / 3600,
        "setup_minutes_per_job": sum(r["setup_seconds"] for r in records) / len(records) / 60,
        "sec_per_step": train_seconds / steps if steps else None,
        "num_gpus": records[-1]["num_gpus"],
        "peak_gpu_mem_gb": max(r["peak_gpu_mem_gb"] for r in records),
    }


# ---------------------------------------------------------------- pre-flight checks


def check_adamw8bit() -> None:
    """Fail fast unless bitsandbytes' 8-bit AdamW really steps on this GPU (aarch64 build)."""
    import bitsandbytes as bnb

    p = nn.Parameter(torch.randn(8192, device="cuda"))  # >= 4096 elements, so the state is really 8-bit
    opt = bnb.optim.AdamW8bit([p], lr=1e-3)
    p.grad = torch.randn_like(p)
    before = p.detach().clone()
    opt.step()
    torch.cuda.synchronize()
    state = opt.state[p]
    if torch.equal(before, p) or state["state1"].dtype != torch.uint8 or not state["state1"].is_cuda:
        raise RuntimeError(f"bitsandbytes AdamW8bit did not take a real 8-bit GPU step: {state.keys()}")


def check_ckpt_dir(args, n_params: int, n_saves: int) -> Path:
    """--ckpt-dir is required, must be outside home/IG_work, and must have room for every planned save."""
    if not args.ckpt_dir:
        raise ValueError("this run needs --ckpt-dir (no default; outside home/IG_work)")
    d = Path(args.ckpt_dir).resolve()
    for forbidden in (Path("/e/home"), Path.home().resolve(), IG_WORK):
        if d == forbidden or forbidden in d.parents:
            raise ValueError(f"--ckpt-dir {d} is under {forbidden}; large checkpoints must not go to home/IG_work")
    d.mkdir(parents=True, exist_ok=True)
    weights = n_params * 2  # bf16 weights-only
    optim = n_params * 2 if args.save_optimizer else 0  # two uint8 moments per parameter (+ small fp32 parts)
    need = int(1.1 * n_saves * (weights + optim))
    free = shutil.disk_usage(d).free
    if free < need:
        raise RuntimeError(f"{d}: need ~{need / 1e9:.0f} GB for {n_saves} saves, only {free / 1e9:.0f} GB free")
    print(f"[ckpt] {d}: {n_saves} saves x {(weights + optim) / 1e9:.1f} GB planned, {free / 1e9:.0f} GB free "
          f"(filesystem free space; a project quota may be lower)", flush=True)
    return d
