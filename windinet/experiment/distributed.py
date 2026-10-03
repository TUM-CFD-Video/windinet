"""Multi-GPU through torchrun: every rank trains on its slice of the data, gradients are averaged, evaluation
sums are reduced, rank 0 writes. Without torchrun (one process) every function is a no-op."""

from __future__ import annotations

import atexit
import os
import sys

import torch
import torch.distributed as dist

rank = int(os.environ.get("RANK", 0))
world = int(os.environ.get("WORLD_SIZE", 1))


def setup() -> None:
    """Process group, this rank's GPU, and stdout only on rank 0."""
    if world == 1:
        return
    dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
    atexit.register(dist.destroy_process_group)
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    if rank:
        sys.stdout = open(os.devnull, "w")


def shard(items: list) -> list:
    """This rank's part of a list."""
    return items[rank::world]


def all_sum(*tensors: torch.Tensor) -> None:
    """In place: each tensor becomes its sum over ranks."""
    if world > 1:
        for t in tensors:
            dist.all_reduce(t)


def all_mean(*tensors: torch.Tensor) -> None:
    """In place: each tensor becomes its mean over ranks (gradient averaging)."""
    if world > 1:
        all_sum(*tensors)
        for t in tensors:
            t /= world


def sync(module: torch.nn.Module) -> None:
    """Copy rank 0's parameters and buffers to every rank (after a data-dependent init)."""
    if world > 1:
        for t in module.state_dict().values():
            dist.broadcast(t, 0)
