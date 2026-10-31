"""Exact continuation of a training run across 12 h Slurm jobs.

A resumed job must see the same data in the same order and draw the same random numbers as one long
job would. Besides weights, optimizer, scheduler and step (saved by each trainer), that takes:

    data order   EpochShuffle: the order of epoch e depends only on (seed, e), and the trainers count
                 the samples each GPU consumed (micro_batches_done); a resumed job rebuilds that epoch's
                 order and skips the samples already seen (batches_from). The epoch is passed with
                 start_epoch(), deliberately NOT set_epoch(): Accelerate's data loader calls set_epoch()
                 with its own iteration counter, which restarts at 0 in every new job.
    random draws every GPU saves its torch / CUDA / numpy / python RNG states (rng_rank<r>.pt next to
                 resume.pt) and restores them just before the training loop, so timesteps and noise
                 continue where they stopped.

Exact up to GPU kernel nondeterminism (bf16 reductions), not bit for bit. If the GPU count changed or a
resume state predates this file, the RNG can't be restored; the trainers then reseed with seed + step
and say so.
"""

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Sampler


class EpochShuffle(Sampler):
    """A shuffled order of range(n) that depends only on (seed, epoch), so it can be rebuilt exactly."""

    def __init__(self, n: int, seed: int):
        self.n, self.seed, self.epoch = n, seed, 0

    def start_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self):
        return self.n

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        return iter(torch.randperm(self.n, generator=generator).tolist())


def batches_from(loader, accelerator, micro_batches_done: int, epoch_aware=()):
    """Endless stream of this GPU's batches, starting after the first micro_batches_done of them.

    loader is the accelerator-prepared DataLoader built on an EpochShuffle; epoch_aware are the objects
    whose start_epoch must be called at every epoch (the sampler, and a dataset with per-epoch randomness).
    """
    per_epoch = len(loader)  # batches per GPU per epoch
    epoch, skip = divmod(micro_batches_done, per_epoch)
    while True:
        for obj in epoch_aware:
            obj.start_epoch(epoch)
        epoch_loader = accelerator.skip_first_batches(loader, skip) if skip else loader
        yield from epoch_loader
        epoch, skip = epoch + 1, 0


def rng_file(checkpoint_dir: Path, rank: int) -> Path:
    return Path(checkpoint_dir) / f"rng_rank{rank}.pt"


def save_rng_states(checkpoint_dir: Path, rank: int) -> None:
    """Called by EVERY GPU: its own RNG states."""
    torch.save({
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }, rng_file(checkpoint_dir, rank))


def restore_rng_states(checkpoint_dir: Path, rank: int, saved_world_size: int | None, world_size: int) -> bool:
    """Called by EVERY GPU right before the training loop. False if the states can't be restored (older
    resume state, or a different number of GPUs); the caller then reseeds."""
    path = rng_file(checkpoint_dir, rank)
    if saved_world_size != world_size or not path.is_file():
        return False
    states = torch.load(path, map_location="cpu", weights_only=False)
    torch.set_rng_state(states["torch"])
    torch.cuda.set_rng_state(states["cuda"])
    np.random.set_state(states["numpy"])
    random.setstate(states["python"])
    return True
