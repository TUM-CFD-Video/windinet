"""The corrector stage's data: decoded Wan rollouts (scripts/wan/decode_rollouts.py) beside the simulations."""

from __future__ import annotations

import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

from windinet.data.frames import SYMMETRIES, FrameSampler, symmetry
from windinet.data.specs import EULERMQ


class RolloutFields(Dataset):
    """Item i: Wan's and the true fields [F, 4, H, W] of sim i and its gamma; item (i, end): a training window."""

    def __init__(self, dirs: list[str], split: str, h5: str, ids: list[str], stride: int, augment: bool = False):
        dirs = [Path(d) / split for d in dirs]
        held = [{p.stem for p in d.glob("*.pt")} for d in dirs]
        self.folders = {sid: [d for d, h in zip(dirs, held, strict=True) if sid in h] for sid in ids}
        self.ids = [sid for sid in ids if self.folders[sid]]
        self.sims = FrameSampler(h5, self.ids, None, stride=stride, spec=EULERMQ)
        self.stride, self.augment = stride, augment
        self.frames = len(self.load(0)) if self.ids else 0  # from the rollout file: the h5 stays closed before the fork

    def __len__(self) -> int:
        return len(self.ids)

    def load(self, i: int, folder: Path | None = None) -> torch.Tensor:
        """Wan's fields of sim i, fp16, memory-mapped: slicing reads only those frames."""
        return torch.load((folder or self.folders[self.ids[i]][0]) / f"{self.ids[i]}.pt", mmap=True)["fields"]

    def __getitem__(self, item: int | tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor, float]:
        """Whole rollout of the first folder; (i, end): Wan frames 0..end (the lookahead) and true frames 0..end-1 of a
        random folder holding the sim, under a random symmetry with `augment`."""
        if isinstance(item, tuple):
            i, end = item
            wan = self.load(i, random.choice(self.folders[self.ids[i]]))[: end + 1]
        else:
            i, wan = item, self.load(item)
            end = len(wan)
        truth = self.sims.read(i, list(range(0, end * self.stride, self.stride)))
        if self.augment:
            k = random.randrange(SYMMETRIES)
            wan, truth = symmetry(wan, k, EULERMQ), symmetry(truth, k, EULERMQ)
        return wan, truth, EULERMQ.scalars(self.ids[i])["gamma"]


class Windows:
    """Batch sampler of RolloutFields: `batch` random sims sharing one window end, items (sim, end); the window start
    is uniform in [1, frames - window] and drawn from `seed` alone, so every rank burns in to the same frame."""

    def __init__(self, sims: int, batch: int, window: int, frames: int, seed: int):
        self.sims, self.batch, self.window, self.frames = sims, batch, window, frames
        self.starts = random.Random(seed)

    def __iter__(self):
        while True:
            end = self.starts.randint(1, self.frames - self.window) + self.window
            yield [(i, end) for i in random.sample(range(self.sims), self.batch)]
