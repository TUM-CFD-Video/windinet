"""Frame-level access to the datasets in `specs`: train/val/test splits, frame or trajectory samplers, symmetries."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import h5py
import numpy as np
import torch

from windinet.data.specs import EULERMQ, Spec
from windinet.training.shockwave_data import ShockWaveDataset

SYMMETRIES = 8  # D4: symmetries of the equations and the open boundaries of both datasets, `symmetry`


def sim_ids(h5_path: str | Path) -> list[str]:
    """Sorted simulation ids of an HDF5 file."""
    with h5py.File(h5_path, "r") as f:
        return sorted(f.keys())


def split_ids(ids: list[str], test_every: int = 5, test_gamma: float | None = None) -> tuple[list[str], list[str]]:
    """Every `test_every`-th simulation of each gamma goes to test; with `test_gamma`, that gamma is the test set."""
    by_gamma: dict[float, list[str]] = defaultdict(list)
    for sid in sorted(ids):
        by_gamma[round(EULERMQ.scalars(sid)["gamma"], 3)].append(sid)
    if test_gamma is not None:
        test = by_gamma.pop(round(test_gamma, 3))
        return [sid for sims in by_gamma.values() for sid in sims], test
    train, test = [], []
    for sims in by_gamma.values():
        for i, sid in enumerate(sims):
            (test if i % test_every == 0 else train).append(sid)
    return train, test


def val_split(ids: list[str], n_val: int = 500, seed: int = 42) -> tuple[list[str], list[str]]:
    """The LTX baseline's split: sorted ids shuffled with a seeded torch generator, the last `n_val` held out."""
    ids = sorted(ids)
    perm = torch.randperm(len(ids), generator=torch.Generator().manual_seed(seed)).tolist()
    return [ids[i] for i in perm[:-n_val]], [ids[i] for i in perm[-n_val:]]


def val_cells(
    ids: list[str], per_cell: int, seed: int, scalars: Callable[[str], dict[str, float]]
) -> tuple[list[str], list[str]]:
    """`per_cell` random validation sims from each cell of equal scalars, the rest train; both sorted."""
    cells: dict[tuple, list[str]] = defaultdict(list)
    for sid in sorted(ids):
        cells[tuple(scalars(sid).values())].append(sid)
    rng = random.Random(seed)
    val = {sid for key in sorted(cells) for sid in rng.sample(cells[key], per_cell)}
    return sorted(set(ids) - val), sorted(val)


def splits(
    spec: Spec,
    h5_path: str,
    test_h5: str | None = None,
    val_sims: int = 500,
    split_seed: int = 42,
    test_every: int = 5,
    test_gamma: float | None = None,
) -> dict[str, tuple[str, list[str]]]:
    """{'train' | 'val' | 'test': (h5 file, sim ids)}. With `test_h5` the LTX protocol: validation sims held out of
    the training file (per cell if the spec says so), the test file whole; else (eulermq only) one file split by
    `split_ids`, and val = test."""
    ids = sim_ids(h5_path)
    if test_h5:
        if spec.val_per_cell:
            train, val = val_cells(ids, spec.val_per_cell, split_seed, spec.scalars)
        else:
            train, val = val_split(ids, val_sims, split_seed)
        return {"train": (h5_path, train), "val": (h5_path, val), "test": (test_h5, sim_ids(test_h5))}
    if spec is not EULERMQ:
        raise ValueError(f"{spec.name} needs data.test_h5")
    train, held_out = split_ids(ids, test_every, test_gamma)
    return {"train": (h5_path, train), "val": (h5_path, held_out), "test": (h5_path, held_out)}


def read_field(group: h5py.Group, name: str, frames) -> np.ndarray:
    """A stored field; log_<f> as ln of the stored f, in float64 before the cast."""
    x = group[name.removeprefix("log_")][frames, 0]
    return np.log(x) if name.startswith("log_") else x


class FrameSampler(ShockWaveDataset):
    """One item = `frames_per_item` frames of one simulation, fields [k, C, H, W] in `spec.fields` order and physical
    units, plus its scalars [S].

    Only the chosen frames are read from HDF5. `seed` None: random frames per access (training), else fixed
    (evaluation). `consecutive`: a clip (every `stride`-th frame) instead of frames spread over the trajectory.
    `frames_per_item` None: the whole trajectory in order (every `stride`-th frame, 4k + 1 of them). `augment`: every
    item under a random one of the dataset's symmetries."""

    def __init__(
        self,
        h5_path: str | Path,
        ids: list[str],
        frames_per_item: int | None,
        seed: int | None = None,
        consecutive: bool = False,
        stride: int = 1,
        augment: bool = False,
        *,
        spec: Spec,
    ):
        super().__init__(h5_path)
        self.ids = list(ids)
        self.k = frames_per_item
        self.seed = seed
        self.consecutive = consecutive
        self.stride = stride
        self.augment = augment
        self.spec = spec

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        self._init_file()
        sid = self.ids[idx]
        group = self._get_group(sid)
        n_frames = group[self.spec.fields[0].removeprefix("log_")].shape[0]
        if self.k is None:  # the whole trajectory, every stride-th frame, 4k + 1 of them (what Wan's VAE takes)
            strided = range(0, n_frames, self.stride)
            frames = list(strided)[: (len(strided) - 1) // 4 * 4 + 1]
        else:
            k = self.k
            # fixed frames belong to the sim, not to its list position, so a sharded loader evaluates the same clips
            rng = random if self.seed is None else random.Random(f"{self.seed}/{sid}")
            if self.consecutive:
                start = rng.randrange(n_frames - (k - 1) * self.stride)
                frames = list(range(start, start + k * self.stride, self.stride))
            else:
                frames = sorted(rng.sample(range(n_frames), k))
        fields = self.read(idx, frames)
        if self.augment:
            fields = symmetry(fields, random.randrange(SYMMETRIES), self.spec)
        return fields, torch.tensor(list(self.spec.scalars(sid).values()), dtype=torch.float32)

    def read(self, idx: int, frames: list[int]) -> torch.Tensor:
        """Fields [len(frames), C, H, W] of sim idx at these (increasing) frames, physical units."""
        self._init_file()
        group = self._get_group(self.ids[idx])
        return torch.from_numpy(np.stack([read_field(group, n, frames) for n in self.spec.fields], axis=1)).float()


def symmetry(fields: torch.Tensor, k: int, spec: Spec) -> torch.Tensor:
    """Element k (0..7) of D4 on fields [..., C, H, W] in `spec.fields` order, as `turn` on the grid; the vector's
    components swap and change sign with the axes."""
    fields = turn(fields, k)
    if spec.vector is None:
        return fields
    h, w = (spec.fields.index(n) for n in spec.vector)
    if k & 1:
        order = list(range(fields.shape[-3]))
        order[h], order[w] = w, h
        fields = fields[..., order, :, :]
    for bit, c in ((2, h), (4, w)):
        if k & bit:  # turn flipped a copy, so the sign change leaves the input alone
            fields[..., c, :, :] = -fields[..., c, :, :]
    return fields


def turn(x: torch.Tensor, k: int) -> torch.Tensor:
    """Element k of D4 on the grid alone [..., H, W]: bit 0 swaps the axes, bits 1 and 2 flip H and W."""
    if k & 1:
        x = x.transpose(-2, -1)
    for bit, dim in ((2, -2), (4, -1)):
        if k & bit:
            x = x.flip(dim)
    return x


def endless(loader):
    """Loop over a loader forever; unlike itertools.cycle it keeps no copy of what it yielded."""
    while True:
        yield from loader


def load_field_stats(stats_json: str | Path) -> dict[str, dict]:
    """scripts/wan/field_stats.py output: {field or log_field: {mean, std, min, max, ...}}."""
    return json.loads(Path(stats_json).read_text())["data_normalization_stats"]
