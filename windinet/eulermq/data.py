"""Frame-level access to euler_mq simulations: field names, train/val/test splits and frame or trajectory samplers."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch

from windinet.training.shockwave_data import CHANNEL_NAMES, ShockWaveDataset, parse_gamma

FIELDS = CHANNEL_NAMES  # dataset order: density, momentum_x, momentum_y, pressure
SYMBOLS = {"density": "ρ", "momentum_x": "m_x", "momentum_y": "m_y", "pressure": "p"}


def sim_ids(h5_path: str | Path) -> list[str]:
    """Sorted simulation ids of an HDF5 file."""
    with h5py.File(h5_path, "r") as f:
        return sorted(f.keys())


def split_ids(ids: list[str], test_every: int = 5, test_gamma: float | None = None) -> tuple[list[str], list[str]]:
    """Every `test_every`-th simulation of each gamma goes to test; with `test_gamma`, that gamma is the test set."""
    by_gamma: dict[float, list[str]] = defaultdict(list)
    for sid in sorted(ids):
        by_gamma[round(parse_gamma(sid), 3)].append(sid)
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


class FrameSampler(ShockWaveDataset):
    """One item = `frames_per_item` frames of one simulation, fields [k, 4, H, W] in physical units, plus gamma.

    Only the chosen frames are read from HDF5. `seed` None: random frames per access (training), else fixed
    (evaluation). `consecutive`: a clip instead of frames spread over the trajectory. `frames_per_item` None:
    the whole trajectory in order."""

    def __init__(
        self,
        h5_path: str | Path,
        ids: list[str],
        frames_per_item: int | None,
        seed: int | None = None,
        consecutive: bool = False,
    ):
        super().__init__(h5_path)
        self.ids = list(ids)
        self.k = frames_per_item
        self.seed = seed
        self.consecutive = consecutive

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, float]:
        self._init_file()
        sid = self.ids[idx]
        group = self._get_group(sid)
        n_frames = group[CHANNEL_NAMES[0]].shape[0]
        k = self.k or n_frames
        # fixed frames belong to the sim, not to its list position, so a sharded loader evaluates the same clips
        rng = random if self.seed is None else random.Random(f"{self.seed}/{sid}")
        start = rng.randrange(n_frames - k + 1)
        frames = list(range(start, start + k)) if self.consecutive else sorted(rng.sample(range(n_frames), k))
        fields = np.stack([group[name][frames, 0] for name in CHANNEL_NAMES], axis=1)  # [k, 4, H, W]
        return torch.from_numpy(fields).float(), parse_gamma(sid)


def load_field_stats(stats_json: str | Path) -> dict[str, dict]:
    """scripts/compute_channel_stats.py output: {field or log_field: {mean, std, min, max}}."""
    return json.loads(Path(stats_json).read_text())["data_normalization_stats"]
