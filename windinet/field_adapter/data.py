"""Frame-level access to euler_mq simulations for per-pixel adapter training."""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from windinet.training.shockwave_data import CHANNEL_NAMES, ShockWaveDataset, parse_gamma


def split_ids(ids: list[str], test_every: int = 5) -> tuple[list[str], list[str]]:
    """Every `test_every`-th simulation of each gamma goes to test, so both splits cover all regimes."""
    by_gamma: dict[float, list[str]] = defaultdict(list)
    for sid in sorted(ids):
        by_gamma[parse_gamma(sid)].append(sid)
    train, test = [], []
    for sims in by_gamma.values():
        for i, sid in enumerate(sims):
            (test if i % test_every == 0 else train).append(sid)
    return train, test


class FrameSampler(ShockWaveDataset):
    """One item = `frames_per_item` frames of one simulation: fields [k, 4, H, W] in physical units, plus gamma.

    Reads only the chosen frames from HDF5 (not the whole 100-frame trajectory).
    Frames are random per access when `seed` is None, fixed otherwise (evaluation).
    """

    def __init__(self, h5_path: str | Path, ids: list[str], frames_per_item: int, seed: int | None = None):
        super().__init__(h5_path)
        self.ids = list(ids)
        self.k = frames_per_item
        self.seed = seed

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, float]:
        self._init_file()
        sid = self.ids[idx]
        group = self._get_group(sid)
        n_frames = group[CHANNEL_NAMES[0]].shape[0]
        rng = random.Random(None if self.seed is None else self.seed + idx)
        frames = sorted(rng.sample(range(n_frames), self.k))
        fields = np.stack([group[name][frames, 0] for name in CHANNEL_NAMES], axis=1)  # [k, 4, H, W]
        return torch.from_numpy(fields).float(), parse_gamma(sid)


def load_field_stats(stats_json: str | Path) -> dict[str, dict]:
    """scripts/compute_channel_stats.py output: {field or log_field: {mean, std, ...}}."""
    import json

    return json.loads(Path(stats_json).read_text())["data_normalization_stats"]
