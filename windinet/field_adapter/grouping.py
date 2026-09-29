"""Which fields each colour image carries, and how the images are merged back to fields.

A group is a list of at most three field names; each group gets its own FieldAdapter
and its own VAE pass. Default (ablations, experiment 3): (rho, p) and (m_x, m_y).
A field that appears in several groups is averaged on the way back.
"""

from __future__ import annotations

import torch
from torch import nn

from windinet.training.shockwave_data import CHANNEL_NAMES as BASE_FIELDS

from .bijections import FieldAdapter, PreNorm

LOG_FIELDS = {"density", "pressure"}


def _norm_args(names: list[str], stats: dict[str, dict]) -> dict:
    keys = [f"log_{n}" if n in LOG_FIELDS else n for n in names]
    return {
        "mean": [stats[k]["mean"] for k in keys],
        "std": [stats[k]["std"] for k in keys],
        "lo": [stats[k]["min"] for k in keys],
        "hi": [stats[k]["max"] for k in keys],
        "log_channels": [n in LOG_FIELDS for n in names],
    }


class GroupedAdapter(nn.Module):
    """One FieldAdapter per group. forward: fields [B, 4, ...] -> list of RGB; inverse: list of RGB -> fields."""

    def __init__(self, groups: list[list[str]], stats: dict[str, dict]):
        super().__init__()
        self.groups = groups
        self.base_norm = PreNorm(**_norm_args(list(BASE_FIELDS), stats))
        self.adapters = nn.ModuleList(FieldAdapter(**_norm_args(g, stats)) for g in groups)

    def _inputs(self, x: torch.Tensor) -> list[torch.Tensor]:
        fields = dict(zip(BASE_FIELDS, x.unbind(1), strict=True))
        return [torch.stack([fields[n] for n in g], dim=1) for g in self.groups]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return [a(inp) for a, inp in zip(self.adapters, self._inputs(x), strict=True)]

    def inverse(self, rgbs: list[torch.Tensor]) -> torch.Tensor:
        parts = [a.inverse(rgb) for a, rgb in zip(self.adapters, rgbs, strict=True)]
        per_field = {n: [] for n in BASE_FIELDS}
        for group, part in zip(self.groups, parts, strict=True):
            for i, n in enumerate(group):
                per_field[n].append(part[:, i])
        return torch.stack([torch.stack(per_field[n]).mean(0) for n in BASE_FIELDS], dim=1)

    def normalized(self, x: torch.Tensor) -> torch.Tensor:
        """Base fields z-scored: the space reconstruction losses are measured in, identical for every grouping."""
        return self.base_norm(x)

    @torch.no_grad()
    def init_from_data(self, x: torch.Tensor) -> None:
        for a, inp in zip(self.adapters, self._inputs(x), strict=True):
            a.init_from_data(inp)
