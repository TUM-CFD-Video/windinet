"""Which fields each colour image carries, and how the images are merged back to fields.

A group is a list of at most three field names; each group gets its own FieldAdapter
and its own VAE pass. Default (ablations, experiment 3): (rho, p) and (m_x, m_y).
A field in no group is dropped; a field in several groups is averaged on the way back.
"""

from __future__ import annotations

import string

import torch
from torch import nn

from windinet.eulermq.data import FIELDS

from .bijections import FieldAdapter, PreNorm

LOG_FIELDS = {"density", "pressure"}


def group_tag(i: int) -> str:
    """Image group i as a letter A, B, ...: file names (vae_A.pt) and labels."""
    return string.ascii_uppercase[i]


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
    """One FieldAdapter per group; `fields` = the grouped dataset fields in dataset order."""

    def __init__(self, groups: list[list[str]], stats: dict[str, dict]):
        super().__init__()
        self.groups = groups
        self.fields = [n for n in FIELDS if any(n in g for g in groups)]
        self.base_norm = PreNorm(**_norm_args(self.fields, stats))
        self.adapters = nn.ModuleList(FieldAdapter(**_norm_args(g, stats)) for g in groups)

    def select(self, x: torch.Tensor) -> torch.Tensor:
        """Dataset fields [B, 4, ...] -> the grouped ones [B, len(fields), ...]."""
        return x[:, [FIELDS.index(n) for n in self.fields]]

    def _inputs(self, x: torch.Tensor) -> list[torch.Tensor]:
        fields = dict(zip(self.fields, x.unbind(1), strict=True))
        return [torch.stack([fields[n] for n in g], dim=1) for g in self.groups]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Fields -> one RGB tensor per group."""
        return [a(inp) for a, inp in zip(self.adapters, self._inputs(x), strict=True)]

    def inverse(self, rgbs: list[torch.Tensor]) -> torch.Tensor:
        parts = [a.inverse(rgb) for a, rgb in zip(self.adapters, rgbs, strict=True)]
        per_field = {n: [] for n in self.fields}
        for group, part in zip(self.groups, parts, strict=True):
            for i, n in enumerate(group):
                per_field[n].append(part[:, i])
        return torch.stack([torch.stack(per_field[n]).mean(0) for n in self.fields], dim=1)

    def normalized(self, x: torch.Tensor) -> torch.Tensor:
        """`fields` z-scored: the space reconstruction losses are measured in, identical for every grouping."""
        return self.base_norm(x)

    @torch.no_grad()
    def init_from_data(self, x: torch.Tensor) -> None:
        for a, inp in zip(self.adapters, self._inputs(x), strict=True):
            a.init_from_data(inp)
