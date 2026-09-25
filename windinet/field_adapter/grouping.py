"""Which fields each 3-channel adapter sees, and how their outputs are merged back.

    single:   one lossy adapter over all four base fields            4 -> 3
    triplets: two exact adapters, A = (m_x, m_y, p), B = (rho, p, E)  2 x (3 -> 3)
    pairs:    two exact adapters, A = (rho, p), B = (m_x, m_y)        2 x (2 -> 3), third colour is redundancy

Pressure sits in both triplets on purpose (shared edges); on the way back it is the
mean of the two. Energy is derived from the base fields with the sim's gamma:
E = p / (gamma - 1) + (m_x^2 + m_y^2) / (2 rho), strictly positive, so log-normalised.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from windinet.training.shockwave_data import CHANNEL_NAMES as BASE_FIELDS

from .bijections import FieldAdapter, PreNorm

LOG_FIELDS = {"density", "pressure", "energy"}
GROUPINGS = {
    "single": [list(BASE_FIELDS)],
    "triplets": [["momentum_x", "momentum_y", "pressure"], ["density", "pressure", "energy"]],
    "pairs": [["density", "pressure"], ["momentum_x", "momentum_y"]],
}


def energy(x: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    """x [B, 4, H, W] base fields in physical units, gamma [B] -> total energy [B, H, W]."""
    rho, mx, my, p = x.unbind(1)
    return p / (gamma.view(-1, 1, 1) - 1) + 0.5 * (mx * mx + my * my) / rho


def energy_stats(x: torch.Tensor, gamma: torch.Tensor) -> dict[str, float]:
    """log-energy mean/std from a data batch, same shape as the entries of the stats JSON."""
    e = energy(x, gamma).log()
    return {"mean": e.mean().item(), "std": e.std().item(), "min": e.min().item(), "max": e.max().item()}


def _norm_args(names: list[str], stats: dict[str, dict]) -> dict:
    keys = [f"log_{n}" if n in LOG_FIELDS else n for n in names]
    return {"mean": [stats[k]["mean"] for k in keys], "std": [stats[k]["std"] for k in keys],
            "lo": [stats[k]["min"] for k in keys], "hi": [stats[k]["max"] for k in keys],
            "log_channels": [n in LOG_FIELDS for n in names]}


class GroupedAdapter(nn.Module):
    """One FieldAdapter per group. forward: fields -> list of RGB; inverse: list of RGB -> fields."""

    def __init__(self, grouping: str, stats: dict[str, dict], bins: int = 8, warp: bool = True):
        super().__init__()
        self.groups = GROUPINGS[grouping]
        self.base_norm = PreNorm(**_norm_args(list(BASE_FIELDS), stats))
        self.adapters = nn.ModuleList(FieldAdapter(**_norm_args(g, stats), bins=bins, warp=warp) for g in self.groups)

    def _fields(self, x: torch.Tensor, gamma: torch.Tensor) -> dict[str, torch.Tensor]:
        fields = dict(zip(BASE_FIELDS, x.unbind(1)))
        if any("energy" in g for g in self.groups):
            fields["energy"] = energy(x, gamma)
        return fields

    def _inputs(self, x: torch.Tensor, gamma: torch.Tensor) -> list[torch.Tensor]:
        fields = self._fields(x, gamma)
        return [torch.stack([fields[n] for n in g], dim=1) for g in self.groups]

    def forward(self, x: torch.Tensor, gamma: torch.Tensor) -> list[torch.Tensor]:
        return [a(inp) for a, inp in zip(self.adapters, self._inputs(x, gamma))]

    def inverse(self, rgbs: list[torch.Tensor]) -> torch.Tensor:
        parts = [a.inverse(rgb) for a, rgb in zip(self.adapters, rgbs)]
        per_field = {n: [] for n in BASE_FIELDS}
        for group, part in zip(self.groups, parts):
            for i, n in enumerate(group):
                if n in per_field:
                    per_field[n].append(part[:, i])
        return torch.stack([torch.stack(per_field[n]).mean(0) for n in BASE_FIELDS], dim=1)

    def normalized(self, x: torch.Tensor) -> torch.Tensor:
        """Base fields z-scored: the space reconstruction losses are measured in, identical for every grouping."""
        return self.base_norm(x)

    @torch.no_grad()
    def init_from_data(self, x: torch.Tensor, gamma: torch.Tensor, mix: str = "pca") -> None:
        for a, inp in zip(self.adapters, self._inputs(x, gamma)):
            a.init_from_data(inp, mix=mix)

    def slope_penalty(self) -> torch.Tensor:
        return torch.stack([a.warp.slope_penalty() for a in self.adapters]).mean()
