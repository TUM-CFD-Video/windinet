"""Which fields each colour image carries, and how the images are merged back to fields.

A group is a list of field names; each group gets its own FieldAdapter and its own VAE pass. Up to three fields
make a colour image; a larger group widens the VAE to as many channels. Default: the dataset spec's groups (eulermq,
ablations experiment 3: (rho, p) and (m_x, m_y)).
A field in no group is dropped; a field in several groups is averaged on the way back.
"""

from __future__ import annotations

import math
import string
from pathlib import Path

import torch
import yaml
from torch import nn

from windinet.data.frames import load_field_stats
from windinet.data.specs import SPECS, Spec

from .bijections import FieldAdapter, PreNorm


def group_tag(i: int) -> str:
    """Image group i as a letter A, B, ...: file names (vae_A.pt) and labels."""
    return string.ascii_uppercase[i]


def _norm_args(names: list[str], stats: dict[str, dict], log: set[str] | frozenset[str]) -> dict:
    keys = [f"log_{n}" if n in log else n for n in names]
    if missing := [k for k in keys if k not in stats]:
        raise KeyError(f"stats_json has no {', '.join(missing)}: recompute it with scripts/wan/field_stats.py")
    return {
        "mean": [stats[k]["mean"] for k in keys],
        "std": [stats[k]["std"] for k in keys],
        "lo": [stats[k]["min"] for k in keys],
        "hi": [stats[k]["max"] for k in keys],
        "log_channels": [n in log for n in names],
    }


def shared_vector_stats(stats: dict[str, dict], vector: tuple[str, str]) -> dict[str, dict]:
    """`stats` with both vector components on one D4-invariant scale: mean 0, the second moment pooled over both
    components (and so both signs), bounds +-the largest |value| of either."""
    std = math.sqrt(sum(stats[n]["mean"] ** 2 + stats[n]["std"] ** 2 for n in vector) / 2)
    m = max(abs(stats[n][k]) for n in vector for k in ("min", "max"))
    return stats | {n: stats[n] | {"mean": 0.0, "std": std, "min": -m, "max": m} for n in vector}


class GroupedAdapter(nn.Module):
    """One FieldAdapter per group; `fields` = the grouped dataset fields in dataset order; `channels` per image.
    The adapters log `log_fields` (None: the spec's); the loss space `base_norm` always logs the spec's.
    With the spec's shared_vector_scale, adapters and loss space take the vector's components on shared_vector_stats."""

    def __init__(
        self,
        groups: list[list[str]],
        stats: dict[str, dict],
        spec: Spec,
        log_fields: list[str] | None = None,
        clip_quantile: float = 0.98,
    ):
        super().__init__()
        self.groups = groups
        self.dataset_fields = spec.fields
        self.fields = [n for n in spec.fields if any(n in g for g in groups)]
        self.channels = max(3, *map(len, groups))
        log = spec.log_fields if log_fields is None else set(log_fields)
        if spec.shared_vector_scale:
            if logged := set(spec.vector) & (log | spec.log_fields):
                raise ValueError(f"{spec.name}: {sorted(logged)} share one vector scale and cannot be log fields")
            stats = shared_vector_stats(stats, spec.vector)
        self.base_norm = PreNorm(**_norm_args(self.fields, stats, spec.log_fields))
        self.adapters = nn.ModuleList(
            FieldAdapter(**_norm_args(g, stats, log), channels=self.channels, clip_quantile=clip_quantile)
            for g in groups
        )

    def select(self, x: torch.Tensor) -> torch.Tensor:
        """Dataset fields [B, C, ...] -> the grouped ones [B, len(fields), ...]."""
        return x[:, [self.dataset_fields.index(n) for n in self.fields]]

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


def load_adapter(run: str | Path, device: torch.device | str) -> GroupedAdapter:
    """The trained adapter of a VAE-stage run folder: groups, dataset and stats from its config.yaml, weights from
    adapter.pt (clip_quantile only sets an init, so the default stands in)."""
    run = Path(run)
    cfg = yaml.safe_load((run / "config.yaml").read_text())
    spec = SPECS[cfg["data"].get("dataset", "eulermq")]
    stats = load_field_stats(cfg["data"]["stats_json"])
    a = cfg["adapter"]
    adapter = GroupedAdapter(a["groups"], stats, spec, a.get("log_fields")).to(device)
    adapter.load_state_dict(torch.load(run / "adapter.pt", map_location=device))
    return adapter
