"""The DiT stage's data: the latents written by scripts/wan/encode_latents.py, one file per simulation."""

from __future__ import annotations

from pathlib import Path

import torch
import yaml
from torch.utils.data import Dataset


class LatentDataset(Dataset):
    """<dir>/<split>/<sim id>.pt -> (latents [G, 16, F', h, w], gamma). `ids` selects simulations (default: every
    file of the split), `frames` keeps the first n latent frames."""

    def __init__(self, dir: str | Path, split: str, ids: list[str] | None = None, frames: int | None = None):
        self.dir = Path(dir) / split
        self.ids = list(ids) if ids is not None else sorted(p.stem for p in self.dir.glob("*.pt"))
        self.frames = frames

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, float]:
        item = torch.load(self.dir / f"{self.ids[i]}.pt")
        return item["latents"][:, :, : self.frames], float(item["gamma"])


def encode_info(dir: str | Path) -> dict:
    """encode.yaml of a latent folder: the adapter run and groups that produced it, and the provenance."""
    return yaml.safe_load((Path(dir) / "encode.yaml").read_text())


def latent_id(sid: str, k: int = 0) -> str:
    """File stem of a simulation's latents: the sim id as recorded, <sim id>_s<k> under eulermq.data.symmetry k."""
    return sid if k == 0 else f"{sid}_s{k}"


def with_augmentations(dir: str | Path, ids: list[str]) -> list[str]:
    """`ids` plus the symmetric copies the folder holds (encode.yaml: augmentations)."""
    n = encode_info(dir)["config"].get("augmentations", 1)
    return [latent_id(sid, k) for sid in ids for k in range(n)]
