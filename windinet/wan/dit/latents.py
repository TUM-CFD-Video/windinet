"""The DiT stage's data: the latents written by scripts/wan/encode_latents.py, one file per simulation."""

from __future__ import annotations

import random
import re
from pathlib import Path

import torch
import yaml
from torch.utils.data import Dataset

from windinet.data.frames import SYMMETRIES, turn, turn_vector
from windinet.data.specs import Spec


class LatentDataset(Dataset):
    """<dir>/<split>/<sim id>.pt -> (latents [G, 16, F', h, w], condition [S]). `ids` selects simulations (default:
    every file of the split), `frames` keeps the first n latent frames, `window` a random window of n instead. With
    `masks` also the sim's detail shares [F', h, w] (<masks>/<sim id>.pt, turned like its copy)."""

    def __init__(
        self,
        dir: str | Path,
        split: str,
        ids=None,
        frames: int | None = None,
        window: int | None = None,
        masks: str | Path | None = None,
    ):
        self.dir = Path(dir) / split
        self.ids = list(ids) if ids is not None else sorted(p.stem for p in self.dir.glob("*.pt"))
        self.frames, self.window, self.masks = frames, window, masks and Path(masks)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int) -> tuple:
        item = torch.load(self.dir / f"{self.ids[i]}.pt")
        z = item["latents"][:, :, : self.frames]
        j = random.randrange(z.shape[2] - self.window + 1) if self.window else 0
        z = z[:, :, j : j + self.window] if self.window else z
        if not self.masks:
            return z, stored_condition(item)
        sid, k, start = parse_latent_id(self.ids[i])
        assert not start, "detail masks of later starts are not supported"
        share = turn(torch.load(self.masks / f"{sid}.pt").float(), k)[: self.frames]
        return z, stored_condition(item), share[j : j + z.shape[2]]


class WithLatents(Dataset):
    """The items (frames, scalars) of a FrameSampler as (frames, latents): each sim's latents from <dir>/<split>/<sim
    id>.pt (DiT rollouts for the VAE stage) in place of its condition."""

    def __init__(self, frames: Dataset, dir: str | Path, split: str):
        self.frames, self.latents, self.ids = frames, LatentDataset(dir, split, frames.ids), frames.ids

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.frames[i][0], self.latents[i][0]


def condition(spec: Spec, sid: str, k: int = 0) -> dict:
    """What a latent file keeps beside its latents: the sim's scalars and, under symmetry k, the dataset's direction."""
    entry = {"scalars": spec.scalars(sid)}
    return entry | ({"direction": turn_vector(spec.direction, k)} if spec.direction else {})


def stored_condition(item: dict) -> torch.Tensor:
    """The condition [S] float32 of a latent file: its scalars, then its direction."""
    scalars = item.get("scalars") or {"gamma": item["gamma"]}  # files before the scalars: eulermq's gamma alone
    return torch.tensor([*scalars.values(), *item.get("direction", ())], dtype=torch.float32)


def encode_info(dir: str | Path) -> dict:
    """encode.yaml of a latent folder: the adapter run and groups that produced it, and the provenance."""
    return yaml.safe_load((Path(dir) / "encode.yaml").read_text())


def check_encoding(dir: str | Path, change_stats: str | None) -> None:
    """Refuse latents whose VAE run (encode.yaml's config.load) has another adapter.change_stats: states and changes
    do not mix."""
    run = Path(encode_info(dir)["config"]["load"])
    theirs = yaml.safe_load((run / "config.yaml").read_text())["adapter"].get("change_stats")
    if theirs != change_stats:
        raise ValueError(f"{dir} was encoded with adapter.change_stats {theirs}, this VAE run has {change_stats}")


def latent_id(sid: str, k: int = 0, start: int = 0) -> str:
    """File stem of a simulation's latents: the sim id as recorded, _s<k> under data.frames.symmetry k, _t<start>
    for the clip from that frame on."""
    return sid + (f"_s{k}" if k else "") + (f"_t{start}" if start else "")


def parse_latent_id(stem: str) -> tuple[str, int, int]:
    """The inverse of latent_id: (sim id, symmetry k, start)."""
    sid, k, start = re.fullmatch(r"(.+?)(?:_s(\d))?(?:_t(\d+))?", stem).groups()
    return sid, int(k or 0), int(start or 0)


def with_augmentations(dir: str | Path, ids: list[str]) -> list[str]:
    """`ids` plus the symmetric copies and the later starts the folder holds (encode.yaml: orientations, starts;
    folders before orientations: all 8 with data.augment)."""
    info = encode_info(dir)
    ks = info.get("orientations", range(SYMMETRIES if info["config"]["data"].get("augment") else 1))
    return [latent_id(sid, k, t) for sid in ids for k in ks for t in info.get("starts", [0])]
