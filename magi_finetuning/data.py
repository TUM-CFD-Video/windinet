"""Euler data for MAGI-1: the fixed train/val split, channel normalization, the training loader, and the
latent layout the DiT trains and generates on.

Split: configs/magi1/train_val_split_256.csv (settings.SPLIT_FILE), one row per simulation of train.h5
with columns sim_id, gamma, split. Made once with a fixed rule, no randomness: within every gamma, sort
the ids and send every 10th one to val, so each of the 10 gammas has 405 train and 45 val simulations.
test.h5 is for inference only. The ids come in blocks by gamma (0000-0449 = 1.13, ...), so "the first N"
simulations are all ONE gamma; use stratified_subset to pick a few spread over all gammas.

Normalization (on the fly, per batch, from a statistics yaml):
    normalized = clamp((f(x) - mean) / (CLIP * std), -1, 1)      per channel
    f = log for the channels in log_fields (density, pressure), identity for the others
It belongs to the VAE: train_vae writes it into the checkpoint metadata, and the DiT trainers and
evaluation read it back (Normalization.of_vae_checkpoint), so a VAE -> DiT -> evaluation chain always
uses one normalization. Checkpoints without it (older runs) used DEFAULT.
"""

import csv
import json
from dataclasses import dataclass
from functools import cached_property

import torch
import yaml
from inference.model.vae import DiagonalGaussianDistribution
from torch.utils.data import DataLoader, Subset

from magi_finetuning.settings import CHUNK_WIDTH, NORMALIZATION_CLIP, NORMALIZATION_STATS, SPLIT_FILE
from magi_finetuning.utils.resume import EpochShuffle
from windinet.training.shockwave_data import CHANNEL_NAMES, ShockWaveDataset, parse_gamma


# ---------------------------------------------------------------- train / val split


def load_split() -> dict:
    """{"train": [sim ids], "val": [sim ids]} from the split table."""
    split = {"train": [], "val": []}
    with open(SPLIT_FILE, newline="") as table:
        for row in csv.DictReader(table):
            split[row["split"]].append(row["sim_id"])
    return split


def by_gamma(sim_ids) -> dict:
    """{gamma: sorted ids}, gamma rounded to 4 decimals (keeps 1.40 and 1.404 apart)."""
    groups = {}
    for sim_id in sorted(sim_ids):
        groups.setdefault(round(parse_gamma(sim_id), 4), []).append(sim_id)
    return groups


def stratified_subset(sim_ids, total: int | None) -> list:
    """~total ids with the same count from every gamma, evenly spaced within each (all ids if total is None)."""
    if not total:
        return sorted(sim_ids)
    groups = by_gamma(sim_ids)
    per_gamma = max(1, total // len(groups))
    picked = []
    for ids in groups.values():
        step = len(ids) / min(per_gamma, len(ids))
        picked += [ids[int(k * step)] for k in range(min(per_gamma, len(ids)))]
    return sorted(picked)


def indices_of(dataset: ShockWaveDataset, sim_ids) -> list:
    """Row numbers of these ids in the dataset."""
    position = {sim_id: index for index, sim_id in enumerate(dataset.ids)}
    return [position[sim_id] for sim_id in sim_ids]


# ---------------------------------------------------------------- normalization


LOGGABLE = ("density", "pressure")  # strictly positive fields


@dataclass(frozen=True)
class Normalization:
    stats_file: str
    log_fields: tuple[str, ...] = ()
    clip: float = NORMALIZATION_CLIP

    def __post_init__(self):
        unknown = set(self.log_fields) - set(LOGGABLE)
        if unknown:
            raise ValueError(f"can only log {LOGGABLE}, not {sorted(unknown)}")

    @cached_property
    def mean_std(self) -> tuple[list[float], list[float]]:
        """Per-channel (mean, std) in CHANNEL_NAMES order, of log(x) for the logged channels."""
        stats = yaml.safe_load(open(self.stats_file))["data_normalization_stats"]
        keys = [f"log_{name}" if name in self.log_fields else name for name in CHANNEL_NAMES]
        missing = [key for key in keys if key not in stats]
        if missing:
            raise KeyError(f"{self.stats_file} has no stats for {missing}")
        return [stats[key]["mean"] for key in keys], [stats[key]["std"] for key in keys]

    def _shape(self, x):
        return (1, len(CHANNEL_NAMES)) + (1,) * (x.ndim - 2)

    def _logged(self, x) -> torch.Tensor:
        """(1, 4, 1, ...) bool: which channels are logged."""
        flags = [name in self.log_fields for name in CHANNEL_NAMES]
        return torch.tensor(flags, device=x.device).view(self._shape(x))

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """Raw physical fields (B, 4, ...) -> normalized, in [-1, 1]."""
        mean, std = self.mean_std
        if self.log_fields:
            x = torch.where(self._logged(x), x.clamp_min(1e-12).log(), x)
        return ((x - x.new_tensor(mean).view(self._shape(x))) / (x.new_tensor(std).view(self._shape(x)) * self.clip)).clamp(-1, 1)

    def invert(self, y: torch.Tensor) -> torch.Tensor:
        """Normalized (B, 4, ...) -> physical fields (exact inverse inside the clip)."""
        mean, std = self.mean_std
        x = y * y.new_tensor(std).view(self._shape(y)) * self.clip + y.new_tensor(mean).view(self._shape(y))
        if self.log_fields:
            x = torch.where(self._logged(x), x.exp(), x)
        return x

    def to_metadata(self) -> dict[str, str]:
        """For safetensors metadata (string values only)."""
        return {"normalization": json.dumps({"stats_file": str(self.stats_file), "log_fields": list(self.log_fields),
                                             "clip": self.clip})}

    @classmethod
    def from_metadata(cls, metadata: dict | None) -> "Normalization":
        if not metadata or "normalization" not in metadata:
            return DEFAULT
        saved = json.loads(metadata["normalization"])
        return cls(saved["stats_file"], tuple(saved["log_fields"]), float(saved["clip"]))

    @classmethod
    def of_vae_checkpoint(cls, vae_checkpoint: str | None) -> "Normalization":
        """The normalization a VAE was trained with (DEFAULT for no checkpoint or an older one)."""
        if not vae_checkpoint:
            return DEFAULT
        from safetensors import safe_open

        with safe_open(vae_checkpoint, framework="pt") as f:
            return cls.from_metadata(f.metadata())

    def describe(self) -> str:
        logged = ", ".join(f"log {name}" for name in self.log_fields) or "no log fields"
        return f"{self.stats_file} ({logged}, clip {self.clip})"


# What every run before the log-variables option used (and the first official run keeps using).
DEFAULT = Normalization(str(NORMALIZATION_STATS))


# ---------------------------------------------------------------- training loader


def make_loader(args, seed: int):
    """Batch size 1 (MAGI's no-MagiAttention path needs it), train split only (val is held out);
    --max-samples takes an even spread over all gammas. The order is an EpochShuffle (exactly
    rebuildable after a resume; resume.batches_from moves it to the next epoch).
    Returns (loader, sampler)."""
    ds = ShockWaveDataset(args.data, num_sim_frames=1 + args.frames)  # frame 0 (given) + frames 1..F
    train_ids = stratified_subset(load_split()["train"], args.max_samples)
    ds = Subset(ds, indices_of(ds, train_ids))
    sampler = EpochShuffle(len(ds), seed)
    return DataLoader(ds, batch_size=1, sampler=sampler, num_workers=4, drop_last=True), sampler


# ---------------------------------------------------------------- latent layout (frozen VAE encoder -> DiT input)


@torch.no_grad()
def encode(enc, x):
    """(1, C, T, 256, 256) in [-1, 1] -> posterior mode, 16-frame tiles (the VAE's native clip length)."""
    z = [DiagonalGaussianDistribution(enc(x[:, :, i:i + 16].bfloat16())).mode() for i in range(0, x.shape[2], 16)]
    return torch.cat(z, dim=2).float()


@torch.no_grad()
def encode_image(enc, frame):
    """One frame (1, C, 1, H, W) -> one latent frame, the way MAGI-1 encodes an image for image-to-video:
    the frame repeated 4x in time (the VAE's temporal patch), first latent frame of the posterior mode."""
    moments = enc(frame.expand(-1, -1, 4, -1, -1).bfloat16())
    return DiagonalGaussianDistribution(moments).mode()[:, :, :1].float()


@torch.no_grad()
def build_latent_sequence(enc, fields, scale):
    """First-frame conditioning (MAGI-1's image-to-video), as in Weihao's LTX runs:

        fields:  normalized (1, 4, 1 + F, H, W) = frame 0 (the given state) + frames 1..F (to generate)
        latents: [ image | F/4 body latents (frames 1..F) | padding ]  ->  whole chunks of 6

    Frames 1..F are padded to a multiple of 16 (the VAE's tile) by repeating frame F, as Weihao pads
    to 105; then whole latents are appended to fill the last chunk. The image latent is kept clean
    (inpainted every step); the padding is generated but never scored or shown.
    F = 100 (Weihao's horizon): 1 image + 25 body + 3 + 1 padding = 30 latents = 5 chunks.
    Returns (latents (1, 16, L, h, w) scaled, number of body latents = F / 4).
    """
    frames = fields[:, :, 1:]
    num_frames = frames.shape[2]
    pad_frames = -num_frames % 16
    frames = torch.cat([frames, frames[:, :, -1:].expand(-1, -1, pad_frames, -1, -1)], dim=2)

    image = encode_image(enc, fields[:, :, :1])
    body = encode(enc, frames)                         # (num_frames + pad_frames) / 4 latents
    pad_latents = -(1 + body.shape[2]) % CHUNK_WIDTH
    filler = body[:, :, -1:].expand(-1, -1, pad_latents, -1, -1)
    return torch.cat([image, body, filler], dim=2) * scale, num_frames // 4


def batch_to_latents(batch, enc, normalization, scale):
    """Training batch -> (latents, gamma (1, 1), number of body latents, number of chunks, tokens per chunk).
    Body latents are latent frames 1 .. num_body (frames 1..F); see build_latent_sequence.
    normalization: the VAE's own (normalization.Normalization.of_vae_checkpoint)."""
    x = torch.stack([batch[c] for c in CHANNEL_NAMES], dim=1)  # (1, 4, 1 + F, H, W), already on GPU
    latents, num_body = build_latent_sequence(enc, normalization.apply(x), scale)
    gamma = batch["meta"]["gamma"].float()[:, None]
    num_chunks = latents.shape[2] // CHUNK_WIDTH
    tokens_per_chunk = CHUNK_WIDTH * (latents.shape[3] // 2) * (latents.shape[4] // 2)
    return latents, gamma, num_body, num_chunks, tokens_per_chunk
