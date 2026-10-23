"""
ShockWave dataset for ShockWaveNet.

Loads ShockWave CFD simulations from HDF5 file.

Two datasets are supported, selected by name (config.VaeDataConfig.dataset;
see DATASETS below):

"euler_mq" -- Euler_MQ, conditioned on the adiabatic index gamma:

train.h5

    <sample_id>/              e.g. 0000_gamma1.22
        density       [T,1,H,W]
        momentum_x    [T,1,H,W]
        momentum_y    [T,1,H,W]
        pressure      [T,1,H,W]

"srba" -- 2D-SRBA-OOOO (shock-induced R22 bubble collapse, open boundaries),
conditioned on the shock Mach number only (bubble count/positions/radii are
visible in the first frame):

train.h5

    <sample_id>/              e.g. 005_Mas1.30_sb1_Ax0.0129_Ay0.0081_Ar0.0035_BxNaN_...
        density, pressure, velocityX, velocityY    [T,1,H,W]
        density_1, density_2, diffuse_volume_fraction_1   (ignored)

    Channels are pressure, velocityX, velocityY and log_density = log(density),
    the latter computed here so everything downstream treats it as an ordinary
    field (stats key "log_density", no log_transform_channels needed).

Returns physical fields for LTX preprocessing.
"""

import json
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import h5py
import torch
from torch import Tensor
from torch.utils.data import Dataset


def parse_gamma(sid: str) -> float:
    """Physical adiabatic index encoded in a sample id, e.g. "0000_gamma1.22" -> 1.22."""
    return float(sid.split("gamma")[1])


def parse_srba_scalars(sid: str) -> dict[str, float]:
    """Scalars encoded in a 2D-SRBA sample id, e.g.
    "005_Mas1.30_sb1_Ax0.0129_..." -> {"mach": 1.30, "num_bubbles": 1}."""
    mach = re.search(r"_Mas([0-9.]+)", sid)
    bubbles = re.search(r"_sb(\d+)", sid)
    if mach is None or bubbles is None:
        raise ValueError(f"not a 2D-SRBA sample id (no _Mas<..>/_sb<..>): {sid!r}")
    return {"mach": float(mach.group(1)), "num_bubbles": float(bubbles.group(1))}


@dataclass(frozen=True)
class DatasetSpec:
    """What a dataset's HDF5 holds and how its sample ids encode the scalars.

    channels: channel name -> (HDF5 field, take log). The channel names are
        what data.channel_order, the normalization stats and every
        sample[name] lookup use.
    parse_scalars: sample id -> scalar conditioning values (sample["meta"]).
    spread_scalar: the scalar visualization picks spread over
        (pick_spread_ids), and the default DiT conditioning scalar.
    """

    channels: dict[str, tuple[str, bool]]
    parse_scalars: Callable[[str], dict[str, float]]
    spread_scalar: str


DATASETS: dict[str, DatasetSpec] = {
    "euler_mq": DatasetSpec(
        channels={name: (name, False) for name in ["density", "momentum_x", "momentum_y", "pressure"]},
        parse_scalars=lambda sid: {"gamma": parse_gamma(sid)},
        spread_scalar="gamma",
    ),
    "srba": DatasetSpec(
        channels={
            "pressure": ("pressure", False),
            "velocityX": ("velocityX", False),
            "velocityY": ("velocityY", False),
            "log_density": ("density", True),
        },
        parse_scalars=parse_srba_scalars,
        spread_scalar="mach",
    ),
}
DEFAULT_DATASET = "euler_mq"


def pick_spread_ids(ids: list[str], n: int, dataset: str = DEFAULT_DATASET) -> set[str]:
    """N sample ids spanning the range of the dataset's spread scalar (gamma /
    Mach) in `ids`: min/median/max for n=3, evenly-spaced quantiles otherwise.
    Deterministic, so every eval script that calls it on the same id list
    shows the same sims in its figures."""
    if n <= 0:
        return set()
    spec = DATASETS[dataset]
    order = sorted(range(len(ids)), key=lambda k: spec.parse_scalars(ids[k])[spec.spread_scalar])
    n = min(n, len(order))
    if n == 1:
        picks = [order[len(order) // 2]]
    else:
        picks = [order[round(i * (len(order) - 1) / (n - 1))] for i in range(n)]
    return {ids[i] for i in picks}


class ShockWaveDataset(Dataset):
    """
    Load ShockWave CFD simulations from HDF5.

    Each sample corresponds to one simulation.

    `dataset` names the DATASETS entry the file belongs to ("euler_mq" or
    "srba"); it fixes which fields are read and how the scalars are parsed.

    Returns:
        <channel name>:
            [T,H,W], one per DATASETS[dataset].channels entry
            (euler_mq: density, momentum_x, momentum_y, pressure;
            srba: pressure, velocityX, velocityY, log_density)

        meta:
            scalar conditions parsed from the id
            (euler_mq: gamma; srba: mach, num_bubbles)

        id:
            sample name
    """

    def __init__(
        self,
        h5_path: str | Path,
        num_sim_frames: int | None = None,
        dataset: str = DEFAULT_DATASET,
    ):
        self.h5_path = Path(h5_path)
        self.file = None
        if dataset not in DATASETS:
            raise ValueError(f"unknown dataset {dataset!r}, expected one of {sorted(DATASETS)}")
        self.dataset = dataset
        self.spec = DATASETS[dataset]

        if not self.h5_path.is_file():
            raise FileNotFoundError(f"Shockwave HDF5 file not found: {self.h5_path}")

        time.sleep(self._rank_stagger_delay())
        with self._open(self.h5_path) as f:
            self.ids = sorted(list(f.keys()))
            self._prewarm_ids = self._pick_prewarm_ids(f, self.ids)

        self.num_sim_frames = num_sim_frames

    @staticmethod
    def _pick_prewarm_ids(f: h5py.File, ids: list[str]) -> list[str]:
        """
        One sample id per distinct external-link target file.

        Each sample in train.h5 is an ``h5py.ExternalLink`` into a separate
        per-gamma shard file (e.g. ``train_subsets/train_gamma1.130_128.hdf5``).
        Resolving a link for the first time makes HDF5 open its target file;
        with several ranks shuffling through samples independently, those
        first-opens land at unpredictable, possibly simultaneous points deep
        into training instead of a controlled startup window. `_prewarm_external_targets`
        uses this list to force one resolution per distinct target up front,
        while ranks are still staggered, so that burst happens once and only
        once rather than scattered through the epoch.
        """
        seen_targets = set()
        prewarm_ids = []
        for sid in ids:
            link = f.get(sid, getlink=True)
            target = getattr(link, "filename", None)
            if target is None:
                # Not an external link (e.g. a self-contained h5 file) --
                # nothing to pre-resolve.
                continue
            if target not in seen_targets:
                seen_targets.add(target)
                prewarm_ids.append(sid)
        return prewarm_ids


    def __len__(self):
        return len(self.ids)


    @staticmethod
    def _rank_stagger_delay(spread: float = 0.3) -> float:
        """
        Per-rank delay to spread out each process's first file access.

        Every rank in a distributed run reaches its first HDF5 access at
        nearly the same wall-clock instant (dataset construction, then the
        first `__getitem__`), so without this, all ranks hit the network
        filesystem in the same burst regardless of the retry jitter in
        `_backoff_sleep` -- that only helps *after* a collision has already
        happened. Staggering by rank spreads the initial burst out
        preemptively. Falls back to 0 outside a distributed run.
        """
        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
        return rank * spread

    @staticmethod
    def _backoff_sleep(attempt: int) -> None:
        """
        Sleep with exponential backoff plus random jitter.

        All ranks in a distributed run hit the dataset near-simultaneously
        (e.g. all fetching the first batch at once), so a fixed backoff
        schedule with no jitter has every rank retry in lockstep -- they
        keep re-colliding on the network filesystem at the same instants
        instead of spreading out. The jitter desynchronizes them.
        """
        base = 0.5 * (2 ** attempt)
        time.sleep(base + random.uniform(0, base))

    @staticmethod
    def _open(path: Path, retries: int = 8) -> h5py.File:
        """
        Open the HDF5 file, retrying with backoff.

        train.h5 stores each sample as an h5py.ExternalLink into a separate
        per-gamma shard file (e.g. train_subsets/train_gamma1.130_128.hdf5).
        On this environment's h5py/HDF5 build, resolving that link depends on
        the process's current working directory rather than falling back to
        train.h5's own directory -- confirmed by the link resolving fine when
        opened with the process cwd'd into the same directory as train.h5,
        but failing (single process, no concurrency involved) when opened by
        absolute path from a different cwd, exactly the situation training
        runs in. Setting HDF5_EXT_PREFIX to this file's own directory makes
        link resolution explicit instead of relying on that cwd-dependent
        fallback. The retry-with-backoff below is a separate, unrelated
        mitigation for genuine transient network-filesystem hiccups on LRZ
        DSS and is kept as defense in depth.
        """
        os.environ["HDF5_EXT_PREFIX"] = str(path.parent)
        for attempt in range(retries):
            try:
                try:
                    return h5py.File(path, "r", locking=False)
                except TypeError:
                    # Installed h5py/libhdf5 too old to support the `locking` kwarg.
                    return h5py.File(path, "r")
            except OSError:
                if attempt == retries - 1:
                    raise
                ShockWaveDataset._backoff_sleep(attempt)
        raise AssertionError("unreachable")

    def _init_file(self):
        """
        Lazy open h5 file.
        Important for DataLoader workers.
        """
        if self.file is None:
            time.sleep(self._rank_stagger_delay())
            self.file = self._open(self.h5_path)

            for sid in self._prewarm_ids:
                self._get_group(sid)

    def __del__(self):
        if self.file is not None:
            self.file.close()

    def _get_group(self, sid: str, retries: int = 8):
        """
        Look up one simulation's group, retrying with a fresh file handle.

        Under multi-rank distributed training, `self.file[sid]` has been
        observed to transiently fail with "Unable to synchronously open
        object (can't open file)" even though `self.file` itself opened fine
        -- this never happens in a single-process run, so it looks like
        contention on the network filesystem rather than a real missing key.
        Retry with a freshly reopened handle rather than trusting the stale
        one. See `_backoff_sleep` for why the retries are jittered.
        """
        for attempt in range(retries):
            try:
                return self.file[sid]
            except KeyError:
                if attempt == retries - 1:
                    raise
                self._backoff_sleep(attempt)
                self.file.close()
                self.file = self._open(self.h5_path)
        raise AssertionError("unreachable")

    def __getitem__(self, idx):

        self._init_file()

        sid = self.ids[idx]

        sample = self._get_group(sid)


        out = {}
        for name, (field, take_log) in self.spec.channels.items():
            data = sample[field]
            if self.num_sim_frames is not None:
                data = data[:self.num_sim_frames]
            else:
                data = data[:]
            x = torch.from_numpy(data).float().squeeze(1)
            out[name] = torch.log(x.clamp(min=LOG_TRANSFORM_EPS)) if take_log else x

        out["meta"] = self.spec.parse_scalars(sid)
        out["id"] = sid
        return out


LOG_TRANSFORM_EPS = 1e-6


def build_shockwave_video(
    sample: dict[str, Tensor],
    device: torch.device,
    channel_mean: list[float] | None = None,
    channel_std: list[float] | None = None,
    normalization_clip: float = 5.0,
    channel_order: list[str] | None = None,
    log_transform_channels: list[str] | None = None,
) -> Tensor:
    """Build a shockwave video [B, C, F, H, W], C = len(channel_order) (default 4).

    Default channel layout:
        0 - density
        1 - momentum_x
        2 - momentum_y
        3 - pressure

    `channel_order` overrides which physical fields are stacked, in which
    channel index -- a permutation of CHANNEL_NAMES or a subset of it (e.g. the
    3-channel ["density", "momentum_x", "momentum_y"] run), positionally
    aligned with `channel_mean`/`channel_std` (config.VaeDataConfig.channel_order
    validates both). Defaults to CHANNEL_NAMES, the layout every run before
    the 2026-08 channel-order sweep used.

    `log_transform_channels` (config.VaeDataConfig.log_transform_channels)
    names physical fields to replace with `log(field)` *before* z-score
    normalization -- restricted by that config's own validator to the two
    strictly-positive fields (density, pressure), since log() is undefined
    for momentum_x/momentum_y's signed values. `channel_mean`/`channel_std`
    for these channels must already be the log-space statistics (e.g.
    euler_mq_128_only_train.yaml's `log_density` block via
    VaeDataConfig.normalization_stats_file, which resolves this
    automatically) -- this function has no way to tell raw stats from
    log-space ones, it just z-scores whatever it's given. Clamped to
    LOG_TRANSFORM_EPS first so a not-quite-zero field value (data noise,
    not expected in practice -- the real dataset's density/pressure floors
    are ~0.02/0.001) can't silently produce -inf/NaN.

    Unlike WinDiNet, ShockWaveNet does not prepend an artificial
    conditioning frame. The first simulation frame is treated as
    the physical initial condition (IC).

    The temporal dimension is padded to 8n+1 as required by the
    LTX-Video VAE.
    """

    order = channel_order or CHANNEL_NAMES

    fields = {name: sample[name] for name in order}

    # Ensure batch dimension
    for name, tensor in fields.items():
        if tensor.ndim == 3:
            fields[name] = tensor.unsqueeze(0)

    for name in log_transform_channels or []:
        fields[name] = torch.log(fields[name].clamp(min=LOG_TRANSFORM_EPS))

    # each field: [B, T, H, W]
    x = torch.stack([fields[name] for name in order], dim=1)  # [B, C, F, H, W]

    if channel_mean is not None and channel_std is not None:
        mean = x.new_tensor(channel_mean).view(1, len(order), 1, 1, 1)
        std = x.new_tensor(channel_std).view(1, len(order), 1, 1, 1)
        x = ((x - mean) / (std * normalization_clip)).clamp(-1.0, 1.0)

    x = pad_frames_8n1(x)

    return x.to(
        device=device,
        dtype=torch.float32,
    ).contiguous()


CHANNEL_NAMES = ["density", "momentum_x", "momentum_y", "pressure"]


def load_training_config(path: str | Path) -> dict:
    """Read a dumped training_config.yaml (VAE or DiT run) as a plain dict.

    It carries python-tagged values (e.g. the model-source enum). Ignore unknown
    tags rather than reaching for unsafe_load, which would execute whatever the
    file names.
    """
    import yaml

    class _TolerantLoader(yaml.SafeLoader):
        pass

    _TolerantLoader.add_multi_constructor("", lambda loader, suffix, node: None)
    return yaml.load(Path(path).read_text(), Loader=_TolerantLoader) or {}


def run_scalar_conditioning(checkpoint: str | Path | None, fallback: dict) -> dict:
    """The scalar_conditioning block of the DiT run a checkpoint belongs to
    (<output_dir>/checkpoints/<ckpt> -> <output_dir>/training_config.yaml), so
    inference conditions on the same scalars/ranges the DiT was trained with
    (gamma for euler_mq, mach for srba). `fallback` (the inference config's own
    block) is used when there is no checkpoint or no dumped config."""
    if checkpoint is not None:
        path = Path(checkpoint).parent.parent / "training_config.yaml"
        if path.is_file():
            block = load_training_config(path).get("scalar_conditioning")
            if block:
                return block
    return fallback


def load_channel_normalization(source: str | Path) -> dict:
    """Read the channel normalization stats that a VAE / latent set was built with.

    Accepts either a ``normalization.json`` written by ``preprocess_dataset.py``
    or a VAE run's ``training_config.yaml``. Encoding, decoding and preprocessing
    must all agree on these numbers -- the VAE only ever saw fields normalized as
    ``(x - mean) / (std * clip)`` clamped to [-1, 1], so any mismatch silently
    produces off-distribution latents or wrongly scaled physical outputs.

    Returns a dict with ``channel_names``, ``channel_mean``, ``channel_std``,
    ``normalization_clip`` and ``dataset``. ``channel_names`` (the stacked field
    order) is the json's ``channel_names`` or the config's ``channel_order``,
    falling back to the 4-channel CHANNEL_NAMES for sources that predate
    recording it; ``dataset`` falls back to "euler_mq" the same way.
    """
    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(f"Normalization source not found: {path}")

    if path.suffix == ".json":
        payload = json.loads(path.read_text())
    else:
        payload = load_training_config(path).get("data", {})

    mean, std = payload.get("channel_mean"), payload.get("channel_std")
    if not mean or not std:
        raise ValueError(f"{path} has no channel_mean / channel_std")
    names = list(payload.get("channel_names") or payload.get("channel_order") or CHANNEL_NAMES)
    if len(names) != len(mean) or len(std) != len(mean):
        raise ValueError(f"{path}: {len(names)} channel names vs {len(mean)} means / {len(std)} stds")
    return {
        "channel_names": names,
        "channel_mean": list(mean),
        "channel_std": list(std),
        "normalization_clip": float(payload.get("normalization_clip", 5.0)),
        "dataset": payload.get("dataset") or DEFAULT_DATASET,
    }


def normalize_fields(
    x: Tensor,
    channel_mean: list[float],
    channel_std: list[float],
    normalization_clip: float,
) -> Tensor:
    """Normalize physical fields [B, C, ...] to [-1, 1]; inverse of denormalize_fields."""
    shape = (1, len(channel_mean)) + (1,) * (x.ndim - 2)
    mean = x.new_tensor(channel_mean).view(shape)
    scale = x.new_tensor(channel_std).view(shape) * normalization_clip
    return ((x - mean) / scale).clamp(-1.0, 1.0)


def extract_scalars(
    meta: dict,
    scalar_names: list[str] | None = None,
) -> dict[str, float]:
    """Extract scalar conditioning values from a bubble sample's metadata.

    Args:
        meta: metadata dictionary.
        scalar_names: names to extract. Defaults to ["gamma"].

    Returns:
        Dictionary mapping scalar names to float values.
    """
    if scalar_names is None:
        scalar_names = ["gamma"]

    scalars = {}

    for name in scalar_names:
        scalars[name] = float(meta[name])

    return scalars

def pad_frames_8n1(x: Tensor) -> Tensor:
    """Pad temporal dimension so F = 8n + 1 (required by LTX-Video VAE).

    Pads by repeating the last frame.

    Args:
        x: [B, C, F, H, W] tensor.

    Returns:
        Padded tensor with F' = 8n + 1 >= F.
    """
    orig_F = x.shape[2]
    target = ((orig_F - 1 + 7) // 8) * 8 + 1
    if target == orig_F:
        return x
    pad_frames = target - orig_F
    last = x[:, :, -1:]
    return torch.cat([x, last.expand(-1, -1, pad_frames, -1, -1)], dim=2)
