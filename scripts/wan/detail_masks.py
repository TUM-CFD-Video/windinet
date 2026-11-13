"""Detail shares per latent cell (windinet.wan.dit.detail) of every simulation in an h5 file: <out>/<sim id>.pt
[F', h, w] float16, for train.detail_masks. Under torchrun the sims are split over ranks. euler_mq only: the
criteria need its density, momentum and pressure."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import torch
import typer

from windinet.data.specs import EULERMQ
from windinet.experiment.distributed import rank, shard
from windinet.wan.dit.detail import detail_share


def main(
    h5: Path = typer.Option(..., help="the simulations"),
    out: Path = typer.Option(..., help="output folder"),
    stride: int = typer.Option(2, help="frame stride of the latents"),
    frames: int = typer.Option(97, help="first n frames, as encoded (1 + 4k latent frames after the stride)"),
    threshold: float = typer.Option(0.2, help="Loehner threshold on density and pressure"),
    q_threshold: float = typer.Option(1.0, help="threshold on the normalised Q-criterion"),
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    device = f"cuda:{rank % torch.cuda.device_count()}" if torch.cuda.is_available() else "cpu"
    settings = dict(h5=str(h5), stride=stride, frames=frames, threshold=threshold, q_threshold=q_threshold)
    if rank == 0:
        (out / "masks.json").write_text(json.dumps(settings))
    with h5py.File(h5, "r") as f:
        for sid in shard(sorted(f)):
            x = torch.from_numpy(np.stack([f[sid][k][:frames:stride, 0] for k in EULERMQ.fields], 1)).to(device)
            torch.save(detail_share(x, threshold, q_threshold).half().cpu(), out / f"{sid}.pt")


if __name__ == "__main__":
    typer.run(main)
