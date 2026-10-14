#!/usr/bin/env python
"""Encode every simulation with a fixed adapter and the frozen Wan encoder: the DiT stage's training data.

    torchrun --standalone --nproc_per_node=8 scripts/wan/encode_latents.py configs/wan/vae.yaml \
        --load results/wan/vae/rgb3_adapter --out $WINDINET_WORK/latents/rgb3

Takes the VAE-stage YAML for the data files and the split, the adapter from `--load`. Writes
<out>/<train|val|test>/<sim id>.pt = {"latents": [groups, 16, F/4 + 1, H/8, W/8] fp32 normalised Wan latents
(posterior mean, the whole trajectory as one clip), "gamma"} and <out>/encode.yaml (what produced them).
Sharded over torchrun ranks; existing files are skipped, so a killed job resumes.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
import typer
import yaml
from torch.utils.data import DataLoader

from windinet.eulermq.data import FrameSampler
from windinet.experiment.config import load_config
from windinet.experiment.distributed import rank, setup, shard, world
from windinet.experiment.runs import provenance
from windinet.field_adapter import load_adapter
from windinet.wan.dit.config import EncodeConfig
from windinet.wan.vae.model import WanVAE


@torch.no_grad()
def main(
    config_path: Path = typer.Argument(..., help="VAE-stage YAML: data files and split"),
    load: str = typer.Option(..., help="VAE-stage run folder with adapter.pt"),
    out: str = typer.Option(..., help="latent folder"),
    sims: int | None = typer.Option(None, help="first n sims per split (smoke)"),
) -> None:
    started = time.time()
    cfg = EncodeConfig(data=load_config(config_path, [])["data"], load=load, out=out, sims=sims)
    run, out_dir = Path(cfg.load), Path(cfg.out)
    setup()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adapter = load_adapter(run, device)
    vae = WanVAE(device=device, channels=adapter.channels)
    sets = {k: (h5, ids[: cfg.sims]) for k, (h5, ids) in cfg.data.splits().items()}
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        record = {"config": cfg.model_dump(), "groups": adapter.groups, "provenance": provenance(started, world)}
        (out_dir / "encode.yaml").write_text(yaml.safe_dump(record))
    for split, (h5, ids) in sets.items():
        (out_dir / split).mkdir(parents=True, exist_ok=True)  # every rank: rank 0 may not have made out_dir yet
        todo = [sid for sid in shard(ids) if not (out_dir / split / f"{sid}.pt").exists()]
        loader = DataLoader(FrameSampler(h5, todo, None, 0, consecutive=True), num_workers=2)  # whole trajectories
        t0 = time.time()
        for i, (sid, (frames, gamma)) in enumerate(zip(todo, loader, strict=True), 1):
            x = adapter.select(frames[0].to(device))  # [F, C, H, W]
            z = torch.stack([vae.encode(rgb.movedim(0, 1)[None])[0] for rgb in adapter(x)])  # [G, 16, F', h, w]
            torch.save({"latents": z.cpu(), "gamma": float(gamma)}, out_dir / split / f"{sid}.pt")
            if i % 50 == 0 or i == len(todo):
                per_sim, left = (time.time() - t0) / i, len(todo) - i
                print(f"{split}: {i}/{len(todo)} sims per rank, {per_sim:.1f} s/sim, eta {per_sim * left / 60:.0f} min")
    print(f"wrote {out_dir}/ ({world} x {device}, {(time.time() - started) / 60:.0f} min)")


if __name__ == "__main__":
    typer.run(main)
