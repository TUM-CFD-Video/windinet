#!/usr/bin/env python
"""Encode every simulation with a fixed adapter and the frozen Wan encoder: the DiT stage's training data.

    torchrun --standalone --nproc_per_node=8 scripts/wan/encode_latents.py configs/wan/vae.yaml \
        --load results/wan/eulermq/vae/res256/rgb3/adapter --out $WINDINET_WORK/latents/rgb3

Takes the VAE-stage YAML for the data files and the split, the adapter from `--load`. Writes
<out>/<train|val|test>/<sim id>.pt = {"latents": [groups, 16, F/4 + 1, H/8, W/8] fp32 normalised Wan latents
(posterior mean, the whole trajectory as one clip), "gamma"} and <out>/encode.yaml (what produced them). With
data.augment every train sim is also encoded under the dataset's other symmetries (<sim id>_s<k>.pt), with
--all-starts also from every chunk boundary of its first half (<sim id>_t<frame>.pt).
Sharded over torchrun ranks; existing files are skipped, so a killed job resumes.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
import typer
import yaml
from torch.utils.data import DataLoader

from windinet.data.frames import SYMMETRIES, FrameSampler, symmetry
from windinet.experiment.config import load_config
from windinet.experiment.distributed import rank, setup, shard, world
from windinet.experiment.runs import provenance
from windinet.field_adapter import load_adapter
from windinet.wan.dit.config import EncodeConfig
from windinet.wan.dit.latents import latent_id
from windinet.wan.vae.model import WanVAE
from windinet.wan.vae.roundtrip import encode


def latent_files(out_dir: Path, split: str, sid: str, augment: bool, starts: list[int]) -> dict[Path, tuple[int, int]]:
    """The files of one simulation, path -> (symmetry k, first frame): its latents and, in the train split, the
    symmetric copies (`augment`) and the clips from the later `starts`."""
    train = split == "train"
    copies = range(SYMMETRIES if augment and train else 1)
    return {out_dir / split / f"{latent_id(sid, k, t)}.pt": (k, t) for k in copies for t in (starts if train else [0])}


@torch.no_grad()
def encode_sim(adapter, vae: WanVAE, frames: torch.Tensor, gamma: float, files: dict, device) -> None:
    """Writes the missing files of one simulation: `frames` [F, 4, H, W] from frame t on, under symmetry k, one clip."""
    for path, (k, t) in files.items():
        if path.exists():
            continue
        x = adapter.select(symmetry(frames[t:], k).to(device))  # [F - t, C, H, W]
        z, _ = encode(adapter, x, [vae], len(x))  # [1, G, 16, F', h, w]
        torch.save({"latents": z[0].cpu(), "gamma": gamma}, path)


def main(
    config_path: Path = typer.Argument(..., help="VAE-stage YAML: data files and split"),
    load: str = typer.Option(..., help="VAE-stage run folder with adapter.pt"),
    out: str = typer.Option(..., help="latent folder"),
    sims: int | None = typer.Option(None, help="first n sims per split (smoke)"),
    all_starts: bool = typer.Option(False, help="also every train sim from each chunk boundary of its first half"),
    set_: list[str] = typer.Option([], "--set", help="dotted override of the YAML, e.g. data.augment=true"),
) -> None:
    started = time.time()
    data = load_config(config_path, set_)["data"]
    cfg = EncodeConfig(data=data, load=load, out=out, sims=sims, all_starts=all_starts)
    run, out_dir = Path(cfg.load), Path(cfg.out)
    setup()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adapter = load_adapter(run, device)
    vae = WanVAE(device=device, channels=adapter.channels)
    sets = {k: (h5, ids[: cfg.sims]) for k, (h5, ids) in cfg.data.splits().items()}
    h5, ids = sets["train"]
    n_frames = len(FrameSampler(h5, ids[:1], None, 0, stride=cfg.data.frame_stride)[0][0])  # 4k + 1 per trajectory
    starts = list(range(0, n_frames // 2 + 1, 4)) if cfg.all_starts else [0]
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        record = {"config": cfg.model_dump(), "groups": adapter.groups, "starts": starts}
        record["provenance"] = provenance(started, world)
        (out_dir / "encode.yaml").write_text(yaml.safe_dump(record))
    for split, (h5, ids) in sets.items():
        (out_dir / split).mkdir(parents=True, exist_ok=True)  # every rank: rank 0 may not have made out_dir yet
        files = {sid: latent_files(out_dir, split, sid, cfg.data.augment, starts) for sid in shard(ids)}
        todo = [sid for sid, paths in files.items() if not all(p.exists() for p in paths)]
        loader = DataLoader(FrameSampler(h5, todo, None, 0, stride=cfg.data.frame_stride), num_workers=2)
        t0 = time.time()
        for i, (sid, (frames, gamma)) in enumerate(zip(todo, loader, strict=True), 1):
            encode_sim(adapter, vae, frames[0], float(gamma), files[sid], device)
            if i % 50 == 0 or i == len(todo):
                per_sim, left = (time.time() - t0) / i, len(todo) - i
                print(f"{split}: {i}/{len(todo)} sims per rank, {per_sim:.1f} s/sim, eta {per_sim * left / 60:.0f} min")
    print(f"wrote {out_dir}/ ({world} x {device}, {(time.time() - started) / 60:.0f} min)")


if __name__ == "__main__":
    typer.run(main)
