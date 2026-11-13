#!/usr/bin/env python
"""Decode DiT rollout latents to fields with a VAE run's decoders: the AR corrector's input frames.

    torchrun --standalone --nproc_per_node=8 scripts/wan/decode_rollouts.py \
        --rollouts $WINDINET_WORK/latents/rollouts/stage3_d4_steps2_val500 \
        --vae results/wan/eulermq/vae/res256/pairs/2dec_clip33 --out $WINDINET_WORK/fields/stage3_d4_steps2_val500

Writes <out>/<split>/<sim id>.pt = {"fields": [F, 4, H, W] fp16 in physical units, and the condition} for every file of
<rollouts>/<split>/ and <out>/decode.yaml (what produced them). Sharded over torchrun ranks; existing files are
skipped, so a killed job resumes.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
import typer
import yaml

from windinet.experiment.distributed import rank, setup, shard, world
from windinet.experiment.runs import provenance
from windinet.field_adapter import load_adapter
from windinet.wan.dit.latents import encode_info
from windinet.wan.vae.model import WanVAE
from windinet.wan.vae.roundtrip import decode
from windinet.wan.weights import load_weights


@torch.no_grad()
def main(
    rollouts: Path = typer.Option(..., help="rollout latent folder (train_dit.py's `rollouts`)"),
    vae: Path = typer.Option(..., help="VAE-stage run folder whose adapter and decoders decode"),
    out: Path = typer.Option(..., help="field folder"),
    sims: int | None = typer.Option(None, help="first n sims per split (smoke)"),
) -> None:
    started = time.time()
    setup()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adapter = load_adapter(vae, device)
    if encode_info(rollouts)["groups"] != adapter.groups:
        raise typer.BadParameter(f"{rollouts} was encoded with other groups than {vae}")
    per_group = yaml.safe_load((vae / "config.yaml").read_text())["train"]["vae_per_group"]
    vaes = [WanVAE(device=device, channels=adapter.channels) for _ in range(len(adapter.groups) if per_group else 1)]
    load_weights(vae, adapter, vaes, device)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        record = {"rollouts": str(rollouts), "vae": str(vae), "sims": sims, "provenance": provenance(started, world)}
        (out / "decode.yaml").write_text(yaml.safe_dump(record))
    for split in ("train", "val", "test"):
        (out / split).mkdir(parents=True, exist_ok=True)
        files = sorted((rollouts / split).glob("*.pt"))[:sims]
        todo = [f for f in shard(files) if not (out / split / f.name).exists()]
        t0 = time.time()
        for i, f in enumerate(todo, 1):
            item = torch.load(f)
            fields = decode(adapter, item["latents"][None].to(device), vaes)[0]  # [F, C, H, W]
            tmp = out / split / f"{f.stem}.tmp"  # renamed when complete: a killed job leaves no broken .pt
            torch.save({"fields": fields.half().cpu(), **{k: v for k, v in item.items() if k != "latents"}}, tmp)
            tmp.rename(tmp.with_suffix(".pt"))
            if i % 50 == 0 or i == len(todo):
                per_sim, left = (time.time() - t0) / i, len(todo) - i
                print(f"{split}: {i}/{len(todo)} sims per rank, {per_sim:.1f} s/sim, eta {per_sim * left / 60:.0f} min")
    print(f"wrote {out}/ ({world} x {device}, {(time.time() - started) / 60:.0f} min)")


if __name__ == "__main__":
    typer.run(main)
