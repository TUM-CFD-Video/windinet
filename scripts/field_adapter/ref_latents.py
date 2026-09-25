#!/usr/bin/env python
"""Encode the reference clips with the frozen Wan VAE and store what "natural" looks like.

Output <out>: {"rgb": {mean, var, cov}, "latent": {mean, var, cov}, "n_clips", "frames"}
  rgb    -- pixel statistics of the clips in (-1, 1)          (C = 3)
  latent -- statistics of the normalised Wan latents          (C = 16)
The tensors are a few KB; the latents themselves are not kept.

    python scripts/field_adapter/ref_latents.py --clips ref_clips/clips.pt \
        --out results/field_adapter/30_reference/wan_ref_stats.pt
"""

from __future__ import annotations

from pathlib import Path

import torch
import typer

from windinet.field_adapter.latent_stats import channel_stats
from windinet.wan.vae import WanVAE


@torch.no_grad()
def main(
    clips: Path = typer.Option(Path("ref_clips/clips.pt")),
    out: Path = typer.Option(Path("results/field_adapter/30_reference/wan_ref_stats.pt")),
    batch: int = typer.Option(4),
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = WanVAE(device=device)
    video = torch.load(clips)  # uint8 [N, F, H, W, 3]
    rgb = video.permute(0, 4, 1, 2, 3).float() / 127.5 - 1  # [N, 3, F, H, W] in (-1, 1)
    latents = torch.cat([vae.encode(rgb[i : i + batch].to(device)).cpu() for i in range(0, len(rgb), batch)])
    stats = {
        "rgb": channel_stats(rgb[:, :, :, ::4, ::4]),
        "latent": channel_stats(latents),
        "n_clips": len(rgb),
        "frames": rgb.shape[2],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(stats, out)
    lat = stats["latent"]
    print(f"{len(rgb)} clips -> latents {tuple(latents.shape[1:])}; per-channel mean in "
          f"[{lat['mean'].min():.2f}, {lat['mean'].max():.2f}], var in [{lat['var'].min():.2f}, {lat['var'].max():.2f}] "
          f"(expect ~0 / ~1 if latents_mean/std are right)")


if __name__ == "__main__":
    typer.run(main)
