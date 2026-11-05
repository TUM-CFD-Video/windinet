#!/usr/bin/env python
"""One video from the Wan 2.1 DiT: zero-shot text-to-video, or a simulation taken partly back to noise and
denoised again (SDEdit; strength 0 is the plain VAE round trip of the simulation).

    python scripts/wan/sample_dit.py configs/wan/sample.yaml --name zs_native --desc "..." [--set height=256 ...]

Writes <results_dir>/<name>/{config.yaml, provenance.json, strip.png, <name>.mp4} (the mp4 in `video_dir` when
set, symlinked). Variants via --set, see jobs/lc/dit_zero_shot.txt.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import typer

from windinet.eulermq.data import FrameSampler
from windinet.experiment.config import start_run
from windinet.experiment.runs import provenance
from windinet.field_adapter import load_adapter
from windinet.wan.dit.conditioning import TextTokens
from windinet.wan.dit.config import SampleConfig
from windinet.wan.dit.model import WanDiT
from windinet.wan.figures import save_strip, save_video
from windinet.wan.vae.model import WanVAE


@torch.no_grad()
def main(
    config_path: Path = typer.Argument(..., help="YAML matching SampleConfig"),
    name: str | None = typer.Option(None, help="run name = results folder (overrides the YAML's)"),
    desc: str | None = typer.Option(None, help="one sentence: what this sample shows"),
    set_: list[str] = typer.Option([], "--set", help="dotted override, e.g. height=256"),
    overwrite: bool = typer.Option(False, help="replace an existing run folder of the same name"),
) -> None:
    started = time.time()
    cfg, run_dir, device = start_run(SampleConfig, config_path, name, desc, set_, overwrite)

    cond = TextTokens(cfg.prompt, cfg.negative_prompt, device).tokens  # [2, 512, 4096]: prompt, negative
    vae = WanVAE(device=device)
    init = None
    if cfg.source:  # the simulation as the adapter's first image, through the encoder
        h5, sid = cfg.source.rsplit(":", 1)
        adapter = load_adapter(cfg.adapter, device)
        frames, _ = FrameSampler(h5, [sid], None, 0, consecutive=True)[0]
        init = vae.encode(adapter(adapter.select(frames.to(device)))[0].movedim(0, 1)[None])
    shape = init.shape if init is not None else (1, 16, (cfg.frames - 1) // 4 + 1, cfg.height // 8, cfg.width // 8)
    dit = WanDiT(device, load=Path(cfg.load) / "dit.pt" if cfg.load else None)
    latents = dit.sample(
        cond[:1], cond[1:], shape, cfg.steps, cfg.guidance, cfg.seed, init, cfg.strength, int(cfg.keep_first_frame)
    )
    video = vae.decode(latents)[0]  # [3, F, H, W] in (-1, 1)

    video_path = (Path(cfg.video_dir) if cfg.video_dir else run_dir) / f"{cfg.name}.mp4"
    video_path.parent.mkdir(parents=True, exist_ok=True)
    save_video(video, video_path, cfg.fps)
    if cfg.video_dir:
        (run_dir / video_path.name).unlink(missing_ok=True)
        (run_dir / video_path.name).symlink_to(video_path)
    save_strip(video, run_dir / "strip.png", vae.decode(init)[0] if init is not None else None)
    (run_dir / "provenance.json").write_text(json.dumps(provenance(started), indent=1))
    print(f"wrote {run_dir}/ and {video_path} ({tuple(video.shape)}, {(time.time() - started) / 60:.0f} min)")


if __name__ == "__main__":
    typer.run(main)
