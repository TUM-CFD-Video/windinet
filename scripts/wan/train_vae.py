#!/usr/bin/env python
"""VAE stage: train the field -> RGB adapter and, optionally, the Wan VAE behind it; measure the reconstruction.

    vae: wan    x -> A -> Wan VAE -> A^-1 -> x_hat     (the real round trip)
    vae: none   x -> A -> A^-1 -> x_hat                (adapter alone, sanity check)

train.steps=0 evaluates an init (or adapter.load=<.pt>) without training. With ref_stats set, the latent
statistics of each group are compared to the natural-video reference (Fréchet distance).

Usage:
    python scripts/wan/train_vae.py configs/wan/vae.yaml --name pairs_joint --desc "..." --set train.steps=1000
Every run needs a name and a one-sentence description. Writes
<results_dir>/<name>/{config.yaml, metrics.json, panel.png, curves.png, adapter.pt[, vae.pt]} and regenerates
the results_dir README.md index. config.yaml is the resolved config and re-runs as is; metrics.json carries the
provenance (commit, command, versions, GPU, wall time).

Evaluation: mid-run on data.eval_sims validation sims with the training clip length ("val"); at the end on every
test simulation's whole trajectory ("final"), the metric of the LTX baseline (thesis eq. 5.11).
"""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # avoids fragmentation OOM on 11 GB

import torch
import typer
from torch.utils.data import DataLoader

from windinet.eulermq.data import FrameSampler, load_field_stats, split_ids, val_split
from windinet.experiment.config import load_config
from windinet.experiment.figures import save_curves, save_panel
from windinet.experiment.runs import SYMBOLS, provenance, write_config, write_index
from windinet.experiment.schedule import warmup_cosine
from windinet.field_adapter import GroupedAdapter
from windinet.training.shockwave_data import ShockWaveDataset
from windinet.wan.roundtrip import clip_len, evaluate, losses
from windinet.wan.vae_stage import VaeStageConfig


def rel_change(params, ref) -> float:
    """||p - ref|| / ||ref|| over a list of tensors: how far weights moved. Computed where `ref` lives."""
    num = sum((p.detach().to(r.device) - r).square().sum() for p, r in zip(params, ref, strict=True))
    return (num / sum(r.square().sum() for r in ref)).sqrt().item()


def drift_per_block(vaes: list, w0) -> dict[str, float]:
    """Relative weight drift from the pretrained VAE, grouped by block (decoder.up_blocks.2, ...; A/B per group)."""
    blocks: dict[str, list] = {}
    w0 = iter(w0)
    for i, vae in enumerate(vaes):
        prefix = f"{'ABC'[i]}: " if len(vaes) > 1 else ""
        for name, p in vae.vae.named_parameters():
            if p.requires_grad:
                blocks.setdefault(prefix + ".".join(name.split(".")[:3]), []).append((p, next(w0)))
    return {b: rel_change([p for p, _ in prs], [r for _, r in prs]) for b, prs in blocks.items()}


def summary(metrics: dict[str, float], fields: list[str]) -> tuple[str, str]:
    """Two log lines: the means (plus Fréchet and drift when present), then the per-field VRMSE in physical units."""
    means = f"mean {metrics['vrmse_mean']:.4f}  clipped {metrics['vrmse_clipped_mean']:.4f}"
    means += "".join(f"  Fréchet {k[6]} {v:.3g}" for k, v in metrics.items() if k.endswith("_frechet"))
    if "vae_drift" in metrics:
        means += f"  drift {metrics['vae_drift']:.3g}"
    per_field = "  ".join(f"{SYMBOLS[f]} {metrics['vrmse_' + f]:.4f}" for f in fields)
    return means, per_field


def save_weights(run_dir: Path, adapter: GroupedAdapter, vaes: list, weights_dir: str | None) -> None:
    """adapter.pt (in git) and, for VAE fine-tunes, the trained VAE weights as vae.pt, or vae_A.pt, vae_B.pt with
    one VAE per group (gitignored; train.vae_load). With `weights_dir` (scratch on a cluster) the VAE files live
    there and the run folder holds symlinks to them."""
    torch.save(adapter.state_dict(), run_dir / "adapter.pt")
    for i, vae in enumerate(vaes):
        if not vae.trainable_params:
            continue
        name = f"vae_{'ABC'[i]}.pt" if len(vaes) > 1 else "vae.pt"
        path = Path(weights_dir) / run_dir.name / name if weights_dir else run_dir / name
        if weights_dir:
            path.parent.mkdir(parents=True, exist_ok=True)
            (run_dir / name).unlink(missing_ok=True)
            (run_dir / name).symlink_to(path)
        torch.save(
            {k: v.detach().cpu() for k, v in vae.vae.state_dict(keep_vars=True).items() if v.requires_grad}, path
        )


def hours(seconds: float) -> str:
    return f"{seconds / 3600:.1f} h" if seconds >= 3600 else f"{seconds / 60:.0f} min"


def main(
    config_path: Path = typer.Argument(..., help="YAML matching VaeStageConfig"),
    name: str | None = typer.Option(None, help="run name = results folder (overrides the YAML's)"),
    desc: str | None = typer.Option(None, help="one sentence: what this run tests (overrides the YAML's description)"),
    set_: list[str] = typer.Option([], "--set", help="dotted override, e.g. train.steps=0"),
    overwrite: bool = typer.Option(False, help="replace an existing run folder of the same name"),
) -> None:
    started = time.time()
    raw = load_config(config_path, set_)
    cfg = VaeStageConfig(**(raw | {k: v for k, v in {"name": name, "description": desc}.items() if v}))
    out_dir = Path(cfg.results_dir)
    run_dir = out_dir / cfg.name
    if (run_dir / "metrics.json").exists() and not overwrite:
        raise typer.BadParameter(f"{run_dir} exists; pick another --name or pass --overwrite")
    torch.manual_seed(cfg.train.seed)
    random.seed(cfg.train.seed)  # frame sampling in the main process; loader workers derive theirs from torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir.mkdir(parents=True, exist_ok=True)
    write_config(cfg.model_dump(), run_dir / "config.yaml")  # written first: a crashed run still shows what it was
    print((run_dir / "config.yaml").read_text())  # the resolved settings, the same text as the file

    ids = ShockWaveDataset(cfg.data.h5_path).ids
    if cfg.data.test_h5:
        train_ids, val_ids = val_split(ids, cfg.data.val_sims, cfg.data.split_seed)
        test_h5, test_ids = cfg.data.test_h5, ShockWaveDataset(cfg.data.test_h5).ids
    else:
        train_ids, val_ids = split_ids(ids, cfg.data.test_every, cfg.data.test_gamma)
        test_h5, test_ids = cfg.data.h5_path, val_ids
    test_ids = test_ids[: cfg.data.test_sims]
    clip = clip_len(cfg)

    def loader(h5, sims, frames, seed, consecutive):  # seed None: random frames every access (training)
        sampler = FrameSampler(h5, sims, frames, seed, consecutive)
        return DataLoader(
            sampler,
            batch_size=cfg.train.batch_sims if seed is None else 1,
            shuffle=seed is None,
            num_workers=cfg.data.num_workers,
            persistent_workers=cfg.data.num_workers > 0,
        )

    train_loader = loader(cfg.data.h5_path, train_ids, cfg.data.frames_per_sim, None, cfg.data.clip)
    val_loader = loader(cfg.data.h5_path, val_ids[: cfg.data.eval_sims], cfg.data.frames_per_sim, 0, cfg.data.clip)
    test_loader = loader(test_h5, test_ids, None, 0, True)  # whole trajectories, in order

    def batches():  # endless; unlike itertools.cycle it keeps no copy of the batches it yielded
        while True:
            yield from train_loader

    batches = batches()
    print(
        f"{len(train_ids)} train / {len(val_ids)} val ({len(val_loader)} evaluated mid-run) / {len(test_ids)} test sims"
        f" (whole trajectories at the end), device {device}"
    )

    stats = load_field_stats(cfg.data.stats_json)
    adapter = GroupedAdapter(cfg.adapter.groups, stats).to(device)
    norm = tuple(torch.tensor([stats[n][k] for n in adapter.fields], device=device) for k in ("mean", "std"))
    adapter.requires_grad_(cfg.train.lr > 0)  # lr 0 = fixed adapter, no gradient through the encoder for it
    if cfg.adapter.load:
        adapter.load_state_dict(torch.load(cfg.adapter.load, map_location=device))
    else:  # tanh scales from 100 fixed sims x 2 frames: large enough that the init does not depend on the draw
        probe = FrameSampler(cfg.data.h5_path, train_ids[:: max(1, len(train_ids) // 100)][:100], 2, seed=0)
        adapter.init_from_data(adapter.select(torch.cat([probe[i][0] for i in range(len(probe))])).to(device))
    vaes = []
    if cfg.vae == "wan":
        from windinet.wan.vae import WanVAE

        n_vaes = len(cfg.adapter.groups) if cfg.train.vae_per_group else 1
        vaes = [WanVAE(device=device, train=cfg.train.vae_parts) for _ in range(n_vaes)]
        if cfg.train.vae_load:  # the same start weights for every VAE
            for vae in vaes:
                vae.vae.load_state_dict(torch.load(cfg.train.vae_load, map_location=device), strict=False)
    ref = torch.load(cfg.ref_stats) if cfg.ref_stats else None
    init, example = evaluate(adapter, val_loader, vaes, ref, norm, clip)
    means, per_field = summary(init, adapter.fields)
    print(f"init | {means}\n     | {per_field}")

    vae_params = [p for vae in vaes for p in vae.trainable_params]
    groups = [{"params": adapter.parameters(), "lr": cfg.train.lr}]
    if vae_params:
        groups.append({"params": vae_params, "lr": cfg.train.vae_lr})
    opt = torch.optim.AdamW(groups, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, warmup_cosine(cfg.train.warmup_steps, cfg.train.steps, cfg.train.lr_floor)
    )
    params = [p for g in opt.param_groups for p in g["params"]]
    w0 = [p.detach().cpu() for p in vae_params]  # start weights, for the drift; on the CPU, read at evals only
    history, steps, t0 = [], [], time.time()  # per eval / per update
    metrics = init  # last validation; stays the init for an eval-only run (steps 0)
    t_log = t0
    chunk = clip * cfg.train.micro_batch  # frames per backward: ~1.5 GB each through the VAE
    for step in range(1, cfg.train.steps + 1):
        x = adapter.select(next(batches)[0].flatten(0, 1).to(device))
        opt.zero_grad(set_to_none=True)
        loss = 0.0
        for xi in x.split(chunk if vaes else len(x)):
            terms, total = losses(cfg, adapter, xi, vaes)
            weight = len(xi) / len(x)
            (total * weight).backward()
            loss += total.item() * weight
        if cfg.train.max_grad_norm:
            torch.nn.utils.clip_grad_norm_(params, cfg.train.max_grad_norm)
        before = [[p.detach().clone() for p in g["params"]] for g in opt.param_groups]  # on the device: no sync
        opt.step()
        sched.step()
        steps.append(
            {
                "step": step,
                "loss": loss,
                "update_ratio": [rel_change(g["params"], b) for g, b in zip(opt.param_groups, before, strict=True)],
            }
        )
        if step % cfg.train.log_every == 0:
            recent = steps[-cfg.train.log_every :]
            lrs = " ".join(f"{g['lr']:.1e}" for g in opt.param_groups)
            updates = " ".join(f"{u:.1e}" for u in recent[-1]["update_ratio"])
            now, per_step = time.time(), (time.time() - t_log) / len(recent)  # recent pace, not the warm-up
            t_log = now
            print(
                f"step {step}/{cfg.train.steps} ({100 * step / cfg.train.steps:3.0f} %) | "
                f"epoch {step * cfg.train.batch_sims / len(train_ids):.2f} | "
                f"loss {sum(s['loss'] for s in recent) / len(recent):.4f} | lr {lrs} | upd {updates} | "
                f"eta {hours(per_step * (cfg.train.steps - step))}"
                + (f" | peak {torch.cuda.max_memory_allocated() / 2**30:.0f} GB" if device.type == "cuda" else "")
            )
        if step % cfg.train.eval_every == 0 or step == cfg.train.steps:
            metrics, example = evaluate(adapter, val_loader, vaes, ref, norm, clip)
            if w0:
                metrics["vae_drift"] = rel_change(vae_params, w0)
            history.append({"step": step, "train": {k: v.item() for k, v in terms.items()}, **metrics})
            means, per_field = summary(metrics, adapter.fields)
            print(f"val {step} | loss {loss:.4f} | {means} | {hours(time.time() - t0)}\n    | {per_field}")
            save_weights(run_dir, adapter, vaes, cfg.weights_dir)  # a killed run keeps its last evaluated weights

    final, example = evaluate(adapter, test_loader, vaes, ref, norm, None)
    means, per_field = summary(final, adapter.fields)
    print(f"test {len(test_ids)} whole trajectories | {means} | {hours(time.time() - t0)}\n     | {per_field}")
    save_panel(example, adapter.fields, run_dir / "panel.png")
    save_curves(steps, history, drift_per_block(vaes, w0) if w0 else {}, adapter.fields, run_dir / "curves.png")
    save_weights(run_dir, adapter, vaes, cfg.weights_dir)
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "config": cfg.model_dump(),
                "provenance": provenance(started),
                "init": init,
                "val": metrics,
                "final": final,
                "history": history,
                "steps": steps,
                "n_params": sum(p.numel() for p in adapter.parameters()),
            },
            indent=1,
        )
    )
    write_index(out_dir)
    print(f"wrote {run_dir}/ and {out_dir / 'README.md'}")


if __name__ == "__main__":
    typer.run(main)
