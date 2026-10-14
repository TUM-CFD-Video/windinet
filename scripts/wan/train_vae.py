#!/usr/bin/env python
"""VAE stage: train the field -> RGB adapter and, optionally, the Wan VAE behind it; measure the reconstruction.

    vae: wan    x -> A -> Wan VAE -> A^-1 -> x_hat     (the real round trip)
    vae: none   x -> A -> A^-1 -> x_hat                (adapter alone, sanity check)

train.steps=0 evaluates an init (or load=<run folder>) without training.

Usage:
    python scripts/wan/train_vae.py configs/wan/vae.yaml --name pairs_joint --desc "..." --set train.steps=1000
    torchrun --standalone --nproc_per_node=8 scripts/wan/train_vae.py ...   # one process per GPU, same config:
    train.batch_sims is the global batch, split over the ranks; evaluation is sharded; rank 0 logs and writes.
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
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # less fragmentation when memory is tight

import torch
import typer
from torch.utils.data import DataLoader

from windinet.eulermq.data import FrameSampler, endless, load_field_stats
from windinet.experiment.config import start_run
from windinet.experiment.distributed import all_mean, rank, shard, sync, world
from windinet.experiment.log import hours, progress, report
from windinet.experiment.metrics import rel_change
from windinet.experiment.optim import adamw
from windinet.experiment.runs import provenance, write_index
from windinet.field_adapter import GroupedAdapter, group_tag
from windinet.wan.figures import save_curves, save_panel
from windinet.wan.vae.config import TrainConfig, VaeStageConfig
from windinet.wan.vae.roundtrip import evaluate, losses
from windinet.wan.weights import load_weights, save_weights


def loader(cfg: VaeStageConfig, h5: str, sims: list[str], frames: int | None, train: bool) -> DataLoader:
    """Training: random frames per access, this rank's share of the global batch. Evaluation: fixed frames, one sim
    per item; `frames` None = whole trajectories."""
    sampler = FrameSampler(h5, shard(sims), frames, None if train else 0, consecutive=cfg.data.clip or frames is None)
    batch = max(1, cfg.train.batch_sims // world) if train else 1  # max: an eval-only run never draws a batch
    return DataLoader(
        sampler,
        batch,
        shuffle=train,
        num_workers=cfg.data.num_workers,
        persistent_workers=cfg.data.num_workers > 0,
    )


def build(cfg: VaeStageConfig, train_ids: list[str], device: torch.device) -> tuple[GroupedAdapter, list, tuple]:
    """Adapter and VAEs with their weights (from `cfg.load`, else data-initialised), and the field (mean, std)
    the clipped metric uses."""
    stats = load_field_stats(cfg.data.stats_json)
    adapter = GroupedAdapter(cfg.adapter.groups, stats).to(device)
    adapter.requires_grad_(cfg.train.lr > 0)  # lr 0 = fixed adapter: no gradient through the encoder for it
    vaes = []
    if cfg.vae == "wan":
        from windinet.wan.vae.model import WanVAE  # diffusers only when a VAE is used

        n_vaes = len(cfg.adapter.groups) if cfg.train.vae_per_group else 1
        vaes = [WanVAE(device=device, train=cfg.train.vae_parts, channels=adapter.channels) for _ in range(n_vaes)]
    if cfg.load:
        load_weights(Path(cfg.load), adapter, vaes, device)
    else:  # tanh scales from 100 fixed sims x 2 frames: large enough that the init does not depend on the draw
        probe = FrameSampler(cfg.data.h5_path, train_ids[:: max(1, len(train_ids) // 100)][:100], 2, seed=0)
        adapter.init_from_data(adapter.select(torch.cat([probe[i][0] for i in range(len(probe))])).to(device))
    sync(adapter)  # the init subsamples at random: every rank takes rank 0's
    norm = tuple(torch.tensor([stats[n][k] for n in adapter.fields], device=device) for k in ("mean", "std"))
    return adapter, vaes, norm


def optimizer(cfg: TrainConfig, adapter: GroupedAdapter, vae_params: list) -> tuple:
    """A param group per lr: the adapter and, when fine-tuned, the VAE."""
    groups = [{"params": list(adapter.parameters()), "lr": cfg.lr}]
    if vae_params:
        groups.append({"params": vae_params, "lr": cfg.vae_lr})
    return adamw(groups, cfg.steps, cfg.warmup_steps, cfg.lr_floor)


def train_step(cfg: VaeStageConfig, adapter: GroupedAdapter, vaes: list, x: torch.Tensor, opt, sched, params) -> dict:
    """One update on the frames `x`: micro-batched backward, gradients averaged over ranks, clip, step.
    Returns the loss terms (mean over the micro-batches) and the relative weight change per param group."""
    opt.zero_grad(set_to_none=True)
    chunk = cfg.data.clip_len * cfg.train.micro_batch if vaes else len(x)  # frames per backward
    result: dict = {}
    for xi in x.split(chunk):
        terms = losses(adapter, xi, vaes, cfg.data.clip_len, cfg.loss)
        weight = len(xi) / len(x)
        (terms["loss"] * weight).backward()
        for k, v in terms.items():
            result[k] = result.get(k, 0.0) + v.item() * weight
    all_mean(*(p.grad for p in params if p.grad is not None))
    if cfg.train.max_grad_norm:
        torch.nn.utils.clip_grad_norm_(params, cfg.train.max_grad_norm)
    before = [[p.detach().clone() for p in g["params"]] for g in opt.param_groups]  # on the device: no sync
    opt.step()
    sched.step()
    result["update_ratio"] = [rel_change(g["params"], b) for g, b in zip(opt.param_groups, before, strict=True)]
    return result


def drift_per_block(vaes: list, w0: list) -> dict[str, float]:
    """Relative weight drift from the pretrained VAE per block (decoder.up_blocks.2, ...; A/B per group)."""
    blocks: dict[str, list] = {}
    w0 = iter(w0)
    for i, vae in enumerate(vaes):
        prefix = f"{group_tag(i)}: " if len(vaes) > 1 else ""
        for name, p in vae.vae.named_parameters():
            if p.requires_grad:
                blocks.setdefault(prefix + ".".join(name.split(".")[:3]), []).append((p, next(w0)))
    return {b: rel_change([p for p, _ in prs], [r for _, r in prs]) for b, prs in blocks.items()}


def write_outputs(cfg: VaeStageConfig, run_dir: Path, adapter, vaes, record: dict, drift: dict, example) -> None:
    """Figures, weights, metrics.json and the results index."""
    save_panel(example, adapter.fields, run_dir / "panel.png")
    save_curves(record["steps"], record["history"], drift, adapter.fields, run_dir / "curves.png")
    save_weights(run_dir, adapter, vaes, cfg.weights_dir)
    (run_dir / "metrics.json").write_text(json.dumps(record, indent=1))
    write_index(run_dir.parent)
    print(f"wrote {run_dir}/ and {run_dir.parent / 'README.md'}")


def main(
    config_path: Path = typer.Argument(..., help="YAML matching VaeStageConfig"),
    name: str | None = typer.Option(None, help="run name = results folder (overrides the YAML's)"),
    desc: str | None = typer.Option(None, help="one sentence: what this run tests (overrides the YAML's description)"),
    set_: list[str] = typer.Option([], "--set", help="dotted override, e.g. train.steps=0"),
    overwrite: bool = typer.Option(False, help="replace an existing run folder of the same name"),
) -> None:
    started = time.time()
    cfg, run_dir, device = start_run(VaeStageConfig, config_path, name, desc, set_, overwrite)

    sets = cfg.data.splits()
    (_, train_ids), (_, val_ids), (test_h5, test_ids) = sets["train"], sets["val"], sets["test"]
    test_ids, eval_ids = test_ids[: cfg.data.test_sims], val_ids[: cfg.data.eval_sims]
    train_loader = loader(cfg, cfg.data.h5_path, train_ids, cfg.data.frames_per_sim, train=True)
    val_loader = loader(cfg, cfg.data.h5_path, eval_ids, cfg.data.frames_per_sim, train=False)
    test_loader = loader(cfg, test_h5, test_ids, None, train=False)
    print(
        f"{len(train_ids)} train / {len(val_ids)} val ({len(eval_ids)} evaluated mid-run) / "
        f"{len(test_ids)} test sims (whole trajectories at the end), {world} x {device}"
    )

    adapter, vaes, norm = build(cfg, train_ids, device)
    clip = cfg.data.clip_len
    init, _ = evaluate(adapter, val_loader, vaes, norm, clip)
    report("init", init, adapter.fields)

    vae_params = [p for vae in vaes for p in vae.trainable_params]
    opt, sched = optimizer(cfg.train, adapter, vae_params)
    params = [p for g in opt.param_groups for p in g["params"]]
    w0 = [p.detach().cpu() for p in vae_params]  # start weights for the drift; read at evals only
    steps, history = [], []  # per update / per evaluation
    metrics = init  # last validation; stays the init for an eval-only run
    batches = endless(train_loader)
    t0 = t_log = time.time()
    for step in range(1, cfg.train.steps + 1):
        x = adapter.select(next(batches)[0].flatten(0, 1).to(device))
        result = train_step(cfg, adapter, vaes, x, opt, sched, params)
        steps.append({"step": step, "loss": result["loss"], "update_ratio": result["update_ratio"]})
        if step % cfg.train.log_every == 0:
            recent = steps[-cfg.train.log_every :]
            now, per_step = time.time(), (time.time() - t_log) / len(recent)
            t_log = now
            print(
                progress(
                    step,
                    cfg.train.steps,
                    step * cfg.train.batch_sims / len(train_ids),
                    sum(s["loss"] for s in recent) / len(recent),
                    [g["lr"] for g in opt.param_groups],
                    recent[-1]["update_ratio"],
                    per_step * (cfg.train.steps - step),
                )
            )
        if step % cfg.train.eval_every == 0 or step == cfg.train.steps:
            metrics, _ = evaluate(adapter, val_loader, vaes, norm, clip)
            if w0:
                metrics["vae_drift"] = rel_change(vae_params, w0)
            history.append({"step": step, "train": {k: result[k] for k in ("rmse", "h1")}, **metrics})
            report(f"val {step}", metrics, adapter.fields, f" | loss {result['loss']:.4f} | {hours(time.time() - t0)}")
            if rank == 0:
                save_weights(run_dir, adapter, vaes, cfg.weights_dir)  # a killed run keeps its last evaluated weights

    final, example = evaluate(adapter, test_loader, vaes, norm, None)
    report(f"test {len(test_ids)} whole trajectories", final, adapter.fields, f" | {hours(time.time() - t0)}")
    if rank != 0:
        return
    record = {
        "config": cfg.model_dump(),
        "provenance": provenance(started, world),
        "init": init,
        "val": metrics,
        "final": final,
        "history": history,
        "steps": steps,
        "n_params": sum(p.numel() for p in adapter.parameters()),
    }
    write_outputs(cfg, run_dir, adapter, vaes, record, drift_per_block(vaes, w0) if w0 else {}, example)


if __name__ == "__main__":
    typer.run(main)
