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
import yaml
from torch.utils.data import DataLoader

from windinet.data.frames import FrameSampler, endless, load_field_stats
from windinet.experiment.config import start_run
from windinet.experiment.distributed import all_mean, rank, shard, sync, world
from windinet.experiment.log import log_metrics, log_progress
from windinet.experiment.optim import adamw, rel_change
from windinet.experiment.runs import provenance, write_index
from windinet.field_adapter import GroupedAdapter, group_tag
from windinet.wan.dit.benchmark import Benchmark
from windinet.wan.dit.latents import WithLatents
from windinet.wan.figures import save_adapter_rgb, save_examples, save_panel, save_vae_curves
from windinet.wan.vae.config import TrainConfig, VaeStageConfig
from windinet.wan.vae.evaluate import evaluate, latent_stats
from windinet.wan.vae.loss import loss_terms
from windinet.wan.weights import load_weights, save_weights, vae_file


def loader(cfg: VaeStageConfig, h5: str, sims: list[str], frames: int | None, train: bool, split: str) -> DataLoader:
    """Training: random frames per access, this rank's share of the global batch. Evaluation: fixed frames, one sim
    per item; `frames` None = whole trajectories. With data.latents: whole trajectories with their rollout latents."""
    d = cfg.data
    if d.latents:
        sims = [s for s in sims if (Path(d.latents) / split / f"{s}.pt").exists()]
        sampler = WithLatents(
            FrameSampler(h5, shard(sims), None, 0, stride=d.frame_stride, spec=d.spec), d.latents, split
        )
    else:
        sampler = FrameSampler(
            h5, shard(sims), frames, None if train else 0, d.clip, d.frame_stride, train and d.augment, spec=d.spec
        )
    batch = max(1, cfg.train.batch_sims // world) if train else 1  # max: an eval-only run never draws a batch
    return DataLoader(
        sampler,
        batch,
        shuffle=train,
        num_workers=cfg.data.num_workers,
        persistent_workers=cfg.data.num_workers > 0,
    )


def check_load(run: dict, cfg: VaeStageConfig) -> None:
    """Refuse a `load` run whose dataset or adapter layout differs: same shapes, so it would load silently wrong."""
    dataset = run.get("data", {}).get("dataset", "eulermq")  # runs before the field are eulermq
    if dataset != cfg.data.dataset:
        raise typer.BadParameter(f"load: {cfg.load} is a run on {dataset}, not {cfg.data.dataset}")
    for key in ("groups", "log_fields"):  # the inverse's clamp bounds follow the current config, the stats the run's
        theirs, ours = run.get("adapter", {}).get(key), getattr(cfg.adapter, key)
        if theirs != ours:
            raise typer.BadParameter(f"load: {cfg.load} has adapter.{key}={theirs}, this config {ours}")


def build(cfg: VaeStageConfig, train_ids: list[str], device: torch.device) -> tuple[GroupedAdapter, list]:
    """Adapter and VAEs with their weights (from `cfg.load`, else data-initialised; VAEs from `cfg.vae_load`)."""
    stats = load_field_stats(cfg.data.stats_json)
    a = cfg.adapter
    adapter = GroupedAdapter(a.groups, stats, cfg.data.spec, a.log_fields, a.clip_quantile).to(device)
    adapter.requires_grad_(cfg.train.adapter_lr > 0)  # fixed adapter: no gradient through the encoder for it
    vaes = []
    if cfg.vae == "wan":
        from windinet.wan.vae.model import WanVAE  # diffusers only when a VAE is used

        n_vaes = len(cfg.adapter.groups) if cfg.train.vae_per_group else 1
        t = cfg.train
        checkpointing = t.checkpointing and t.vae_parts != "none"
        vaes = [
            WanVAE(
                device, train=t.vae_parts, channels=adapter.channels, checkpointing=checkpointing, precision=t.precision
            )
            for _ in range(n_vaes)
        ]
    if cfg.load:
        check_load(yaml.safe_load((Path(cfg.load) / "config.yaml").read_text()), cfg)
        load_weights(Path(cfg.load), adapter, [] if cfg.vae_load else vaes, device)
    else:  # tanh scales from 100 fixed sims x 2 frames: large enough that the init does not depend on the draw
        ids = train_ids[:: max(1, len(train_ids) // 100)][:100]
        probe = FrameSampler(cfg.data.h5_path, ids, 2, seed=0, spec=cfg.data.spec)
        adapter.init_from_data(adapter.select(torch.cat([probe[i][0] for i in range(len(probe))])).to(device))
    if cfg.vae_load:  # VAE i takes vae_<i-th letter>.pt, else vae.pt: groups are matched by position, not by name
        run, n = Path(cfg.vae_load), len(vaes)
        if not n or not all((run / vae_file(i, n)).exists() or (run / "vae.pt").exists() for i in range(n)):
            raise typer.BadParameter(f"vae_load: {run} has no weights for {len(vaes)} VAE(s)")
        load_weights(run, None, vaes, device)
    sync(adapter)  # the init subsamples at random: every rank takes rank 0's
    return adapter, vaes


def start_copies(vaes: list, channels: int, device: torch.device) -> list:
    """Frozen copies of the VAEs as the run starts: the reference of the latent drift."""
    from windinet.wan.vae.model import WanVAE

    copies = [WanVAE(device, channels=channels) for _ in vaes]
    for c, v in zip(copies, vaes, strict=True):
        c.vae.load_state_dict(v.vae.state_dict())
    return copies


def optimizer(cfg: TrainConfig, adapter: GroupedAdapter, vaes: list) -> tuple:
    """A param group per lr: the adapter, the fine-tuned VAE and, with io_lr, its in and out convolutions."""
    io = [p for vae in vaes for p in vae.io_params] if cfg.io_lr else []
    rest = [p for vae in vaes for p in vae.trainable_params if all(p is not q for q in io)]
    groups = [{"params": list(adapter.parameters()), "lr": cfg.adapter_lr}]
    groups += [{"params": ps, "lr": lr} for ps, lr in ((rest, cfg.vae_lr), (io, cfg.io_lr)) if ps]
    return adamw(groups, cfg.steps, cfg.warmup_steps, cfg.lr_floor)


def train_step(
    cfg: VaeStageConfig, adapter: GroupedAdapter, vaes: list, x: torch.Tensor, opt, sched, params, z=None
) -> dict:
    """One update on the frames `x` (with `z`, the sims' rollout latents: decoded against them): micro-batched
    backward, gradients averaged over ranks, clip, step.
    Returns the loss terms (mean over the micro-batches) and the relative weight change per param group."""
    opt.zero_grad(set_to_none=True)
    per = len(x) // len(z) if z is not None else cfg.data.clip_len  # frames per clip (a whole trajectory with z)
    chunk = per * cfg.train.micro_batch if vaes else len(x)  # frames per backward
    zs = z.split(cfg.train.micro_batch) if z is not None else [None] * -(-len(x) // chunk)
    result: dict = {}
    for xi, zi in zip(x.split(chunk), zs, strict=True):
        terms = loss_terms(adapter, xi, vaes, cfg.data.clip_len, cfg.loss, zi)
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


def write_outputs(
    cfg: VaeStageConfig, run_dir: Path, adapter, vaes, record: dict, drift: dict, rgbs: list, examples: dict
) -> None:
    """Figures, videos, weights, metrics.json and the results index."""
    save_panel(examples, adapter.fields, cfg.data, run_dir / "panel.png")
    save_adapter_rgb(rgbs, run_dir / "adapter_rgb.png")
    kind = "rollout" if cfg.data.latents else "recon"
    save_examples(examples, adapter.fields, cfg.data, run_dir, cfg.weights_dir, kind)
    save_vae_curves(record["steps"], record["history"], adapter.fields, run_dir / "curves.png", drift)
    save_weights(run_dir, adapter, vaes, cfg.weights_dir)
    (run_dir / "metrics.json").write_text(json.dumps(record, indent=1))
    write_index(Path(cfg.results_dir))
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
    train_loader = loader(cfg, cfg.data.h5_path, train_ids, cfg.data.frames_per_sim, True, "train")
    val_loader = loader(cfg, cfg.data.h5_path, eval_ids, cfg.data.frames_per_sim, False, "val")
    test_loader = loader(cfg, test_h5, test_ids, None, False, "test")
    print(
        f"{len(train_ids)} train / {len(val_ids)} val ({len(eval_ids)} evaluated mid-run) / "
        f"{len(test_ids)} test sims (whole trajectories at the end), {world} x {device}"
    )

    adapter, vaes = build(cfg, train_ids, device)
    start_vaes = start_copies(vaes, adapter.channels, device)
    clip, spec = cfg.data.clip_len, cfg.data.spec
    rollouts = bool(cfg.data.latents)
    val_args = (None, cfg.data.frame_stride, True) if rollouts else (clip,)  # rollouts: whole trajectories
    init, *_ = evaluate(adapter, val_loader, vaes, *val_args, spec=spec)
    log_metrics("init", init)

    vae_params = [p for vae in vaes for p in vae.trainable_params]
    opt, sched = optimizer(cfg.train, adapter, vaes)
    params = [p for g in opt.param_groups for p in g["params"]]
    w0 = [p.detach().to("cpu", copy=True) for p in vae_params]  # start weights for the drift; read at evals only
    steps, history = [], []  # per update / per evaluation
    metrics = init  # last validation; stays the init for an eval-only run
    batches = endless(train_loader)
    t0 = t_log = time.time()
    for step in range(1, cfg.train.steps + 1):
        t_step = time.time()
        frames, z = next(batches)
        x = adapter.select(frames.flatten(0, 1).to(device))
        result = train_step(cfg, adapter, vaes, x, opt, sched, params, z.to(device) if rollouts else None)
        seconds = time.time() - t_step  # data wait + update, no evaluation
        steps.append({"step": step, "loss": result["loss"], "update_ratio": result["update_ratio"], "seconds": seconds})
        if step % cfg.train.log_every == 0:
            recent = steps[-cfg.train.log_every :]
            now, per_step = time.time(), (time.time() - t_log) / len(recent)
            t_log = now
            log_progress(
                step,
                cfg.train.steps,
                step * cfg.train.batch_sims / len(train_ids),
                sum(s["loss"] for s in recent) / len(recent),
                [g["lr"] for g in opt.param_groups],
                recent[-1]["update_ratio"],
                per_step * (cfg.train.steps - step),
                f"{sum(s['seconds'] for s in recent) / len(recent):.2f} s/update",
            )
        if step % cfg.train.eval_every == 0 or step == cfg.train.steps:
            metrics, *_ = evaluate(adapter, val_loader, vaes, *val_args, spec=spec)
            if w0:
                metrics["vae_drift"] = rel_change(vae_params, w0)
            history.append({"step": step, "train": {k: result[k] for k in ("rmse", "h1", "latent")}, **metrics})
            loss = result["loss"]
            log_metrics(f"val {step}", metrics, loss, time.time() - t0)
            if rank == 0:
                save_weights(run_dir, adapter, vaes, cfg.weights_dir)  # a killed run keeps its last evaluated weights
                (run_dir / "history.json").write_text(json.dumps(history, indent=1))  # and its curve so far

    stride = cfg.data.frame_stride
    bench = None if rollouts else Benchmark(cfg.data.stats_json, adapter.fields, stride, device, "bench_vae")
    final, rgbs, examples = evaluate(adapter, test_loader, vaes, None, stride, rollouts, spec=spec, bench=bench)
    log_metrics(f"test {len(test_ids)} sims, whole trajectories", final, elapsed=time.time() - t0)
    medians = "  ".join(f"{f} {final[f'vrmse_median_{f}']:.3f}" for f in adapter.fields)
    print(f"median over sims: VRMSE {final['vrmse_median_mean']:.3f} | {medians}")
    latent_start = None
    if vaes and not rollouts:
        final |= latent_stats(adapter, test_loader, vaes, start_vaes)
        latent_start = latent_stats(adapter, test_loader, start_vaes, start_vaes)
        print(
            f"latents: |mean| {final['latent_mean_abs']:.3f} (start {latent_start['latent_mean_abs']:.3f}), std "
            f"{final['latent_std_mean']:.3f} (start {latent_start['latent_std_mean']:.3f}), drift from start "
            f"{final['latent_drift']:.3f}"
        )
    if rank != 0:
        return
    record = {
        "config": cfg.model_dump(),
        "provenance": provenance(started, world),
        "init": init,
        "val": metrics,
        "final": final,
        "latent_start": latent_start,
        "history": history,
        "steps": steps,
        "n_params": sum(p.numel() for p in adapter.parameters()),
    }
    write_outputs(cfg, run_dir, adapter, vaes, record, drift_per_block(vaes, w0) if w0 else {}, rgbs, examples)


if __name__ == "__main__":
    typer.run(main)
