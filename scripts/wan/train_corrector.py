#!/usr/bin/env python
"""Corrector stage: train a small U-Net that corrects the frozen DiT's decoded rollouts frame by frame (one window
of train.window frames per update, the history arm burnt in to it by its own rollout), then correct every test rollout
from the true frame 0 and score it beside Wan alone.

    torchrun --standalone --nproc_per_node=8 scripts/wan/train_corrector.py configs/wan/corrector.yaml \
        --name ar --desc "..."                                         # autoregressive: own previous frame as input
    torchrun ... --set model.history=0 --name control --desc "..."  # control: Wan's frames alone

Writes <results_dir>/<name>/{config.yaml, metrics.json, curves.png, corrected_gamma<g>.mp4, example_gamma<g>.pt,
corrector.pt, corrector_<step>.pt} (big files in weights_dir, symlinked) and regenerates the results_dir index.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import typer
import yaml
from torch.utils.data import DataLoader

from windinet.data.frames import load_field_stats
from windinet.data.specs import EULERMQ
from windinet.experiment.config import load_config, start_run
from windinet.experiment.distributed import all_mean, rank, shard, sync, world
from windinet.experiment.log import log_metrics, log_progress
from windinet.experiment.metrics import vrmse
from windinet.experiment.optim import EMA, adamw
from windinet.experiment.runs import provenance, write_index
from windinet.wan.corrector.config import CorrectorStageConfig
from windinet.wan.corrector.data import RolloutFields, Windows
from windinet.wan.corrector.evaluate import evaluate
from windinet.wan.corrector.model import Corrector, rollout, teacher_forced
from windinet.wan.dit.latents import encode_info
from windinet.wan.figures import save_dit_curves, save_examples
from windinet.wan.vae.config import DataConfig
from windinet.wan.weights import save_state


def eval_loader(cfg: CorrectorStageConfig, split: str, h5: str, ids: list[str], batch: int) -> DataLoader:
    """Whole rollouts of the first folder with their sims, this rank's share, in order; every sim must have one."""
    data = RolloutFields(cfg.rollouts[:1], split, h5, shard(ids), cfg.data.frame_stride)
    if len(data) != len(shard(ids)):  # a partial folder would score other sims than the DiT's evaluation
        raise typer.BadParameter(f"{cfg.rollouts[0]}/{split} lacks {len(shard(ids)) - len(data)} of this rank's sims")
    return DataLoader(data, batch, num_workers=cfg.data.num_workers)


def check_stride(cfg: CorrectorStageConfig) -> None:
    """Every rollout folder decodes latents of data.frame_stride."""
    for d in cfg.rollouts:
        latents = yaml.safe_load((Path(d) / "decode.yaml").read_text())["rollouts"]
        if encode_info(latents)["config"]["data"].get("frame_stride", 1) != cfg.data.frame_stride:
            raise typer.BadParameter(f"{d} holds rollouts of another frame stride than data.frame_stride")


def build(cfg: CorrectorStageConfig, device: torch.device) -> Corrector:
    """The corrector, from `load` or fresh; the same weights on every rank."""
    model = Corrector(cfg.model, load_field_stats(cfg.data.stats_json)).to(device)
    if cfg.load:
        model.load_state_dict(torch.load(Path(cfg.load) / "corrector.pt", map_location=device))
    sync(model)  # start_run seeds every rank differently
    return model


def train_step(cfg: CorrectorStageConfig, model: Corrector, wan, truth, gamma, opt, sched, params) -> float:
    """One update on a window, teacher-forced or after the history arm's burn-in under no_grad; gradients averaged over
    ranks, clip, step."""
    opt.zero_grad(set_to_none=True)
    tc, start = cfg.train, truth.shape[1] - cfg.train.window
    window, target, past = range(start, truth.shape[1]), truth[:, start:], truth[:, :1]
    if tc.teacher_forcing:
        pred = teacher_forced(model, wan, truth, gamma, window, tc.history_noise)
    else:
        if model.history:
            with torch.no_grad():
                past = rollout(model, wan, past, gamma, range(1, start))
        pred = rollout(model, wan, past, gamma, window)[:, past.shape[1] :]
    if tc.loss == "mse":
        loss = ((pred - target) / model.std).square().mean()
    else:
        loss = torch.stack([vrmse(p, x).square() for p, x in zip(pred, target, strict=True)]).mean()
    loss.backward()
    all_mean(*(p.grad for p in params if p.grad is not None))
    if cfg.train.max_grad_norm:
        torch.nn.utils.clip_grad_norm_(params, cfg.train.max_grad_norm)
    opt.step()
    sched.step()
    return loss.item()


def validate(model: Corrector, val: DataLoader, train_eval: DataLoader, stride: int) -> dict:
    """Val metrics plus, as train_*, the means and gain on train sims: train gain far above val gain = overfitting."""
    metrics, _ = evaluate(model, val, stride)
    train, _ = evaluate(model, train_eval, stride)
    return metrics | {f"train_{k}": v for k, v in train.items() if k.endswith("_mean") or k == "gain"}


def main(
    config_path: Path = typer.Argument(..., help="YAML matching CorrectorStageConfig"),
    name: str | None = typer.Option(None, help="run name = results folder (overrides the YAML's)"),
    desc: str | None = typer.Option(None, help="one sentence: what this run tests (overrides the YAML's description)"),
    set_: list[str] = typer.Option([], "--set", help="dotted override, e.g. model.history=0"),
    overwrite: bool = typer.Option(False, help="replace an existing run folder of the same name"),
) -> None:
    started = time.time()
    # before start_run: a refused run leaves no folder that blocks its name
    dataset = DataConfig(**load_config(config_path, set_).get("data", {})).dataset
    if dataset != "eulermq":  # the rollouts and the model assume eulermq's fields and gamma
        raise typer.BadParameter(f"not yet supported on {dataset}")
    cfg, run_dir, device = start_run(CorrectorStageConfig, config_path, name, desc, set_, overwrite)
    stride, batch = cfg.data.frame_stride, max(1, cfg.train.batch_sims // world)

    sets = cfg.data.splits()
    (train_h5, train_ids), (val_h5, val_ids), (test_h5, test_ids) = sets["train"], sets["val"], sets["test"]
    eval_ids, test_ids = val_ids[: cfg.data.eval_sims], test_ids[: cfg.data.test_sims]
    train_eval_ids = train_ids[: cfg.data.eval_sims // 2]
    check_stride(cfg)
    val = eval_loader(cfg, "val", val_h5, eval_ids, batch)
    test = eval_loader(cfg, "test", test_h5, test_ids, batch)
    train_eval = eval_loader(cfg, "train", train_h5, train_eval_ids, batch)
    data = RolloutFields(cfg.rollouts, "train", train_h5, shard(train_ids), stride, cfg.data.augment)
    windows = Windows(len(data), batch, cfg.train.window, data.frames, cfg.train.seed)
    batches = iter(DataLoader(data, batch_sampler=windows, num_workers=cfg.data.num_workers))
    held = [sum(Path(r, "train") in fs for fs in data.folders.values()) for r in cfg.rollouts]
    print(
        f"{len(data)} train sims on rank 0 (rollouts per folder {held}) / {len(eval_ids)} val / {len(test_ids)} test "
        f"sims, {data.frames} frames, {world} x {device}"
    )

    model = build(cfg, device)
    init = validate(model, val, train_eval, stride)
    log_metrics("init", init)

    params = list(model.parameters())
    groups = [{"params": params, "lr": cfg.train.lr}]
    opt, sched = adamw(groups, cfg.train.steps, cfg.train.warmup_steps, cfg.train.lr_floor, cfg.train.weight_decay)
    ema = EMA(params, cfg.train.ema) if cfg.train.ema and cfg.train.steps else None
    steps, history = [], []  # per update / per evaluation
    metrics = init
    t0 = t_log = time.time()
    for step in range(1, cfg.train.steps + 1):
        wan, truth, gamma = next(batches)
        wan, truth, gamma = wan.to(device).float(), truth.to(device), gamma.to(device).float()
        loss = train_step(cfg, model, wan, truth, gamma, opt, sched, params)
        if ema:
            ema.update()
        steps.append({"step": step, "loss": loss})
        if step % cfg.train.log_every == 0:
            recent = steps[-cfg.train.log_every :]
            now = time.time()
            per_step, t_log = (now - t_log) / len(recent), now
            mean, lrs = sum(s["loss"] for s in recent) / len(recent), [g["lr"] for g in opt.param_groups]
            epoch = step * cfg.train.batch_sims / (len(data) * world)
            log_progress(step, cfg.train.steps, epoch, mean, lrs, [], per_step * (cfg.train.steps - step))
        if step % cfg.train.eval_every == 0 or step == cfg.train.steps:
            if ema:  # evaluate and save the average, train on the raw weights
                ema.swap()
            metrics = validate(model, val, train_eval, stride)
            history.append({"step": step, "train": {"loss": loss}, **metrics})
            log_metrics(f"val {step}", metrics, loss, time.time() - t0)
            if rank == 0 and step < cfg.train.steps:  # the last point is the final corrector.pt below
                save_state(run_dir, f"corrector_{step}.pt", model.state_dict(), cfg.weights_dir)
                (run_dir / "history.json").write_text(json.dumps(history, indent=1))  # the curve so far, if killed
            if ema and step < cfg.train.steps:  # from the last step on: the average
                ema.swap()

    final, examples = evaluate(model, test, stride)
    log_metrics(f"test {len(test_ids)} sims", final, elapsed=time.time() - t0)
    if rank != 0:
        return
    fields = list(EULERMQ.fields)
    save_examples(examples, fields, cfg.data, run_dir, cfg.weights_dir, "rollout")
    save_dit_curves(steps, history, final, fields, run_dir / "curves.png")
    if cfg.train.steps:
        save_state(run_dir, "corrector.pt", model.state_dict(), cfg.weights_dir)
    record = {
        "config": cfg.model_dump(),
        "provenance": provenance(started, world),
        "init": init,
        "val": metrics,
        "final": final,
        "history": history,
        "steps": steps,
        "n_params": sum(p.numel() for p in params),
    }
    (run_dir / "metrics.json").write_text(json.dumps(record, indent=1))
    write_index(Path(cfg.results_dir))
    print(f"wrote {run_dir}/ and {Path(cfg.results_dir) / 'README.md'}")


if __name__ == "__main__":
    typer.run(main)
