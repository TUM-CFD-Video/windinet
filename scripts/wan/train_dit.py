#!/usr/bin/env python
"""DiT stage: fine-tune the Wan 2.1 transformer on the latents of a fixed adapter + frozen encoder, and evaluate it by
rolling out every test simulation from its first frame and gamma, decoded through the VAE stage's decoders.

    torchrun --standalone --nproc_per_node=8 scripts/wan/train_dit.py configs/wan/dit.yaml --name rgb3_dit --desc "..."
    python scripts/wan/train_dit.py configs/wan/dit.yaml --name x --desc "..." --set load=results/wan/dit/rgb3_dit \
        --set train.steps=0                                            # evaluate a run

Rectified flow: x_σ = (1 - σ) z + σ ε, target ε - z, σ logit-normal (train.sigma_*); with cond.first_frame the first
latent frame stays clean at timestep 0 and is left out of the loss. train.batch_sims is the global batch, split over
the ranks; evaluation is sharded; rank 0 logs and writes <results_dir>/<name>/{config.yaml, metrics.json, strip.png,
curves.png, dit.pt, cond.pt} (the weights in weights_dir, symlinked) and regenerates the results_dir index.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import typer
import yaml
from torch.utils.data import DataLoader

from windinet.eulermq.data import FrameSampler, endless, load_field_stats, parse_gamma
from windinet.experiment.config import start_run
from windinet.experiment.distributed import all_mean, rank, shard, sync, world
from windinet.experiment.log import hours, progress, report
from windinet.experiment.metrics import rel_change
from windinet.experiment.optim import adamw
from windinet.experiment.runs import provenance, write_index
from windinet.field_adapter import load_adapter
from windinet.wan.dit import conditioning
from windinet.wan.dit.config import DitStageConfig
from windinet.wan.dit.flow import draw_sigmas, noised, velocity_target
from windinet.wan.dit.latents import LatentDataset, encode_info
from windinet.wan.dit.model import WanDiT, autocast
from windinet.wan.dit.rollout import evaluate
from windinet.wan.figures import save_curves, save_fields_strip
from windinet.wan.vae.model import WanVAE
from windinet.wan.weights import load_weights, save_state


def train_loader(cfg: DitStageConfig, ids: list[str]) -> DataLoader:
    """This rank's share of the global batch, shuffled, whole or truncated latent trajectories."""
    data = LatentDataset(cfg.latents.dir, "train", shard(ids), cfg.latents.frames)
    workers = cfg.data.num_workers
    return DataLoader(data, max(1, cfg.train.batch_sims // world), shuffle=True, num_workers=workers)


def eval_loaders(cfg: DitStageConfig, split: str, h5: str, ids: list[str]) -> tuple[DataLoader, DataLoader]:
    """The latents (first frame + gamma) and the simulations (ground truth) of this rank's sims, same order."""
    ids = shard(ids)
    latents = LatentDataset(cfg.latents.dir, split, ids, cfg.latents.frames)
    fields = FrameSampler(h5, ids, None, 0, consecutive=True)  # whole trajectories; evaluate() takes the first frames
    batch, workers = cfg.sampling.batch, cfg.data.num_workers
    return DataLoader(latents, batch, num_workers=workers), DataLoader(fields, batch, num_workers=workers)


def build(cfg: DitStageConfig, device: torch.device) -> tuple:
    """Adapter and decoders of the VAE run, the transformer, the conditioning, and the field (mean, std) of the
    clipped metric."""
    run = Path(cfg.vae)
    vae_cfg = yaml.safe_load((run / "config.yaml").read_text())
    if encode_info(cfg.latents.dir)["groups"] != vae_cfg["adapter"]["groups"]:
        raise typer.BadParameter(f"{cfg.latents.dir} was encoded with other groups than {run}")
    adapter = load_adapter(run, device)
    vaes = [WanVAE(device=device) for _ in range(len(adapter.groups) if vae_cfg["train"]["vae_per_group"] else 1)]
    load_weights(run, adapter, vaes, device)
    dtype = torch.bfloat16 if cfg.model.precision == "bf16" else torch.float32
    dit = WanDiT(
        device, dtype, cfg.model.repo, cfg.load, train=cfg.train.steps > 0, checkpointing=cfg.model.checkpointing
    )
    cond = conditioning.build(cfg.cond, device)
    if cfg.load and (Path(cfg.load) / "cond.pt").exists():
        cond.load_state_dict(torch.load(Path(cfg.load) / "cond.pt", map_location=device))
    sync(cond)  # a fresh embedding is random: every rank takes rank 0's
    stats = load_field_stats(cfg.data.stats_json)
    norm = tuple(torch.tensor([stats[n][k] for n in adapter.fields], device=device) for k in ("mean", "std"))
    return adapter, vaes, dit, cond, norm


def train_step(cfg: DitStageConfig, dit: WanDiT, cond, latents, gamma, opt, sched, params) -> dict:
    """One update: micro-batched flow-matching loss, gradients averaged over ranks, clip, step. Returns the loss
    (mean over the micro-batches) and the relative weight change per param group."""
    if latents.shape[1] != 1:
        raise NotImplementedError("two latent streams per sample: the pairs DiT is not built yet")
    opt.zero_grad(set_to_none=True)
    z_all, loss_total = latents[:, 0], 0.0
    clean = z_all.shape[-2] * z_all.shape[-1] // 4 if cfg.cond.first_frame else 0  # tokens of latent frame 0
    for z, g in zip(z_all.split(cfg.train.micro_batch), gamma.split(cfg.train.micro_batch), strict=True):
        noise = torch.randn_like(z)
        sigma = draw_sigmas(len(z), z.device, cfg.train.sigma_mean, cfg.train.sigma_std, cfg.train.sigma_shift)
        x = noised(z, noise, sigma)
        if clean:
            x[:, :, :1] = z[:, :, :1]
        with autocast(cfg.model.precision, dit.device):
            v = dit.velocity(x, sigma * 1000, cond(g), clean)
        err = (v - velocity_target(z, noise)).square()
        loss = (err[:, :, 1:] if clean else err).mean() * (len(z) / len(z_all))
        loss.backward()
        loss_total += loss.item()
    all_mean(*(p.grad for p in params if p.grad is not None))
    if cfg.train.max_grad_norm:
        torch.nn.utils.clip_grad_norm_(params, cfg.train.max_grad_norm)
    before = [[p.detach().clone() for p in g["params"]] for g in opt.param_groups]
    opt.step()
    sched.step()
    ratio = [rel_change(g["params"], b) for g, b in zip(opt.param_groups, before, strict=True)]
    return {"loss": loss_total, "update_ratio": ratio}


def save_run(run_dir: Path, dit: WanDiT, cond, weights_dir: str | None) -> None:
    save_state(run_dir, "dit.pt", dit.transformer.state_dict(), weights_dir)
    if list(cond.parameters()):
        save_state(run_dir, "cond.pt", cond.state_dict(), weights_dir)


def main(
    config_path: Path = typer.Argument(..., help="YAML matching DitStageConfig"),
    name: str | None = typer.Option(None, help="run name = results folder (overrides the YAML's)"),
    desc: str | None = typer.Option(None, help="one sentence: what this run tests (overrides the YAML's description)"),
    set_: list[str] = typer.Option([], "--set", help="dotted override, e.g. train.steps=0"),
    overwrite: bool = typer.Option(False, help="replace an existing run folder of the same name"),
) -> None:
    started = time.time()
    cfg, run_dir, device = start_run(DitStageConfig, config_path, name, desc, set_, overwrite)

    sets = cfg.data.splits()
    (_, train_ids), (val_h5, val_ids), (test_h5, test_ids) = sets["train"], sets["val"], sets["test"]
    eval_ids, test_ids = val_ids[: cfg.data.eval_sims], test_ids[: cfg.data.test_sims]
    gammas = sorted({round(parse_gamma(sid), 3) for sid in eval_ids + test_ids})
    batches = endless(train_loader(cfg, train_ids))
    val_loaders = eval_loaders(cfg, "val", val_h5, eval_ids)
    test_loaders = eval_loaders(cfg, "test", test_h5, test_ids)
    print(
        f"{len(train_ids)} train / {len(eval_ids)} val / {len(test_ids)} test sims, latents from {cfg.latents.dir}, "
        f"{world} x {device}"
    )

    adapter, vaes, dit, cond, norm = build(cfg, device)
    init, _ = evaluate(dit, cond, adapter, vaes, val_loaders, gammas, norm, cfg)
    report("init", init, adapter.fields)

    groups = [{"params": dit.trainable_params, "lr": cfg.train.lr}]
    if list(cond.parameters()):
        groups.append({"params": list(cond.parameters()), "lr": cfg.train.lr})
    opt, sched = adamw(groups, cfg.train.steps, cfg.train.warmup_steps, cfg.train.lr_floor, cfg.train.weight_decay)
    params = [p for g in groups for p in g["params"]]
    steps, history = [], []  # per update / per evaluation
    metrics = init
    t0 = t_log = time.time()
    for step in range(1, cfg.train.steps + 1):
        latents, gamma = next(batches)
        result = train_step(cfg, dit, cond, latents.to(device), gamma.to(device), opt, sched, params)
        steps.append({"step": step, **result})
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
            metrics, _ = evaluate(dit, cond, adapter, vaes, val_loaders, gammas, norm, cfg)
            history.append({"step": step, "train": {"loss": result["loss"]}, **metrics})
            report(f"val {step}", metrics, adapter.fields, f" | loss {result['loss']:.4f} | {hours(time.time() - t0)}")
            if rank == 0:
                save_run(run_dir, dit, cond, cfg.weights_dir)

    final, example = evaluate(dit, cond, adapter, vaes, test_loaders, gammas, norm, cfg)
    report(f"test {len(test_ids)} trajectories", final, adapter.fields, f" | {hours(time.time() - t0)}")
    if rank != 0:
        return
    truth, pred = example
    save_fields_strip(pred, truth, adapter.fields, norm[1].cpu(), run_dir / "strip.png")
    save_curves(steps, history, {}, adapter.fields, run_dir / "curves.png")
    save_run(run_dir, dit, cond, cfg.weights_dir)
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
    write_index(run_dir.parent)
    print(f"wrote {run_dir}/ and {run_dir.parent / 'README.md'}")


if __name__ == "__main__":
    typer.run(main)
