#!/usr/bin/env python
"""DiT stage: fine-tune the Wan 2.1 transformer on the latents of a fixed adapter + frozen encoder, and evaluate it by
rolling out every test simulation from its first frame and gamma, decoded through the VAE stage's decoders. An adapter
with several images per sample (pairs) gives several latents; they enter the DiT stacked in the channels.

    torchrun --standalone --nproc_per_node=8 scripts/wan/train_dit.py configs/wan/dit.yaml --name rgb3_dit --desc "..."
    python scripts/wan/train_dit.py configs/wan/dit_eval.yaml --name <run>/<eval> --desc "..." \
        --set load=results/wan/eulermq/dit/<run>                               # evaluate a run
    torchrun ... scripts/wan/train_dit.py configs/wan/dit_anneal.yaml --set load=results/wan/eulermq/dit/<run> \
        --name <run>/stage2_anneal                                     # continue a run (nested stage folder)

Rectified flow: x_σ = (1 - σ) z + σ ε, target ε - z, σ logit-normal (train.sigma_*); the first cond.context
latent frames stay clean at timestep 0 and are left out of the loss. train.pixel_loss adds, for the lowest-sigma
latents, L1 + gradient-difference L1 between the decoded clean-latent estimate and the decoded clean latent.
train.batch_sims is the global batch, split over
the ranks; evaluation is sharded; rank 0 logs and writes <results_dir>/<name>/{config.yaml, metrics.json, curves.png,
rollout_<label>.mp4 + example_<label>.pt per `examples` (gamma<g>, worst, best), dit.pt,
cond.pt, dit_<step>.pt per eval point} (mp4, examples and weights in weights_dir, symlinked) and regenerates the
results_dir index. `load` + `load_step` evaluate (or continue from) an eval point. With train.ema, every evaluation
and every saved weight file is the EMA of the weights, and the raw weights are evaluated beside it (raw_* metrics).
"""

from __future__ import annotations

import copy
import json
import shutil
import time
from pathlib import Path

import torch
import typer
import yaml
from torch.utils.data import DataLoader, WeightedRandomSampler

from windinet.data.frames import FrameSampler, endless, parse_gamma
from windinet.experiment.config import start_run
from windinet.experiment.distributed import all_mean, all_sum, rank, shard, sync, world
from windinet.experiment.log import log_metrics, log_progress
from windinet.experiment.optim import EMA, adamw
from windinet.experiment.runs import provenance, write_index
from windinet.field_adapter import load_adapter
from windinet.wan.dit import conditioning
from windinet.wan.dit.config import DitStageConfig
from windinet.wan.dit.delta import scales, to_delta
from windinet.wan.dit.distill import Teacher, closeness
from windinet.wan.dit.evaluate import evaluate, render
from windinet.wan.dit.flow import clean_estimate, draw_sigmas, frame_sigmas, gated, noised, velocity_target
from windinet.wan.dit.latents import LatentDataset, encode_info, parse_latent_id, with_augmentations
from windinet.wan.dit.model import WanDiT, autocast
from windinet.wan.dit.prune import calibrate, choose, compose, importance, shrink
from windinet.wan.dit.rollout import noise_seeds, rollout
from windinet.wan.figures import save_dit_curves, save_examples
from windinet.wan.vae.loss import pixel_loss
from windinet.wan.vae.model import WanVAE
from windinet.wan.weights import load_weights, save_state


def train_loader(cfg: DitStageConfig, ids: list[str]) -> DataLoader:
    """This rank's share of the global batch, shuffled: whole or truncated latent trajectories, or random windows of
    latents.ar_window frames, from every file the folder holds per sim (symmetric copies, later starts)."""
    ids = with_augmentations(cfg.latents.dir, shard(ids))
    masks = cfg.train.detail_masks if cfg.train.detail_weight else None
    data = LatentDataset(cfg.latents.dir, "train", ids, cfg.latents.frames, cfg.latents.ar_window, masks)
    sampler = None
    if cfg.train.sim_weights:
        w = json.loads(Path(cfg.train.sim_weights).read_text())
        sampler = WeightedRandomSampler([w[parse_latent_id(i)[0]] for i in ids], len(ids))
    batch, workers = max(1, cfg.train.batch_sims // world), cfg.data.num_workers
    return DataLoader(data, batch, shuffle=sampler is None, sampler=sampler, num_workers=workers)


def eval_loaders(
    cfg: DitStageConfig, split: str, h5: str, ids: list[str], sharded: bool = True
) -> tuple[DataLoader, DataLoader]:
    """The latents (first frame + gamma) and the simulations (ground truth) of this rank's sims, same order."""
    ids = shard(ids) if sharded else ids
    latents = LatentDataset(cfg.latents.dir, split, ids, cfg.latents.frames)
    fields = FrameSampler(h5, ids, None, 0, stride=cfg.data.frame_stride)  # whole trajectories; evaluate() truncates
    batch, workers = cfg.sampling.batch, cfg.data.num_workers
    return DataLoader(latents, batch, num_workers=workers), DataLoader(fields, batch, num_workers=workers)


def tag(step: int | None) -> str:
    """Suffix of the weight files of an eval point; the final weights have none."""
    return f"_{step}" if step else ""


def build(cfg: DitStageConfig, device: torch.device) -> tuple:
    """Adapter and decoders of the VAE run, the transformer and the conditioning."""
    run = Path(cfg.vae)
    vae_cfg = yaml.safe_load((run / "config.yaml").read_text())
    info = encode_info(cfg.latents.dir)
    groups = info["groups"]
    if groups != vae_cfg["adapter"]["groups"]:
        raise typer.BadParameter(f"{cfg.latents.dir} was encoded with other groups than {run}")
    if info["config"]["data"].get("frame_stride", 1) != cfg.data.frame_stride:
        raise typer.BadParameter(f"{cfg.latents.dir} was encoded with another frame stride than data.frame_stride")
    adapter = load_adapter(run, device)
    n_vaes = len(adapter.groups) if vae_cfg["train"]["vae_per_group"] else 1
    checkpointing = cfg.train.pixel_loss > 0
    vaes = [WanVAE(device=device, channels=adapter.channels, checkpointing=checkpointing) for _ in range(n_vaes)]
    load_weights(run, adapter, vaes, device)
    adapter.requires_grad_(False)  # frozen in this stage; the pixel loss back-propagates through it
    dit, cond = load_dit(cfg, device, cfg.load, cfg.load_step, len(groups), train=cfg.train.steps > 0)
    return adapter, vaes, dit, cond


def load_dit(cfg: DitStageConfig, device, run: str | None, step: int | None, num_latents: int, train: bool) -> tuple:
    """The transformer of `run` (cut to its prune.json if it was pruned; None = pretrained) and its conditioning."""
    dtype = torch.bfloat16 if cfg.model.precision == "bf16" else torch.float32
    load = Path(run) if run else None
    pruned = load and (load / "prune.json").exists()
    dit = WanDiT(
        device,
        dtype,
        cfg.model.repo,
        load / f"dit{tag(step)}.pt" if load else None,
        train=train,
        checkpointing=cfg.model.checkpointing and train,
        num_latents=num_latents,
        shift=cfg.sampling.shift,
        pruned=json.loads((load / "prune.json").read_text())["spec"] if pruned else None,
    )
    cond = conditioning.build(cfg.cond, device, dit.transformer.condition_embedder)
    if load and (load / f"cond{tag(step)}.pt").exists():
        state = torch.load(load / f"cond{tag(step)}.pt", map_location=device)
        if state.keys() == cond.state_dict().keys():
            cond.load_state_dict(state)
        else:
            print(f"{load} conditioned the DiT differently ({cfg.cond.kind} now): the conditioning starts fresh")
    sync(cond)  # a fresh embedding is random: every rank takes rank 0's
    cond.requires_grad_(train)
    if cfg.noise != "random":
        learned = train and cfg.noise == "learned"
        dit.noise = torch.nn.Parameter(shared_noise(cfg, load, step).to(device), requires_grad=learned)
    return dit, cond


def shared_noise(cfg: DitStageConfig, load: Path | None, step: int | None) -> torch.Tensor:
    """The noise every sample shares: the loaded run's noise.pt, else one seeded N(0, 1) draw of a latent's shape."""
    saved = load / f"noise{tag(step)}.pt" if load else None
    if saved and saved.exists():
        return torch.load(saved, map_location="cpu")
    z = torch.load(next((Path(cfg.latents.dir) / "train").glob("*.pt")))["latents"][:, :, : cfg.latents.frames]
    return torch.randn(z.flatten(0, 1).shape, generator=torch.Generator().manual_seed(cfg.sampling.seed))


def states(cfg: DitStageConfig, dit: WanDiT, cond, z: torch.Tensor, g: torch.Tensor) -> tuple:
    """A micro-batch's training inputs: x_σ at drawn sigmas, train.unrolled of them replaced by the sampler's own state
    (call c of an n-step chain from noise, Euler steps, without gradient). Returns x_σ, sigma [B, F'], the noise,
    the frames in the loss and how many lead samples are unrolled."""
    t = cfg.train
    noise = torch.randn_like(z) if dit.noise is None else dit.noise.expand_as(z)
    draws = draw_sigmas(z.shape[0] * z.shape[2], z.device, t.sigma_mean, t.sigma_std, t.sigma_shift)
    sigma, target = frame_sigmas(draws.view(len(z), -1), cfg.cond.context, t.history_noise)
    x_sigma = noised(z, noise, sigma)
    k = round((t.unrolled or 0) * len(z))
    if k:
        n = int(torch.randint(t.unrolled_steps[0], t.unrolled_steps[1] + 1, ()))
        c = int(torch.randint(2, n + 1, ()))
        s, shift = torch.linspace(1, 0, n + 1)[:c], cfg.sampling.shift
        sigmas = (shift * s / (1 + (shift - 1) * s)).tolist()  # the sampler's, as UniPC uses them
        levels = [(sigma[:k] > 0) * level for level in sigmas]  # the context stays clean
        sigma = sigma.clone()  # the noise mix above keeps sigma for its gradient (learned noise)
        sigma[:k] = levels[-1]
        x = noised(z[:k], noise[:k], levels[0])
        with torch.no_grad(), autocast(cfg.model.precision, dit.device):
            for a, b in zip(levels, levels[1:], strict=False):
                x = clean_estimate(x, dit.velocity(x, a * 1000, cond(g[:k])), a - b)  # x + (b - a) v
        x_sigma[:k] = x
    return x_sigma, sigma, noise, target, k


def train_step(
    cfg: DitStageConfig, dit: WanDiT, cond, adapter, vaes, latents, gamma, opt, sched, params, teacher, detail=None
):
    """One update: micro-batched flow-matching loss (train.unrolled of the samples on the sampler's own states), plus
    train.pixel_loss on the decoded clean-latent estimates of the lowest-sigma latents, plus with a `teacher` the
    distillation terms at the same inputs, gradients averaged over ranks, clip, step. `detail` [B, F', h, w]: the
    latent cells' detail shares (train.detail_weight). Returns the terms (means over the batch)."""
    opt.zero_grad(set_to_none=True)
    z_all = latents.flatten(1, 2)  # a sample's latents in channels
    terms = {"loss": 0.0, "pixel": 0.0, "front": 0.0, "flow": 0.0, "kd_velocity": 0.0, "kd_features": 0.0}
    if dit.delta_scale is not None:
        z_all = to_delta(z_all, dit.delta_scale)
    t, groups, kd = cfg.train, latents.shape[1], cfg.distill
    mb = t.micro_batch
    for i, (z, g) in enumerate(zip(z_all.split(mb), gamma.split(mb), strict=True)):
        x_sigma, sigma, noise, target, k = states(cfg, dit, cond, z, g)
        with autocast(cfg.model.precision, dit.device):
            v = dit.velocity(x_sigma, sigma * 1000, cond(g))
        err = (v - velocity_target(z, noise)).square().mean(1)  # per latent frame and cell
        if k:  # their target is the clean latent itself, the noise is no longer the gap to it
            err[:k] = (clean_estimate(x_sigma[:k], v[:k], sigma[:k]) - z[:k]).square().mean(1)
        share = len(z) / len(z_all)
        w = 1 + t.late_weight * torch.linspace(0, 1, err.shape[1], device=err.device)[:, None, None]  # per frame
        if detail is not None:
            w = w * (1 + t.detail_weight * detail[i * mb : (i + 1) * mb])
        loss = (err * w)[target].mean() / w.expand_as(err)[target].mean() * share
        if teacher:  # on the same inputs, the unrolled ones being the student's own sampler states
            with autocast(cfg.model.precision, dit.device):
                v_teacher = teacher.velocity(x_sigma, sigma, g)
            parts = {"flow": loss, "kd_velocity": (v - v_teacher).square().mean((1, 3, 4))[target].mean() * share}
            if kd.features:
                parts["kd_features"] = teacher.feature_loss() * share
            loss = kd.flow * loss + kd.velocity * parts["kd_velocity"] + kd.features * parts["kd_features"]
            for key, part in parts.items():
                terms[key] += part.item()
        sel = gated(sigma, t.pixel_gate, t.pixel_gate_samples) if t.pixel_loss else []
        if len(sel):
            x0_hat = clean_estimate(x_sigma[sel], v[sel], sigma[sel]).unflatten(1, (groups, -1))
            ref, share = z[sel].unflatten(1, (groups, -1)), len(sel) / len(z_all)
            pixel, front = pixel_loss(adapter, vaes, x0_hat, ref, t.pixel_window, t.front_loss)
            loss = loss + t.pixel_loss * share * pixel
            terms["pixel"] += share * pixel.item()
            terms["front"] += share * front.item()
        loss.backward()
        terms["loss"] += loss.item()
    all_mean(*(p.grad for p in params if p.grad is not None))
    if cfg.train.max_grad_norm:
        torch.nn.utils.clip_grad_norm_(params, cfg.train.max_grad_norm)
    opt.step()
    sched.step()
    return terms


def validate(dit, cond, adapter, vaes, val_loaders, train_loaders, gammas, cfg: DitStageConfig, close=None) -> dict:
    """Rollout metrics on the val sims plus, as train_*, the means on a few training sims (the gap = overfitting),
    and with `close` the closeness to the teacher on its fixed batch."""
    metrics, _ = evaluate(dit, cond, adapter, vaes, val_loaders, gammas, cfg)
    train, _ = evaluate(dit, cond, adapter, vaes, train_loaders, gammas, cfg)
    train = {f"train_{k}": v for k, v in train.items() if "gamma" not in k and k not in ("vrmse_per_frame", "per_sim")}
    return {k: v for k, v in metrics.items() if k != "per_sim"} | train | (close() if close else {})


def probe_batch(cfg: DitStageConfig, ids: list[str], device: torch.device) -> list[tuple]:
    """The fixed batch on which the closeness to the teacher is measured: the first distill.probe_sims val sims at
    the sigmas of a 4-step chain, one fixed noise; micro-batches of (x_σ, sigma, gamma)."""
    n = cfg.distill.probe_sims if cfg.distill else 8
    data = LatentDataset(cfg.latents.dir, "val", ids[:n], cfg.latents.frames)
    noise, out = torch.Generator().manual_seed(0), []
    for latents, gamma in DataLoader(data, cfg.train.micro_batch):
        z = latents.flatten(1, 2)
        for level in (1.0, 0.9, 0.75, 0.5):
            sigma = (torch.arange(z.shape[2]) >= cfg.cond.context).float().expand(len(z), -1) * level
            out.append((noised(z, torch.randn(z.shape, generator=noise), sigma).to(device), sigma.to(device), gamma))
    return [(x, sigma, gamma.to(device)) for x, sigma, gamma in out]


@torch.no_grad()
def probe_velocity(cfg: DitStageConfig, velocity, probe: list[tuple]) -> torch.Tensor:
    """`velocity(x, sigma, gamma)` on every micro-batch of the probe."""
    with autocast(cfg.model.precision, probe[0][0].device):
        return torch.cat([velocity(*batch) for batch in probe])


def n_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def prune_model(cfg: DitStageConfig, dit: WanDiT, cond, batches, probe: list[tuple], before: dict | None) -> tuple:
    """Cut the loaded transformer to cfg.prune: statistics on train sims (GT-noised and sampler-chain states, as in
    training), importance, the cut with the least-squares refit. Returns the spec (composed with the loaded run's)
    and a report: parameters, block-drop errors and the closeness to the unpruned model with and without the refit."""
    p, t, mb, start = cfg.prune, dit.transformer, cfg.train.micro_batch, time.time()
    inputs = []
    while len(inputs) * mb < p.calib_sims // world:
        latents, gamma, *_ = next(batches)
        z_all, gamma = latents.flatten(1, 2).to(dit.device), gamma.to(dit.device)
        for z, g in zip(z_all.split(mb), gamma.split(mb), strict=True):
            inputs.append(states(cfg, dit, cond, z, g)[:2] + (g,))

    def forward(x: torch.Tensor, sigma: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        return dit.velocity(x, sigma * 1000, cond(gamma))

    with autocast(cfg.model.precision, dit.device):
        stats = calibrate(t, p, lambda batch: forward(*batch), inputs)
    all_sum(*(stats["grams"][k] for k in sorted(stats["grams"])), stats["residual"], stats["drop"])
    scores = importance(t, p, stats)
    local = choose(t, p, scores)
    unpruned, report = probe_velocity(cfg, forward, probe), {"params_before": n_params(t)}
    variants = [(False, copy.deepcopy(t) if p.compensate else t)] + [(True, t)] * p.compensate  # plain on a copy
    for refit, cut in variants:
        shrink(cut, local | {"bias": local["bias"] and refit}, scores, refit)
        dit.transformer = cut
        report[f"close_{'refit' if refit else 'plain'}"] = closeness(probe_velocity(cfg, forward, probe), unpruned)
    if p.compensate and report["close_refit"]["teacher_rel_mse"] > report["close_plain"]["teacher_rel_mse"]:
        t = variants[0][1]  # the refit did not help on the probe: the plain cut goes on
        local["bias"] = False
    report["refit_kept"] = p.compensate and t is not variants[0][1]
    dit.transformer = t
    del variants
    sync(t)  # the cut is deterministic; guard against rank-wise rounding in the solves anyway
    report |= {"params": n_params(dit.transformer), "blocks": local["blocks"], "seconds": round(time.time() - start)}
    if p.depth and p.blocks == "scan":
        report["block_drop"] = scores["drop"]
    print(f"pruned to {report['params'] / 1e6:.1f} M parameters: {report}")
    return compose(before, local), report


def as_raw(metrics: dict) -> dict:
    """The raw weights' metrics beside the EMA's."""
    return {f"raw_{k}": v for k, v in metrics.items()}


@torch.no_grad()
def write_rollouts(cfg: DitStageConfig, dit: WanDiT, cond, splits: dict) -> None:
    """Every sim's rollout latents (the evaluation's noise seeds, so the test rollouts are the evaluated ones) to
    cfg.rollouts/<split>/<sim id>.pt, with the latent folder's encode.yaml."""
    out = Path(cfg.rollouts)
    for split, (h5, ids) in splits.items():
        (out / split).mkdir(parents=True, exist_ok=True)
        loader = eval_loaders(cfg, split, h5, ids)[0]
        names, seen = loader.dataset.ids, 0
        for z, gamma in loader:
            batch = names[seen : seen + len(gamma)]
            seeds = noise_seeds(batch, cfg.sampling.seed)
            pred = rollout(dit, cond, z.to(dit.device), gamma.to(dit.device), cfg, seeds).cpu()
            for sid, p, g in zip(batch, pred, gamma.tolist(), strict=True):
                tmp = out / split / f"{sid}.tmp"  # renamed when complete: a killed job leaves no broken .pt
                torch.save({"latents": p.clone(), "gamma": g}, tmp)
                tmp.rename(tmp.with_suffix(".pt"))
            seen += len(pred)
        print(f"{split}: {len(names)} rollouts on rank {rank}", flush=True)
    if rank == 0:
        shutil.copy(Path(cfg.latents.dir) / "encode.yaml", out / "encode.yaml")


def save_run(run_dir: Path, dit: WanDiT, cond, weights_dir: str | None, step: int | None = None) -> None:
    """Final weights as dit.pt and cond.pt; with `step`, dit_<step>.pt and cond_<step>.pt (every eval point stays)."""
    save_state(run_dir, f"dit{tag(step)}.pt", dit.transformer.state_dict(), weights_dir)
    if list(cond.parameters()):
        save_state(run_dir, f"cond{tag(step)}.pt", cond.state_dict(), weights_dir)
    if dit.noise is not None:
        torch.save(dit.noise.detach().cpu(), run_dir / f"noise{tag(step)}.pt")  # 1.7 MB: in the run folder, gitignored


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
    (train_h5, train_ids), (val_h5, val_ids), (test_h5, test_ids) = sets["train"], sets["val"], sets["test"]
    eval_ids, test_ids = val_ids[: cfg.data.eval_sims], test_ids[: cfg.data.test_sims]
    train_eval_ids = train_ids[: cfg.data.eval_sims // 2]  # one per GCD at 8 ranks: costs as much time as a single sim
    if min(len(eval_ids), len(train_eval_ids), len(test_ids)) < world:
        raise typer.BadParameter(f"every one of the {world} processes needs a sim to evaluate: raise data.eval_sims")
    gammas = sorted({round(parse_gamma(sid), 3) for sid in eval_ids + test_ids + train_eval_ids})
    batches = endless(train_loader(cfg, train_ids))
    val_loaders = eval_loaders(cfg, "val", val_h5, eval_ids)
    train_eval_loaders = eval_loaders(cfg, "train", train_h5, train_eval_ids)
    test_loaders = eval_loaders(cfg, "test", test_h5, test_ids)
    print(
        f"{len(train_ids)} train ({len(train_eval_ids)} evaluated) / {len(eval_ids)} val / {len(test_ids)} test sims, "
        f"latents from {cfg.latents.dir}, {world} x {device}"
    )

    adapter, vaes, dit, cond = build(cfg, device)
    loaded = Path(cfg.load or "") / "prune.json"
    spec = json.loads(loaded.read_text())["spec"] if cfg.load and loaded.exists() else None
    probe = probe_batch(cfg, val_ids, device) if cfg.prune or cfg.distill else []
    prune_report = {}
    if cfg.prune:
        spec, prune_report = prune_model(cfg, dit, cond, batches, probe, spec)
    if spec and rank == 0:  # a pruned run keeps its shapes beside its weights
        (run_dir / "prune.json").write_text(json.dumps({"spec": spec, "report": prune_report}, indent=1))
    teacher, close = None, None
    if cfg.distill:
        teacher_dit, teacher_cond = load_dit(cfg, device, cfg.distill.teacher, None, len(adapter.groups), False)
        teacher = Teacher(teacher_dit, teacher_cond, dit, spec, cfg.distill.features > 0)
        reference = probe_velocity(cfg, teacher.velocity, probe)

        def close() -> dict:
            mine = probe_velocity(cfg, lambda x, sigma, g: dit.velocity(x, sigma * 1000, cond(g)), probe)
            return closeness(mine, reference)

        sizes = n_params(dit.transformer) / 1e6, n_params(teacher_dit.transformer) / 1e6
        print("student {:.1f} M, teacher {:.1f} M parameters".format(*sizes))
    if cfg.latents.delta:  # the delta scale of the run continued from, else measured on the train latents
        loaded = Path(cfg.load or "") / "delta.pt"
        if cfg.load and loaded.exists():
            dit.delta_scale = torch.load(loaded, map_location=device)
        else:
            sims = LatentDataset(cfg.latents.dir, "train", shard(train_ids), cfg.latents.frames)
            dit.delta_scale = scales(DataLoader(sims, 8, num_workers=cfg.data.num_workers), device)
        if rank == 0:  # with the run from the start: every eval point (load_step) uses the same scale
            torch.save(dit.delta_scale.cpu(), run_dir / "delta.pt")
        first, later = dit.delta_scale.median(1).values.tolist()
        print(f"delta scales (median over channels): into latent 1 {first:.3g}, later {later:.3g}")
    if cfg.rollouts:
        write_rollouts(
            cfg, dit, cond, {"test": (test_h5, test_ids), "val": (val_h5, eval_ids), "train": (train_h5, train_ids)}
        )
        return
    init = validate(dit, cond, adapter, vaes, val_loaders, train_eval_loaders, gammas, cfg, close)
    log_metrics("init", init)

    groups = [{"params": dit.trainable_params, "lr": cfg.train.lr}]
    if dit.patchify_params:
        groups.append({"params": dit.patchify_params, "lr": cfg.train.patchify_lr or cfg.train.lr})
    if list(cond.parameters()):
        groups.append({"params": list(cond.parameters()), "lr": cfg.train.lr})
    if cfg.noise == "learned":
        groups.append({"params": [dit.noise], "lr": cfg.train.noise_lr})
    opt, sched = adamw(groups, cfg.train.steps, cfg.train.warmup_steps, cfg.train.lr_floor, cfg.train.weight_decay)
    params = [p for g in groups for p in g["params"]]
    moments = Path(cfg.load or "") / "opt.pt"
    if cfg.train.steps and cfg.load and cfg.load_step is None and moments.exists():  # lrs and schedule are this run's
        state = torch.load(moments, map_location=device)
        shapes = [s["exp_avg"].shape for _, s in sorted(state["state"].items())]
        if shapes == [p.shape for p in params]:
            opt.load_state_dict(state | {"param_groups": opt.state_dict()["param_groups"]})
        else:  # load_state_dict matches by position: other parameters would get foreign moments
            print(f"{moments} holds the moments of other parameters: AdamW starts fresh")
    ema = EMA(params, cfg.train.ema) if cfg.train.ema and cfg.train.steps else None  # eval only: nothing to average
    steps, history = [], []  # per update / per evaluation
    aux = [k for k in ("pixel", "front") if getattr(cfg.train, f"{k}_loss")]  # extra terms in the progress line
    aux += ["flow", "kd_velocity", "kd_features"] if teacher else []
    metrics = init
    t0 = t_log = time.time()
    for step in range(1, cfg.train.steps + 1):
        latents, gamma, *detail = next(batches)
        lr = opt.param_groups[0]["lr"]  # this update's: the cosine makes a flat curve ambiguous without it
        terms = train_step(
            cfg,
            dit,
            cond,
            adapter,
            vaes,
            latents.to(device),
            gamma.to(device),
            opt,
            sched,
            params,
            teacher,
            detail[0].to(device) if detail else None,
        )
        if ema:
            ema.update()
        steps.append({"step": step, "lr": lr, **terms})
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
                [],
                per_step * (cfg.train.steps - step),
                f"{per_step:.1f} s/update "
                + " ".join(f"{k} {sum(s[k] for s in recent) / len(recent):.4f}" for k in aux),
            )
        if step % cfg.train.eval_every == 0 or step == cfg.train.steps:
            raw = validate(dit, cond, adapter, vaes, val_loaders, train_eval_loaders, gammas, cfg, close) if ema else {}
            if raw:
                log_metrics(f"raw {step}", raw)
            if ema:
                ema.swap()
            metrics = validate(dit, cond, adapter, vaes, val_loaders, train_eval_loaders, gammas, cfg, close)
            history.append({"step": step, "lr": lr, "train": terms, **metrics, **as_raw(raw)})
            log_metrics(f"val {step} (lr {lr:.1e})", metrics, terms["loss"], time.time() - t0)
            if rank == 0 and step < cfg.train.steps:  # the last point is the final dit.pt below
                save_run(run_dir, dit, cond, cfg.weights_dir, step)
                (run_dir / "history.json").write_text(json.dumps(history, indent=1))  # the curve so far, if killed
            if ema:
                ema.swap()

    raw = evaluate(dit, cond, adapter, vaes, test_loaders, gammas, cfg)[0] if ema else {}
    if raw:
        log_metrics(f"raw test {len(test_ids)} sims", raw)
    if ema:
        ema.swap()  # from here on the average: test, videos, dit.pt
    final, examples = evaluate(dit, cond, adapter, vaes, test_loaders, gammas, cfg)
    final |= as_raw(raw)
    log_metrics(f"test {len(test_ids)} sims", final, elapsed=time.time() - t0)
    if rank != 0:
        return
    ranked = sorted(final["per_sim"], key=lambda sid: final["per_sim"][sid]["vrmse"])
    picks = {k: ranked[i] for k, i in (("worst", -1), ("best", 0)) if k in cfg.examples}
    examples = {g: e for g, e in examples.items() if "gamma" in cfg.examples}
    if picks:
        loaders = eval_loaders(cfg, "test", test_h5, list(picks.values()), sharded=False)
        sims = render(dit, cond, adapter, vaes, loaders, cfg)
        examples |= {k: sims[sid] for k, sid in picks.items()}
    save_examples(
        examples, adapter.fields, cfg.data.stats_json, run_dir, cfg.weights_dir, "rollout", cfg.data.frame_stride
    )
    save_dit_curves(steps, history, final, adapter.fields, run_dir / "curves.png")
    if cfg.train.steps:
        save_run(run_dir, dit, cond, cfg.weights_dir)
        save_state(run_dir, "opt.pt", opt.state_dict(), cfg.weights_dir)  # AdamW moments: a later run can continue
    record = {
        "config": cfg.model_dump(),
        "provenance": provenance(started, world),
        "init": init,
        "val": metrics,
        "final": final,
        "history": history,
        "steps": steps,
        "n_params": sum(p.numel() for p in params),
        "transformer_params": n_params(dit.transformer),
        "prune": prune_report,
    }
    (run_dir / "metrics.json").write_text(json.dumps(record, indent=1))
    write_index(Path(cfg.results_dir))
    print(f"wrote {run_dir}/ and {Path(cfg.results_dir) / 'README.md'}")


if __name__ == "__main__":
    typer.run(main)
