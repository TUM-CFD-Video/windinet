#!/usr/bin/env python
"""Train and evaluate the per-pixel field -> RGB adapter on euler_mq through the frozen Wan VAE.

    vae: wan    x -> A -> frozen Wan VAE -> A^-1 -> x_hat     (the real round trip)
    vae: none   x -> A -> A^-1 -> x_hat                       (adapter alone, sanity check)

train.steps=0 evaluates an init (or adapter.load=<.pt>) without training. With ref_stats
set, the latent statistics of each group are compared to the natural-video reference
(Fréchet distance).

Usage:
    python scripts/field_adapter/train_adapter.py configs/field_adapter/eulermq.yaml \
        --name pairs_600 --desc "pairs, 600 steps" --set train.steps=600
Every run needs a name and a one-sentence description. Writes
results/field_adapter/<stage>/<name>/{config.yaml, metrics.json, panel.png, curves.png, adapter.pt[, vae.pt]}
and regenerates the stage's README.md index. config.yaml is the resolved config and re-runs as is;
metrics.json carries the provenance (commit, command, versions, GPU, wall time).
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from itertools import cycle
from pathlib import Path

os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
)  # fine-tuning fills the 11 GB card; avoids fragmentation OOM

import torch
import typer
import yaml
from torch.utils.data import DataLoader

from windinet.field_adapter import GroupedAdapter
from windinet.field_adapter.config import FieldAdapterConfig, TrainConfig
from windinet.field_adapter.data import FrameSampler, load_field_stats, split_ids
from windinet.field_adapter.latent_stats import channel_stats, frechet_distance
from windinet.field_adapter.runs import provenance, write_config, write_index
from windinet.losses import h1_seminorm_loss, rmse_loss, vrms_per_channel
from windinet.training.shockwave_data import CHANNEL_NAMES, ShockWaveDataset

RESULTS = Path("results/field_adapter")


def _as_clip(x: torch.Tensor) -> torch.Tensor:
    """[N, C, H, W] frames -> [1, C, N, H, W], the layout windinet.losses expect (frames pooled as time)."""
    return x.movedim(0, 1).unsqueeze(0)


def round_trip(adapter: GroupedAdapter, x: torch.Tensor, vae, clip: int = 1):
    """Returns (x_hat, [rgb per group], [latents per group]); latents is empty without a VAE.

    `x` holds [N, 4, H, W] frames; the VAE sees them as N / clip clips of `clip` frames (1 = single frames)."""
    rgbs = adapter(x)
    if vae is None:
        return adapter.inverse(rgbs), rgbs, []
    latents = [vae.encode(rgb.unflatten(0, (-1, clip)).movedim(1, 2)) for rgb in rgbs]  # -> [B, 3, clip, H, W]
    return adapter.inverse([vae.decode(z).movedim(2, 1).flatten(0, 1) for z in latents]), rgbs, latents


def clip_len(cfg: FieldAdapterConfig) -> int:
    return cfg.data.frames_per_sim if cfg.data.clip else 1


def losses(cfg: FieldAdapterConfig, adapter: GroupedAdapter, x: torch.Tensor, vae) -> tuple[dict, torch.Tensor]:
    x_hat, _, _ = round_trip(adapter, x, vae, clip_len(cfg))
    z, z_hat = _as_clip(adapter.normalized(x)), _as_clip(adapter.normalized(x_hat))
    terms = {"rmse": rmse_loss(z_hat, z), "h1": h1_seminorm_loss(z_hat, z)}
    return terms, cfg.loss.rmse * terms["rmse"] + cfg.loss.h1 * terms["h1"]


def lr_factor(t: TrainConfig):
    """Multiplier on every group's lr: linear warm-up from 1 %, then cosine decay to `lr_floor`; constant without."""

    def factor(step: int) -> float:
        if not t.warmup_steps:
            return 1.0
        if step < t.warmup_steps:
            return 0.01 + 0.99 * step / t.warmup_steps
        progress = (step - t.warmup_steps) / max(1, t.steps - t.warmup_steps)
        return t.lr_floor + (1 - t.lr_floor) * (1 + math.cos(math.pi * progress)) / 2

    return factor


@torch.no_grad()
def evaluate(
    adapter: GroupedAdapter, loader: DataLoader, vae, ref: dict | None, device, clip: int = 1
) -> tuple[dict[str, float], tuple]:
    """Per-field VRMSE (physical units) per held-out sim, averaged; latent Fréchet per group; one panel example.

    The loader yields one sim at a time so the variance normalisation is per sim whatever the training batch."""
    vrmse, n, example = 0, 0, None
    latents_all = [[] for _ in adapter.groups]
    for frames, _ in loader:
        x = frames.flatten(0, 1).to(device)
        x_hat, rgbs, latents = round_trip(adapter, x, vae, clip)
        vrmse, n = vrmse + vrms_per_channel(_as_clip(x_hat), _as_clip(x)), n + 1
        example = example or (x[0].cpu(), x_hat[0].cpu(), [r[0].cpu() for r in rgbs])
        for g, z in enumerate(latents):
            latents_all[g].append(z.cpu())
    metrics = {f"vrmse_{name}": (vrmse[i] / n).item() for i, name in enumerate(CHANNEL_NAMES)}
    if ref and latents_all[0]:
        for g, zs in enumerate(latents_all):
            metrics[f"latent{'ABC'[g]}_frechet"] = frechet_distance(channel_stats(torch.cat(zs)), ref["latent"]).item()
    return metrics, example


def rel_change(params, ref) -> float:
    """||p - ref|| / ||ref|| over a list of tensors: how far weights moved. `ref` lives on the CPU (GPU is full)."""
    num = sum((p.detach().cpu() - r).square().sum() for p, r in zip(params, ref, strict=True))
    return (num / sum(r.square().sum() for r in ref)).sqrt().item()


def drift_per_block(vae, w0) -> dict[str, float]:
    """Relative weight drift from the pretrained VAE, grouped by block (decoder.up_blocks.2, ...)."""
    names = [n for n, p in vae.vae.named_parameters() if p.requires_grad]
    blocks: dict[str, list] = {}
    for n, p, r in zip(names, vae.trainable_params, w0, strict=True):
        blocks.setdefault(".".join(n.split(".")[:3]), []).append((p, r))
    return {b: rel_change([p for p, _ in prs], [r for _, r in prs]) for b, prs in blocks.items()}


def save_curves(steps: list[dict], history: list[dict], blocks: dict[str, float], path: Path) -> None:
    """Train loss per update, validation VRMSE per field, relative weight change per group, drift per block."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    loss = torch.tensor([s["loss"] for s in steps])
    ax[0, 0].semilogy([s["step"] for s in steps], loss, lw=0.5, alpha=0.5)
    if len(loss) >= 10:  # heavy-tailed (shock frames): a running mean shows the trend
        ax[0, 0].semilogy(range(10, len(loss) + 1), loss.unfold(0, 10, 1).mean(1), "k", label="10-update mean")
        ax[0, 0].legend(fontsize=8)
    ax[0, 0].set(title="train loss per update", xlabel="step")
    for name in CHANNEL_NAMES:
        ax[0, 1].plot([h["step"] for h in history], [h[f"vrmse_{name}"] for h in history], marker="o", label=name)
    ax[0, 1].set(title="validation VRMSE", xlabel="step", ylim=(0, None))
    ax[0, 1].legend(fontsize=8)
    for g, label in enumerate(["adapter", "vae"][: len(steps[0]["update_ratio"])] if steps else []):
        ax[1, 0].semilogy(
            [s["step"] for s in steps], [s["update_ratio"][g] for s in steps], lw=0.8, label=f"{label}: step |dw|/|w|"
        )
    if history and "vae_drift" in history[-1]:
        ax[1, 0].semilogy(
            [h["step"] for h in history],
            [h["vae_drift"] for h in history],
            "k--",
            marker="o",
            label="vae: drift from start",
        )
    ax[1, 0].set(title="relative weight change", xlabel="step")
    ax[1, 0].legend(fontsize=8)
    ax[1, 1].barh(list(blocks), list(blocks.values()))
    ax[1, 1].set(title="drift from start weights per block" if blocks else "VAE frozen")
    ax[1, 1].tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def save_panel(example: tuple, path: Path) -> None:
    """Per field: GT / reconstruction / residual; last column the RGB image(s) fed to the VAE. Small PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x, x_hat, rgbs = example
    fig, axes = plt.subplots(4, 4, figsize=(12, 11))
    for i, name in enumerate(CHANNEL_NAMES):
        lo, hi = x[i].min().item(), x[i].max().item()
        panels = [(x[i], f"{name} GT", {}), (x_hat[i], "recon", {}), (x_hat[i] - x[i], "residual", {"cmap": "RdBu_r"})]
        for j, (img, title, kw) in enumerate(panels):
            im = axes[i, j].imshow(img, **({"vmin": lo, "vmax": hi} if j < 2 else kw))
            axes[i, j].set_title(title, fontsize=9)
            plt.colorbar(im, ax=axes[i, j], fraction=0.046)
    for g, rgb in enumerate(rgbs):
        axes[g, 3].imshow((rgb.permute(1, 2, 0) + 1) / 2)
        axes[g, 3].set_title(f"adapter RGB {'ABC'[g]}", fontsize=9)
    for ax in axes.flat:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=70)
    plt.close(fig)


def main(
    config_path: Path = typer.Argument(..., help="YAML matching FieldAdapterConfig"),
    name: str | None = typer.Option(None, help="run name = results folder (overrides the YAML's)"),
    desc: str | None = typer.Option(None, help="one sentence: what this run tests (overrides the YAML's description)"),
    set_: list[str] = typer.Option([], "--set", help="dotted override, e.g. train.steps=0"),
    overwrite: bool = typer.Option(False, help="replace an existing run folder of the same name"),
) -> None:
    started = time.time()
    raw = yaml.safe_load(config_path.read_text())
    for item in set_:  # "a.b=c" -> raw["a"]["b"] = yaml(c)
        key, _, value = item.partition("=")
        *parents, leaf = key.split(".")
        node = raw
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(value)
    cfg = FieldAdapterConfig(**(raw | {k: v for k, v in {"name": name, "description": desc}.items() if v}))
    out_dir = RESULTS / ("10_compression" if cfg.vae == "none" else "20_wan_roundtrip")
    run_dir = out_dir / cfg.name
    if (run_dir / "metrics.json").exists() and not overwrite:
        raise typer.BadParameter(f"{run_dir} exists; pick another --name or pass --overwrite")
    torch.manual_seed(cfg.train.seed)
    random.seed(cfg.train.seed)  # frame sampling in the main process; loader workers derive theirs from torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir.mkdir(parents=True, exist_ok=True)
    write_config(cfg.model_dump(), run_dir / "config.yaml")  # written first: a crashed run still shows what it was

    train_ids, test_ids = split_ids(ShockWaveDataset(cfg.data.h5_path).ids, cfg.data.test_every, cfg.data.test_gamma)
    clip = clip_len(cfg)

    def make_loader(ids, seed):  # seed=None: random frames every access (training); fixed seed: same frames (eval)
        sampler = FrameSampler(cfg.data.h5_path, ids, cfg.data.frames_per_sim, seed, consecutive=cfg.data.clip)
        return DataLoader(
            sampler,
            batch_size=cfg.train.batch_sims if seed is None else 1,
            shuffle=seed is None,
            num_workers=cfg.data.num_workers,
            persistent_workers=cfg.data.num_workers > 0,
        )

    train_loader, test_loader = make_loader(train_ids, None), make_loader(test_ids, 0)
    batches = cycle(train_loader)
    print(f"{len(train_ids)} train / {len(test_ids)} test sims, device {device}")

    adapter = GroupedAdapter(cfg.adapter.groups, load_field_stats(cfg.data.stats_json)).to(device)
    adapter.requires_grad_(cfg.train.lr > 0)  # lr 0 = fixed adapter, no gradient through the encoder for it
    if cfg.adapter.load:
        adapter.load_state_dict(torch.load(cfg.adapter.load))
    else:
        adapter.init_from_data(torch.cat([next(batches)[0].flatten(0, 1) for _ in range(4)]).to(device))
    vae = None
    if cfg.vae == "wan":
        from windinet.wan.vae import WanVAE

        vae = WanVAE(device=device, train=cfg.train.vae_parts)
        if cfg.train.vae_load:
            vae.vae.load_state_dict(torch.load(cfg.train.vae_load), strict=False)
    ref = torch.load(cfg.ref_stats) if cfg.ref_stats else None
    metrics, example = evaluate(adapter, test_loader, vae, ref, device, clip)
    metrics_init = dict(metrics)
    print("init:", {k: round(v, 4) for k, v in metrics.items()})

    groups = [{"params": adapter.parameters(), "lr": cfg.train.lr}]
    if vae and vae.trainable_params:
        groups.append({"params": vae.trainable_params, "lr": cfg.train.vae_lr})
    opt = torch.optim.AdamW(groups, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor(cfg.train))
    params = [p for g in opt.param_groups for p in g["params"]]
    w0 = (
        [p.detach().cpu() for p in vae.trainable_params] if vae else []
    )  # start weights (pretrained or vae_load), for the drift
    history, steps, t0 = [], [], time.time()  # per eval / per update
    for step in range(1, cfg.train.steps + 1):
        x = next(batches)[0].flatten(0, 1).to(device)
        opt.zero_grad(set_to_none=True)
        loss = 0.0
        for xi in x.split(
            clip if vae else len(x)
        ):  # backprop through the VAE costs ~1.5 GB per frame: one clip per backward
            terms, total = losses(cfg, adapter, xi, vae)
            (total * len(xi) / len(x)).backward()
            loss += total.item() * len(xi) / len(x)
        if cfg.train.max_grad_norm:
            torch.nn.utils.clip_grad_norm_(params, cfg.train.max_grad_norm)
        before = [[p.detach().cpu() for p in g["params"]] for g in opt.param_groups]
        opt.step()
        sched.step()
        steps.append(
            {
                "step": step,
                "loss": loss,
                "update_ratio": [rel_change(g["params"], b) for g, b in zip(opt.param_groups, before, strict=True)],
            }
        )
        if step % cfg.train.eval_every == 0 or step == cfg.train.steps:
            metrics, example = evaluate(adapter, test_loader, vae, ref, device, clip)
            if w0:
                metrics["vae_drift"] = rel_change(vae.trainable_params, w0)
            history.append({"step": step, "train": {k: v.item() for k, v in terms.items()}, **metrics})
            print(
                f"step {step:5d}  loss {loss:.4f}  "
                + "  ".join(f"{k} {v:.4f}" for k, v in metrics.items())
                + f"  [{time.time() - t0:.0f}s]"
            )

    save_panel(example, run_dir / "panel.png")
    save_curves(steps, history, drift_per_block(vae, w0) if w0 else {}, run_dir / "curves.png")
    torch.save(adapter.state_dict(), run_dir / "adapter.pt")
    if w0:  # fine-tuned VAE weights (gitignored); reload with train.vae_load
        torch.save(
            {k: v.detach().cpu() for k, v in vae.vae.state_dict(keep_vars=True).items() if v.requires_grad},
            run_dir / "vae.pt",
        )
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "config": cfg.model_dump(),
                "provenance": provenance(started),
                "init": metrics_init,
                "final": metrics,
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
