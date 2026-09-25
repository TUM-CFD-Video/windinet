#!/usr/bin/env python
"""Train and evaluate per-pixel field -> RGB adapters on euler_mq, optionally through the frozen Wan VAE.

    vae: none   x -> A -> A^-1 -> x_hat                       (what does the per-pixel map alone lose?)
    vae: wan    x -> A -> frozen Wan VAE -> A^-1 -> x_hat     (what survives the video VAE?)

adapter.grouping picks 'single' (one 4 -> 3 adapter) or 'triplets' (two exact 3 -> 3 adapters,
each with its own VAE pass). train.steps=0 evaluates an init (or adapter.load=<.pt>) without
training. With ref_stats set, RGB and latent statistics are compared to natural-video reference
statistics (per-channel moment error, Fréchet distance); without it the raw per-channel
mean / var summaries are still reported.

Usage:
    python scripts/field_adapter/train_adapter.py configs/field_adapter/eulermq.yaml \
        --name wan_triplets --set vae=wan --set adapter.grouping=triplets
Writes <results_dir>/<stage>/<name>.{json,png,pt} and appends one row to the ablations table.
"""

from __future__ import annotations

import json
import time
from itertools import cycle
from pathlib import Path

import torch
import typer
import yaml
from torch.utils.data import DataLoader

from windinet.field_adapter.config import FieldAdapterConfig
from windinet.field_adapter.data import FrameSampler, load_field_stats, split_ids
from windinet.field_adapter.grouping import GroupedAdapter, energy_stats
from windinet.field_adapter.latent_stats import (
    IMAGENET_RGB_MEAN,
    IMAGENET_RGB_STD,
    channel_stats,
    coral_loss,
    frechet_distance,
    moment_loss,
    psd_loss,
)
from windinet.losses import h1_seminorm_loss, rmse_loss, vrms_per_channel
from windinet.training.shockwave_data import CHANNEL_NAMES, ShockWaveDataset

app = typer.Typer(pretty_exceptions_enable=False, no_args_is_help=True)


def _apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    for item in overrides:
        key, _, value = item.partition("=")
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(value)
    return cfg


def _as_clip(x: torch.Tensor) -> torch.Tensor:
    """[N, C, H, W] frames -> [1, C, N, H, W], the layout windinet.losses expect (frames pooled as time)."""
    return x.movedim(0, 1).unsqueeze(0)


def _batch(item, device) -> tuple[torch.Tensor, torch.Tensor]:
    """DataLoader item ([B, k, 4, H, W], gamma [B]) -> frames [B*k, 4, H, W], gamma [B*k]."""
    frames, gamma = item
    return frames.flatten(0, 1).to(device), gamma.float().repeat_interleave(frames.shape[1]).to(device)


def round_trip(adapter: GroupedAdapter, x: torch.Tensor, gamma: torch.Tensor, vae):
    """Returns (x_hat, [rgb per group], [latents per group] or [])."""
    rgbs = adapter(x, gamma)
    if vae is None:
        return adapter.inverse(rgbs), rgbs, []
    latents = [vae.encode(rgb.unsqueeze(2)) for rgb in rgbs]  # single frames as 1-frame clips
    return adapter.inverse([vae.decode(z).squeeze(2) for z in latents]), rgbs, latents


def losses(cfg: FieldAdapterConfig, adapter: GroupedAdapter, x, gamma, vae, ref: dict | None) -> tuple[dict, torch.Tensor]:
    x_hat, rgbs, latents = round_trip(adapter, x, gamma, vae)
    z, z_hat = _as_clip(adapter.normalized(x)), _as_clip(adapter.normalized(x_hat))
    out = {"rmse": rmse_loss(z_hat, z), "h1": h1_seminorm_loss(z_hat, z)}
    if cfg.adapter.warp and cfg.loss.slope_penalty:
        out["slope_penalty"] = adapter.slope_penalty()
    if cfg.loss.rgb_moment:
        mean, var = (ref["rgb"]["mean"], ref["rgb"]["var"]) if ref else (IMAGENET_RGB_MEAN, IMAGENET_RGB_STD.square())
        out["rgb_moment"] = sum(moment_loss(r, mean.to(r), var.to(r)) for r in rgbs) / len(rgbs)
    if latents and ref:
        lat, n = ref["latent"], len(latents)
        if cfg.loss.latent_moment:
            out["latent_moment"] = sum(moment_loss(z, lat["mean"].to(z), lat["var"].to(z)) for z in latents) / n
        if cfg.loss.latent_cov:
            out["latent_cov"] = sum(coral_loss(z, lat["cov"].to(z)) for z in latents) / n
        if cfg.loss.latent_psd:
            out["latent_psd"] = sum(psd_loss(z, lat["log_psd"].to(z)) for z in latents) / n
    return out, sum(getattr(cfg.loss, k) * v for k, v in out.items())


def _dist_metrics(prefix: str, samples: list[torch.Tensor], ref: dict | None) -> dict[str, float]:
    """Per-channel mean/var summary of `samples`, plus moment error and Fréchet distance to `ref` if given."""
    s = channel_stats(torch.cat(samples))
    m = {f"{prefix}_mean_abs": s["mean"].abs().mean().item(), f"{prefix}_var": s["var"].mean().item()}
    if ref:
        m[f"{prefix}_moment_err"] = ((s["mean"] - ref["mean"]).square().mean() + (s["var"] - ref["var"]).square().mean()).item()
        m[f"{prefix}_frechet"] = frechet_distance(s["mean"], s["cov"], ref["mean"], ref["cov"]).item()
    return m


@torch.no_grad()
def evaluate(adapter: GroupedAdapter, loader: DataLoader, vae, ref: dict | None, device) -> tuple[dict[str, float], tuple]:
    """Per-field VRMSE (physical units) over the held-out sims, distribution metrics per group, one panel example."""
    vrmse, n, example = torch.zeros(len(CHANNEL_NAMES), device=device), 0, None
    rgb_all, lat_all = [[] for _ in adapter.groups], [[] for _ in adapter.groups]
    for item in loader:
        x, gamma = _batch(item, device)
        x_hat, rgbs, latents = round_trip(adapter, x, gamma, vae)
        vrmse += vrms_per_channel(_as_clip(x_hat), _as_clip(x))
        n += 1
        example = example or (x[0].cpu(), x_hat[0].cpu(), [r[0].cpu() for r in rgbs])
        for g, rgb in enumerate(rgbs):
            rgb_all[g].append(rgb[:, :, ::4, ::4].cpu())  # subsampled sites are plenty for 3x3 statistics
        for g, z in enumerate(latents):
            lat_all[g].append(z.cpu())
    metrics = {f"vrmse_{name}": (vrmse[i] / n).item() for i, name in enumerate(CHANNEL_NAMES)}
    for g in range(len(adapter.groups)):
        tag = "" if len(adapter.groups) == 1 else "AB"[g]
        metrics.update(_dist_metrics(f"rgb{tag}", rgb_all[g], ref and ref["rgb"]))
        if lat_all[g]:
            metrics.update(_dist_metrics(f"latent{tag}", lat_all[g], ref and ref["latent"]))
    return metrics, example


def save_panel(example: tuple, path: Path) -> None:
    """GT / reconstruction / residual per field, plus the RGB image(s) the adapter feeds the VAE. Small PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x, x_hat, rgbs = example
    fig, axes = plt.subplots(4, 4, figsize=(12, 11))
    for i, name in enumerate(CHANNEL_NAMES):
        lo, hi = x[i].min().item(), x[i].max().item()
        for j, (img, title, kw) in enumerate([(x[i], f"{name} GT", dict(vmin=lo, vmax=hi)),
                                              (x_hat[i], "recon", dict(vmin=lo, vmax=hi)),
                                              (x_hat[i] - x[i], "residual", dict(cmap="RdBu_r"))]):
            im = axes[i, j].imshow(img, **kw)
            axes[i, j].set_title(title, fontsize=9)
            axes[i, j].axis("off")
            plt.colorbar(im, ax=axes[i, j], fraction=0.046)
        axes[i, 3].axis("off")
    for g, rgb in enumerate(rgbs):
        axes[g, 3].imshow((rgb.permute(1, 2, 0) + 1) / 2)
        axes[g, 3].set_title(f"adapter RGB {'AB'[g] if len(rgbs) > 1 else ''}", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=70)
    plt.close(fig)


def append_row(cfg: FieldAdapterConfig, metrics: dict[str, float], stage: str) -> None:
    weights = ", ".join(f"{k}={v:g}" for k, v in cfg.loss.model_dump().items() if v) if cfg.train.steps else "init only"
    extra = ", ".join(f"{k}={v:.3g}" for k, v in metrics.items() if not k.startswith("vrmse_"))
    cells = [stage, cfg.name, cfg.vae, f"{cfg.adapter.grouping}, {cfg.adapter.init}, warp={cfg.adapter.warp}", weights]
    cells += [f"{metrics[f'vrmse_{n}']:.4f}" for n in CHANNEL_NAMES] + [extra]
    with open(cfg.ablations_md, "a") as fh:
        fh.write("| " + " | ".join(cells) + " |\n")


@app.command()
def main(
    config_path: Path = typer.Argument(..., help="YAML matching FieldAdapterConfig"),
    name: str | None = typer.Option(None, help="run name (overrides the YAML's)"),
    set_: list[str] = typer.Option([], "--set", help="dotted override, e.g. adapter.warp=false"),
) -> None:
    raw = _apply_overrides(yaml.safe_load(config_path.read_text()), set_)
    if name:
        raw["name"] = name
    cfg = FieldAdapterConfig(**raw)
    torch.manual_seed(cfg.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg.results_dir) / ("10_compression" if cfg.vae == "none" else "20_wan_roundtrip")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ids, test_ids = split_ids(ShockWaveDataset(cfg.data.h5_path).ids, cfg.data.test_every)

    def make_loader(ids, seed):  # seed=None: random frames every access (training); fixed seed: same frames (eval)
        sampler = FrameSampler(cfg.data.h5_path, ids, cfg.data.frames_per_sim, seed)
        return DataLoader(sampler, batch_size=cfg.train.batch_sims, shuffle=seed is None,
                          num_workers=cfg.data.num_workers, persistent_workers=cfg.data.num_workers > 0)

    train_loader, test_loader = make_loader(train_ids, None), make_loader(test_ids, 0)
    print(f"{len(train_ids)} train / {len(test_ids)} test sims, device {device}")

    batches = cycle(train_loader)
    init = [_batch(next(batches), device) for _ in range(4)]
    x0, gamma0 = torch.cat([b[0] for b in init]), torch.cat([b[1] for b in init])
    stats = load_field_stats(cfg.data.stats_json)
    stats["log_energy"] = energy_stats(x0, gamma0)  # derived field, only the triplet grouping uses it
    adapter = GroupedAdapter(cfg.adapter.grouping, stats, bins=cfg.adapter.bins, warp=cfg.adapter.warp).to(device)
    if cfg.adapter.load:
        adapter.load_state_dict(torch.load(cfg.adapter.load))
    else:
        adapter.init_from_data(x0, gamma0, mix=cfg.adapter.init)
    vae = None
    if cfg.vae == "wan":
        from windinet.wan.vae import WanVAE

        vae = WanVAE(device=device)
    ref = torch.load(cfg.ref_stats) if cfg.ref_stats else None
    metrics, example = evaluate(adapter, test_loader, vae, ref, device)
    metrics_init = dict(metrics)
    print("init:", {k: round(v, 4) for k, v in metrics.items()})

    opt = torch.optim.Adam(adapter.parameters(), lr=cfg.train.lr)
    history, t0 = [], time.time()
    for step in range(1, cfg.train.steps + 1):
        x, gamma = _batch(next(batches), device)
        opt.zero_grad(set_to_none=True)
        chunk = 1 if vae else len(x)  # backprop through the VAE costs ~3.5 GB per frame: one frame at a time
        for xi, gi in zip(x.split(chunk), gamma.split(chunk)):
            terms, total = losses(cfg, adapter, xi, gi, vae, ref)
            (total * len(xi) / len(x)).backward()
        opt.step()
        if step % cfg.train.eval_every == 0 or step == cfg.train.steps:
            metrics, example = evaluate(adapter, test_loader, vae, ref, device)
            history.append({"step": step, "train": {k: v.item() for k, v in terms.items()}, **metrics})
            print(f"step {step:5d}  loss {total.item():.4f}  " + "  ".join(f"{k} {v:.4f}" for k, v in metrics.items())
                  + f"  [{time.time() - t0:.0f}s]")

    stem = out_dir / cfg.name
    save_panel(example, stem.with_suffix(".png"))
    torch.save(adapter.state_dict(), stem.with_suffix(".pt"))
    stem.with_suffix(".json").write_text(json.dumps({"config": cfg.model_dump(), "stats": stats, "init": metrics_init,
                                                     "final": metrics, "history": history,
                                                     "n_params": sum(p.numel() for p in adapter.parameters())}, indent=1))
    append_row(cfg, metrics, out_dir.name)
    print(f"wrote {stem}.{{json,png,pt}}")


if __name__ == "__main__":
    app()
