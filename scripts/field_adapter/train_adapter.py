#!/usr/bin/env python
"""Train and evaluate the per-pixel field -> RGB adapter on euler_mq through the frozen Wan VAE.

    vae: wan    x -> A -> frozen Wan VAE -> A^-1 -> x_hat     (the real round trip)
    vae: none   x -> A -> A^-1 -> x_hat                       (adapter alone, sanity check)

train.steps=0 evaluates an init (or adapter.load=<.pt>) without training. With ref_stats
set, the latent statistics of each group are compared to the natural-video reference
(Fréchet distance).

Usage:
    python scripts/field_adapter/train_adapter.py configs/field_adapter/eulermq.yaml \
        --name pairs_600 --set train.steps=600
Writes results/field_adapter/<stage>/<name>.{json,png,pt} and appends one row to the ablations table.
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

from windinet.field_adapter import GroupedAdapter
from windinet.field_adapter.config import FieldAdapterConfig
from windinet.field_adapter.data import FrameSampler, load_field_stats, split_ids
from windinet.field_adapter.latent_stats import channel_stats, frechet_distance
from windinet.losses import h1_seminorm_loss, rmse_loss, vrms_per_channel
from windinet.training.shockwave_data import CHANNEL_NAMES, ShockWaveDataset

RESULTS = Path("results/field_adapter")
ABLATIONS_MD = Path("docs/field_adapter/ablations.md")


def _as_clip(x: torch.Tensor) -> torch.Tensor:
    """[N, C, H, W] frames -> [1, C, N, H, W], the layout windinet.losses expect (frames pooled as time)."""
    return x.movedim(0, 1).unsqueeze(0)


def round_trip(adapter: GroupedAdapter, x: torch.Tensor, vae):
    """Returns (x_hat, [rgb per group], [latents per group]); latents is empty without a VAE."""
    rgbs = adapter(x)
    if vae is None:
        return adapter.inverse(rgbs), rgbs, []
    latents = [vae.encode(rgb.unsqueeze(2)) for rgb in rgbs]  # single frames as 1-frame clips
    return adapter.inverse([vae.decode(z).squeeze(2) for z in latents]), rgbs, latents


def losses(cfg: FieldAdapterConfig, adapter: GroupedAdapter, x: torch.Tensor, vae) -> tuple[dict, torch.Tensor]:
    x_hat, _, _ = round_trip(adapter, x, vae)
    z, z_hat = _as_clip(adapter.normalized(x)), _as_clip(adapter.normalized(x_hat))
    terms = {"rmse": rmse_loss(z_hat, z), "h1": h1_seminorm_loss(z_hat, z)}
    return terms, cfg.loss.rmse * terms["rmse"] + cfg.loss.h1 * terms["h1"]


@torch.no_grad()
def evaluate(adapter: GroupedAdapter, loader: DataLoader, vae, ref: dict | None, device) -> tuple[dict[str, float], tuple]:
    """Per-field VRMSE (physical units) over the held-out sims, latent Fréchet per group, one panel example."""
    vrmse, n, example = 0, 0, None
    latents_all = [[] for _ in adapter.groups]
    for frames, _ in loader:
        x = frames.flatten(0, 1).to(device)
        x_hat, rgbs, latents = round_trip(adapter, x, vae)
        vrmse, n = vrmse + vrms_per_channel(_as_clip(x_hat), _as_clip(x)), n + 1
        example = example or (x[0].cpu(), x_hat[0].cpu(), [r[0].cpu() for r in rgbs])
        for g, z in enumerate(latents):
            latents_all[g].append(z.cpu())
    metrics = {f"vrmse_{name}": (vrmse[i] / n).item() for i, name in enumerate(CHANNEL_NAMES)}
    if ref and latents_all[0]:
        for g, zs in enumerate(latents_all):
            metrics[f"latent{'ABC'[g]}_frechet"] = frechet_distance(channel_stats(torch.cat(zs)), ref["latent"]).item()
    return metrics, example


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


def append_row(cfg: FieldAdapterConfig, metrics: dict[str, float], stage: str) -> None:
    groups = " + ".join("(" + ",".join(g) + ")" for g in cfg.adapter.groups)
    train = f"lr={cfg.train.lr:g}, {cfg.train.steps} steps" if cfg.train.steps else "init only"
    frechet = ", ".join(f"{k}={v:.3g}" for k, v in metrics.items() if "frechet" in k)
    cells = [stage, cfg.name, cfg.vae, groups, train, *(f"{metrics[f'vrmse_{n}']:.4f}" for n in CHANNEL_NAMES), frechet]
    with open(ABLATIONS_MD, "a") as fh:
        fh.write("| " + " | ".join(cells) + " |\n")


def main(
    config_path: Path = typer.Argument(..., help="YAML matching FieldAdapterConfig"),
    name: str | None = typer.Option(None, help="run name (overrides the YAML's)"),
    set_: list[str] = typer.Option([], "--set", help="dotted override, e.g. train.steps=0"),
) -> None:
    raw = yaml.safe_load(config_path.read_text())
    for item in set_:  # "a.b=c" -> raw["a"]["b"] = yaml(c)
        key, _, value = item.partition("=")
        *parents, leaf = key.split(".")
        node = raw
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(value)
    cfg = FieldAdapterConfig(**(raw | {"name": name} if name else raw))
    torch.manual_seed(cfg.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = RESULTS / ("10_compression" if cfg.vae == "none" else "20_wan_roundtrip")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ids, test_ids = split_ids(ShockWaveDataset(cfg.data.h5_path).ids, cfg.data.test_every)

    def make_loader(ids, seed):  # seed=None: random frames every access (training); fixed seed: same frames (eval)
        sampler = FrameSampler(cfg.data.h5_path, ids, cfg.data.frames_per_sim, seed)
        return DataLoader(sampler, batch_size=cfg.train.batch_sims, shuffle=seed is None,
                          num_workers=cfg.data.num_workers, persistent_workers=cfg.data.num_workers > 0)

    train_loader, test_loader = make_loader(train_ids, None), make_loader(test_ids, 0)
    batches = cycle(train_loader)
    print(f"{len(train_ids)} train / {len(test_ids)} test sims, device {device}")

    adapter = GroupedAdapter(cfg.adapter.groups, load_field_stats(cfg.data.stats_json)).to(device)
    if cfg.adapter.load:
        adapter.load_state_dict(torch.load(cfg.adapter.load))
    else:
        adapter.init_from_data(torch.cat([next(batches)[0].flatten(0, 1) for _ in range(4)]).to(device))
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
        x = next(batches)[0].flatten(0, 1).to(device)
        opt.zero_grad(set_to_none=True)
        for xi in x.split(1 if vae else len(x)):  # backprop through the VAE costs ~3.5 GB per frame
            terms, total = losses(cfg, adapter, xi, vae)
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
    stem.with_suffix(".json").write_text(json.dumps({"config": cfg.model_dump(), "init": metrics_init, "final": metrics,
                                                     "history": history, "n_params": sum(p.numel() for p in adapter.parameters())}, indent=1))
    append_row(cfg, metrics, out_dir.name)
    print(f"wrote {stem}.{{json,png,pt}}")


if __name__ == "__main__":
    typer.run(main)
