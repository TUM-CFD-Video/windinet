"""The VAE-stage round trip fields -> adapter -> Wan VAE -> adapter^-1 -> fields: training loss and evaluation."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from windinet.experiment.distributed import all_sum
from windinet.experiment.latent_stats import ChannelMoments, frechet_distance
from windinet.experiment.metrics import clipped, vrmse
from windinet.experiment.runs import group_tag
from windinet.field_adapter import GroupedAdapter
from windinet.losses import h1_seminorm_loss, rmse_loss
from windinet.wan.vae_stage import VaeStageConfig


def round_trip(adapter: GroupedAdapter, x: torch.Tensor, vaes: list, clip: int = 1):
    """Returns (x_hat, [rgb per group], [latents per group]); latents is empty without a VAE.

    `x` holds [N, F, H, W] frames; a VAE sees them as N / clip clips of `clip` frames (1 = single frames).
    `vaes`: one WanVAE shared by every group, or one per group (train.vae_per_group); empty = adapter only."""
    rgbs = adapter(x)
    if not vaes:
        return adapter.inverse(rgbs), rgbs, []
    per_group = [vaes[g % len(vaes)] for g in range(len(rgbs))]
    latents = [v.encode(rgb.unflatten(0, (-1, clip)).movedim(1, 2)) for v, rgb in zip(per_group, rgbs, strict=True)]
    decoded = [v.decode(z).movedim(2, 1).flatten(0, 1) for v, z in zip(per_group, latents, strict=True)]
    return adapter.inverse(decoded), rgbs, latents


def clip_len(cfg: VaeStageConfig) -> int:
    return cfg.data.frames_per_sim if cfg.data.clip else 1


def _as_clip(x: torch.Tensor) -> torch.Tensor:
    """[N, F, H, W] frames -> [1, F, N, H, W], the layout windinet.losses expect (frames pooled as time)."""
    return x.movedim(0, 1).unsqueeze(0)


def losses(cfg: VaeStageConfig, adapter: GroupedAdapter, x: torch.Tensor, vaes: list) -> tuple[dict, torch.Tensor]:
    x_hat, _, _ = round_trip(adapter, x, vaes, clip_len(cfg))
    z, z_hat = _as_clip(adapter.normalized(x)), _as_clip(adapter.normalized(x_hat))
    terms = {"rmse": rmse_loss(z_hat, z), "h1": h1_seminorm_loss(z_hat, z)}
    return terms, cfg.loss.rmse * terms["rmse"] + cfg.loss.h1 * terms["h1"]


@torch.no_grad()
def evaluate(
    adapter: GroupedAdapter, loader: DataLoader, vaes: list, ref: dict | None, norm: tuple, clip: int | None
) -> tuple[dict[str, float], tuple]:
    """Per-field VRMSE per simulation, averaged over simulations, in physical units and in the LTX baseline's
    5-sigma clipped space (`norm` = field means and stds); latent Fréchet per group; one panel example.

    The loader yields one simulation at a time, so the variance normalisation is that simulation's own.
    `clip`: frames per VAE pass; None = a simulation's frames as one clip (whole trajectories).
    Under torchrun each rank evaluates its shard of the loader and the sums are reduced over ranks."""
    device = norm[0].device
    total, total_clipped = torch.zeros(2, len(adapter.fields), device=device)
    n = torch.zeros((), device=device)  # a tensor so it can be all-reduced
    example = None
    moments = [ChannelMoments(len(ref["latent"]["mean"]), device) for _ in adapter.groups] if ref and vaes else []
    for frames, _ in loader:
        x = adapter.select(frames.flatten(0, 1).to(device))
        x_hat, rgbs, latents = round_trip(adapter, x, vaes, clip or len(x))
        total += vrmse(x_hat, x)
        total_clipped += vrmse(clipped(x_hat, *norm), clipped(x, *norm))
        n += 1
        mid = len(x) // 2  # frame 0 is the piecewise-constant initial condition; the middle shows developed flow
        example = example or (x[mid].cpu(), x_hat[mid].cpu(), [r[mid].cpu() for r in rgbs])
        for m, z in zip(moments, latents, strict=False):  # moments is empty without ref
            m.add(z)
    all_sum(total, total_clipped, n, *[t for m in moments for t in m.sums()])
    metrics = {}
    for tag, per_field in (("vrmse", total / n), ("vrmse_clipped", total_clipped / n)):
        metrics |= {f"{tag}_{name}": v.item() for name, v in zip(adapter.fields, per_field, strict=True)}
        metrics[f"{tag}_mean"] = per_field.mean().item()
    for g, m in enumerate(moments):
        stats = {k: v.cpu() for k, v in m.stats().items()}
        metrics[f"latent{group_tag(g)}_frechet"] = frechet_distance(stats, ref["latent"]).item()
    return metrics, example
