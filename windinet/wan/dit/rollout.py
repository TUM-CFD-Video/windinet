"""From latents to fields: the DiT's rollout decoded by the VAE(s) and the adapter inverse, and its evaluation
against the simulations (VRMSE per field, physical and clipped, overall and per gamma)."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader

from windinet.experiment.distributed import all_sum
from windinet.experiment.metrics import clipped, vrmse
from windinet.field_adapter import GroupedAdapter
from windinet.wan.dit.config import DitStageConfig
from windinet.wan.dit.model import WanDiT, autocast


def to_fields(latents: torch.Tensor, adapter: GroupedAdapter, vaes: list) -> torch.Tensor:
    """[B, G, 16, F', h, w] latents -> [B, F, C, H, W] fields: each group through its decoder, then adapter^-1."""
    rgbs = [vaes[g % len(vaes)].decode(latents[:, g]).movedim(2, 1).flatten(0, 1) for g in range(latents.shape[1])]
    return adapter.inverse(rgbs).unflatten(0, (latents.shape[0], -1))


def rollout(
    dit: WanDiT, cond: nn.Module, latents: torch.Tensor, gamma: torch.Tensor, cfg: DitStageConfig
) -> torch.Tensor:
    """Every sample's trajectory from noise, conditioned on `cond(gamma)` and, with cfg.cond.first_frame, on the clean
    first latent frame of `latents` [B, 1, 16, F', h, w]; same shape back."""
    if latents.shape[1] != 1:
        raise NotImplementedError("two latent streams per sample: the pairs DiT is not built yet")
    z, first = latents[:, 0], cfg.cond.first_frame
    with autocast(cfg.model.precision, dit.device):
        out = dit.sample(
            cond(gamma),
            cond.uncond(gamma),
            z.shape,
            cfg.sampling.steps,
            cfg.sampling.guidance,
            cfg.train.seed,
            init=z if first else None,
            keep_first_frame=first,
        )
    return out[:, None]


@torch.no_grad()
def evaluate(
    dit: WanDiT,
    cond: nn.Module,
    adapter: GroupedAdapter,
    vaes: list,
    loaders: tuple[DataLoader, DataLoader],
    gammas: list[float],
    norm: tuple[torch.Tensor, torch.Tensor],
    cfg: DitStageConfig,
) -> tuple[dict[str, float], tuple]:
    """Rollouts of every simulation of `loaders` (latents, fields; same ids, same order) against the simulation:
    VRMSE per field per sim, physical and in the 5-sigma clipped space, averaged over sims and per gamma; sums
    reduced over ranks. Returns the metrics and one example (ground truth, prediction) as [F, C, H, W]."""
    mean, std = norm
    total = torch.zeros(len(gammas), 2, len(adapter.fields), device=mean.device)  # per gamma: physical, clipped
    n = torch.zeros(len(gammas), device=mean.device)
    example = None
    for (latents, gamma), (frames, _) in zip(*loaders, strict=True):
        gamma = gamma.to(mean.device)
        pred = to_fields(rollout(dit, cond, latents.to(mean.device), gamma, cfg), adapter, vaes)
        truth = adapter.select(frames.to(mean.device).flatten(0, 1)).unflatten(0, (len(frames), -1))
        truth = truth[:, : pred.shape[1]]  # the frames the (possibly truncated) latents cover
        for p, x, g in zip(pred, truth, gamma, strict=True):
            i = gammas.index(round(g.item(), 3))
            total[i, 0] += vrmse(p, x)
            total[i, 1] += vrmse(clipped(p, mean, std), clipped(x, mean, std))
            n[i] += 1
        example = example or (truth[0].cpu(), pred[0].cpu())
    all_sum(total, n)
    metrics = {}
    for tag, per_field in zip(("vrmse", "vrmse_clipped"), total.sum(0) / n.sum(), strict=True):
        metrics |= {f"{tag}_{name}": v.item() for name, v in zip(adapter.fields, per_field, strict=True)}
        metrics[f"{tag}_mean"] = per_field.mean().item()
    for i, g in enumerate(gammas):
        if n[i]:
            metrics[f"vrmse_clipped_mean_gamma{g:g}"] = (total[i, 1] / n[i]).mean().item()
    return metrics, example
