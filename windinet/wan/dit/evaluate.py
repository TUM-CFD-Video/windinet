"""Evaluation of the DiT's rollouts against the simulations: the per-field metrics per simulation, aggregated,
the VRMSE also per gamma and per frame."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader

from windinet.experiment.distributed import all_sum
from windinet.experiment.metrics import Aggregate
from windinet.field_adapter import GroupedAdapter
from windinet.wan.dit.benchmark import Benchmark
from windinet.wan.dit.config import DitStageConfig
from windinet.wan.dit.model import WanDiT
from windinet.wan.dit.rollout import rollout
from windinet.wan.vae.roundtrip import decode


@torch.no_grad()
def evaluate(
    dit: WanDiT,
    cond: nn.Module,
    adapter: GroupedAdapter,
    vaes: list,
    loaders: tuple[DataLoader, DataLoader],
    gammas: list[float],
    cfg: DitStageConfig,
) -> tuple[dict, dict]:
    """Rollouts of every simulation of `loaders` (latents, fields; same ids, same order) against the simulation:
    the per-field metrics (FIELD_METRICS) per sim, averaged over sims; the VRMSE also with its std over sims, per
    gamma and per frame (The Well's metric, also over its windows); sums reduced over ranks, so every rank needs at
    least one sim. Returns the
    metrics and the examples {gamma: (sim id, truth, pred)} of the lowest and the highest gamma this rank saw,
    as [F, C, H, W]."""
    device = dit.device
    aggregate = Aggregate(adapter.fields, device)
    by_gamma, n_gamma = torch.zeros(len(gammas), device=device), torch.zeros(len(gammas), device=device)  # VRMSE
    ids, seen, examples = loaders[0].dataset.ids, 0, {}
    args = (cfg.data.stats_json, adapter.fields, cfg.data.frame_stride, device)
    bench, vae_bench = (Benchmark(*args), Benchmark(*args, prefix="bench_vae")) if cfg.benchmark else (None, None)
    for (latents, gamma), (frames, _) in zip(*loaders, strict=True):
        gamma = gamma.to(device)
        pred = decode(adapter, rollout(dit, cond, latents.to(device), gamma, cfg), vaes)
        recon = decode(adapter, latents.to(device), vaes) if bench else None  # the round trip: his vae_only row
        truth = adapter.select(frames.to(device).flatten(0, 1)).unflatten(0, (len(frames), -1))
        truth = truth[:, : pred.shape[1]]  # the frames the (possibly truncated) latents cover
        for j, (p, x, g) in enumerate(zip(pred, truth, gamma, strict=True)):
            g = round(g.item(), 3)
            by_gamma[gammas.index(g)] += aggregate.add(p, x)[0].mean()
            n_gamma[gammas.index(g)] += 1
            if bench:
                bench.add(p, x)
                vae_bench.add(recon[j], x)
            if g in (gammas[0], gammas[-1]) and g not in examples:
                examples[g] = (ids[seen + j], x.cpu(), p.cpu())
        seen += len(gamma)
    aggregate.reduce()
    all_sum(by_gamma, n_gamma)
    metrics = aggregate.metrics(cfg.data.frame_stride)
    metrics |= bench.metrics() | vae_bench.metrics() if bench else {}
    metrics |= {f"vrmse_mean_gamma{g:g}": (by_gamma[i] / n_gamma[i]).item() for i, g in enumerate(gammas) if n_gamma[i]}
    return metrics, examples
