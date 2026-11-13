"""Evaluation of the DiT's rollouts against the simulations: the per-field metrics per simulation, aggregated,
the VRMSE also per gamma and per frame; the error split into VAE and DiT; with several samples the ensemble scores."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader

from windinet.experiment.distributed import all_sum, gather
from windinet.experiment.metrics import FIELD_METRICS, Aggregate, ensemble_scores, well_windows
from windinet.field_adapter import GroupedAdapter
from windinet.wan.dit.benchmark import SHARED, Benchmark
from windinet.wan.dit.config import DitStageConfig
from windinet.wan.dit.model import WanDiT
from windinet.wan.dit.rollout import noise_seeds, rollout
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
    the per-field metrics (FIELD_METRICS) of one rollout over the predicted frames per sim, averaged over sims; the
    VRMSE also with its std over sims, per gamma and per frame (The Well's metric, also over its windows; all frames);
    vae_*: the decoded simulation latents against the simulation, dit_*: the rollout against them; with
    sampling.samples > 1, ens_*: the mean of the samples, its spread-skill ratio and fair CRPS; per_sim: each sim's
    metrics (means over fields, the cfd_bench VRMSE). Frames before score_from and the context are not scored. Sums
    reduced over ranks, so every rank needs at least one sim. Returns the
    metrics and the examples {gamma: (sim id, truth, pred)} of the lowest and the highest gamma this rank saw,
    as [F, C, H, W]."""
    device = dit.device
    k = cfg.cond.context
    stride, n = cfg.data.frame_stride, cfg.sampling.samples
    given = max(4 * k - 3 if k else 0, -(-cfg.score_from // stride))  # unscored leading frames: context, score_from
    aggregate, vae, dit_only, ens = (Aggregate(adapter.fields, device, given=given) for _ in range(4))
    ens_sums = torch.zeros(0, device=device)  # per frame, summed over sims
    by_gamma, n_gamma = torch.zeros(len(gammas), device=device), torch.zeros(len(gammas), device=device)  # VRMSE
    ids, seen, examples, per_sim = loaders[0].dataset.ids, 0, {}, {}
    args = (cfg.data.stats_json, adapter.fields, cfg.data.frame_stride, device)
    bench, shared, vae_bench = (
        (Benchmark(*args), Benchmark(*args, prefix="bench_shared", frames=SHARED), Benchmark(*args, prefix="bench_vae"))
        if cfg.benchmark
        else (None, None, None)
    )
    for (latents, gamma), (frames, _) in zip(*loaders, strict=True):
        gamma = gamma.to(device)
        z, batch = latents.to(device), ids[seen : seen + len(gamma)]
        seeds = [noise_seeds(batch, cfg.sampling.seed, i) for i in range(n)]
        samples = torch.stack([decode(adapter, rollout(dit, cond, z, gamma, cfg, s), vaes) for s in seeds])
        pred, recon = samples[0], decode(adapter, z, vaes)  # recon: the VAE's share, his vae_only row
        truth = adapter.select(frames.to(device).flatten(0, 1)).unflatten(0, (len(frames), -1))
        truth = truth[:, : pred.shape[1]]  # the frames the (possibly truncated) latents cover
        for j, (p, x, g) in enumerate(zip(pred, truth, gamma, strict=True)):
            g = round(g.item(), 3)
            m = aggregate.add(p, x).mean(1)
            by_gamma[gammas.index(g)] += m[0]
            n_gamma[gammas.index(g)] += 1
            per_sim[batch[j]] = dict(zip(FIELD_METRICS, m.tolist(), strict=True))
            vae.add(recon[j], x)
            dit_only.add(p, recon[j])
            if n > 1:
                ens.add(samples[:, j].mean(0), x)
                scores = ensemble_scores(samples[:, j], x, given)
                ens_sums = ens_sums + scores if len(ens_sums) else scores
            if bench:
                per_sim[batch[j]]["bench_vrmse"] = bench.add(p, x)[1].item()
                shared.add(p, x)
                vae_bench.add(recon[j], x)
            if g in (gammas[0], gammas[-1]) and g not in examples:
                examples[g] = (batch[j], x.cpu(), p.cpu())
        seen += len(gamma)
    for a in (aggregate, vae, dit_only, ens):
        a.reduce()
    all_sum(by_gamma, n_gamma, ens_sums)
    metrics = aggregate.metrics(stride)
    for prefix, a in (("vae", vae), ("dit", dit_only)) + ((("ens", ens),) if n > 1 else ()):
        split = a.metrics(stride)
        metrics |= {f"{prefix}_{key}": v for key, v in split.items() if key != "vrmse_per_frame"}
    if n > 1:
        spread, skill, crps = ens_sums[:, given:]  # [frames, fields] each

        def spread_skill(frames: slice | torch.Tensor) -> torch.Tensor:
            return ((n + 1) / n * spread[frames].sum(0) / skill[frames].sum(0)).sqrt()

        scores = {"spread_skill": spread_skill(slice(None)), "crps": crps.mean(0) / ens.n}
        for name, v in scores.items():
            metrics |= {f"ens_{name}_{f}": x.item() for f, x in zip(adapter.fields, v, strict=True)}
            metrics[f"ens_{name}_mean"] = v.mean().item()
        for w, idx in well_windows(torch.arange(len(ens_sums[0]), device=device), stride).items():
            if len(idx := idx[idx >= given]):
                metrics[f"ens_spread_skill_frames_{w}"] = spread_skill(idx - given).mean().item()
        metrics["ens_samples"] = n
    metrics |= bench.metrics() | shared.metrics() | vae_bench.metrics() if bench else {}
    metrics |= {f"vrmse_mean_gamma{g:g}": (by_gamma[i] / n_gamma[i]).item() for i, g in enumerate(gammas) if n_gamma[i]}
    metrics["per_sim"] = {sid: v for part in gather(per_sim) for sid, v in part.items()}
    return metrics, examples


@torch.no_grad()
def render(
    dit: WanDiT, cond: nn.Module, adapter: GroupedAdapter, vaes: list, loaders: tuple, cfg: DitStageConfig
) -> dict[str, tuple]:
    """{sim id: (sim id, truth, pred)} of every sim of `loaders` with the evaluation's noise, on this rank alone."""
    out, ids, device = {}, loaders[0].dataset.ids, dit.device
    for (latents, gamma), (frames, _) in zip(*loaders, strict=True):
        batch = ids[len(out) : len(out) + len(gamma)]
        z = rollout(dit, cond, latents.to(device), gamma.to(device), cfg, noise_seeds(batch, cfg.sampling.seed))
        pred = decode(adapter, z, vaes)
        truth = adapter.select(frames.to(device).flatten(0, 1)).unflatten(0, (len(frames), -1))
        for sid, p, x in zip(batch, pred, truth, strict=True):
            out[sid] = (sid, x[: len(p)].cpu(), p.cpu())
    return out
