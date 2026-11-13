"""Evaluation of the DiT's rollouts against the simulations: the per-field metrics per simulation, aggregated,
the VRMSE also per scalar value and per frame; the error split into VAE and DiT; with several samples the ensemble
scores."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader

from windinet.experiment.distributed import all_sum, gather
from windinet.experiment.metrics import FIELD_METRICS, Aggregate, by_scalar, ensemble_scores, well_windows
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
    ends: tuple[float, float],
    cfg: DitStageConfig,
) -> tuple[dict, dict]:
    """Rollouts of every simulation of `loaders` (latents, fields; same ids, same order) against the simulation:
    the per-field metrics (FIELD_METRICS) of one rollout over the predicted frames per sim, averaged over sims; the
    VRMSE also with its std over sims, per value of each scalar and per frame (The Well's metric, also over its
    windows; all frames); vae_*: the decoded simulation latents against the simulation, dit_*: the rollout against
    them; with sampling.samples > 1, ens_*: the mean of the samples, its spread-skill ratio and fair CRPS; per_sim:
    each sim's metrics (means over fields, the cfd_bench VRMSE). Frames before score_from and the context are not
    scored. Sums reduced over ranks, so every rank needs at least one sim. Returns the metrics and the examples
    {value: (sim id, truth, pred)} of the first sim this rank saw at each of `ends` (values of the first scalar), as
    [F, C, H, W]."""
    k, stride, n, spec = cfg.cond.context, cfg.data.frame_stride, cfg.sampling.samples, cfg.data.spec
    given = max(4 * k - 3 if k else 0, -(-cfg.score_from // stride))  # unscored leading frames: context, score_from
    scores = {name: Aggregate(adapter.fields, dit.device, given=given) for name in ("all", "vae", "dit", "ens")}
    benches, ens_sums, examples, per_sim = _benchmarks(cfg, adapter, dit.device), 0, {}, {}
    for sid, x, samples, recon in _rollouts(dit, cond, adapter, vaes, loaders, cfg):
        var = x[given:].var((0, 2, 3), unbiased=False) if spec.uniform_start else None  # ~0 before the shock
        per_sim[sid] = _score(scores, benches, samples[0], x, recon, var)
        if n > 1:
            scores["ens"].add(samples.mean(0), x, var)
            ens_sums = ens_sums + ensemble_scores(samples, x, given)
        first = spec.first(sid)
        if first in ends and first not in examples:
            examples[first] = (sid, x.cpu(), samples[0].cpu())
    for a in scores.values():
        a.reduce()
    metrics = scores["all"].metrics(stride)
    for name in ("vae", "dit") + (("ens",) if n > 1 else ()):
        metrics |= {f"{name}_{key}": v for key, v in scores[name].metrics(stride).items() if key != "vrmse_per_frame"}
    if n > 1:
        all_sum(ens_sums)
        metrics |= _ensemble_metrics(ens_sums, n, scores["ens"].n, given, adapter.fields, stride)
    for bench in benches:
        metrics |= bench.metrics()
    sims = [s for part in gather([spec.scalars(i) for i in per_sim]) for s in part]  # in the order of per_sim
    metrics |= by_scalar(scores["all"].per_sim, sims)
    metrics["per_sim"] = {sid: v for part in gather(per_sim) for sid, v in part.items()}
    return metrics, examples


def _rollouts(dit: WanDiT, cond: nn.Module, adapter: GroupedAdapter, vaes: list, loaders: tuple, cfg: DitStageConfig):
    """Per sim of `loaders`: its id, truth [F, C, H, W], the samples' decoded rollouts [n, F, C, H, W] and its decoded
    latents (the VAE's share, his vae_only row), over the frames the (possibly truncated) latents cover."""
    ids, seen, device = loaders[0].dataset.ids, 0, dit.device
    for (latents, condition), (frames, _) in zip(*loaders, strict=True):
        z, c, batch = latents.to(device), condition.to(device), ids[seen : seen + len(latents)]
        seeds = [noise_seeds(batch, cfg.sampling.seed, i) for i in range(cfg.sampling.samples)]
        samples = torch.stack([decode(adapter, rollout(dit, cond, z, c, cfg, s), vaes) for s in seeds])
        recon = decode(adapter, z, vaes)
        truth = adapter.select(frames.to(device).flatten(0, 1)).unflatten(0, (len(frames), -1))
        yield from zip(batch, truth[:, : samples.shape[2]], samples.transpose(0, 1), recon, strict=True)
        seen += len(latents)


def _benchmarks(cfg: DitStageConfig, adapter: GroupedAdapter, device: torch.device) -> list[Benchmark]:
    """With cfg.benchmark: cfd_bench's scores of the rollout, of the rollout on the shared frames, of the decoded
    latents."""
    if not cfg.benchmark:
        return []
    args = (cfg.data.stats_json, adapter.fields, cfg.data.frame_stride, device)
    shared = Benchmark(*args, prefix="bench_shared", frames=SHARED)
    return [Benchmark(*args), shared, Benchmark(*args, prefix="bench_vae")]


def _score(
    scores: dict, benches: list, pred: torch.Tensor, x: torch.Tensor, recon: torch.Tensor, var: torch.Tensor | None
) -> dict:
    """Adds one sim to `scores` and `benches` (rollout and decoded latents against the truth, rollout against the
    decoded latents); returns its per_sim entry."""
    entry = dict(zip(FIELD_METRICS, scores["all"].add(pred, x, var).mean(1).tolist(), strict=True))
    scores["vae"].add(recon, x, var)
    scores["dit"].add(pred, recon, var)
    if benches:
        bench, shared, vae_bench = benches
        entry["bench_vrmse"] = bench.add(pred, x)[1].item()
        shared.add(pred, x)
        vae_bench.add(recon, x)
    return entry


def _ensemble_metrics(
    sums: torch.Tensor, n: int, sims: torch.Tensor, given: int, fields: list[str], stride: int
) -> dict[str, float]:
    """From the ensemble_scores of `n` samples summed over `sims` sims: the spread-skill ratio and CRPS per field and
    their means, the ratio also over The Well's windows."""
    spread, skill, crps = sums[:, given:]  # [frames, fields] each

    def spread_skill(frames: slice | torch.Tensor) -> torch.Tensor:
        return ((n + 1) / n * spread[frames].sum(0) / skill[frames].sum(0)).sqrt()

    metrics = {}
    for name, v in {"spread_skill": spread_skill(slice(None)), "crps": crps.mean(0) / sims}.items():
        metrics |= {f"ens_{name}_{f}": x.item() for f, x in zip(fields, v, strict=True)}
        metrics[f"ens_{name}_mean"] = v.mean().item()
    for w, idx in well_windows(torch.arange(len(sums[0]), device=sums.device), stride).items():
        if len(idx := idx[idx >= given]):
            metrics[f"ens_spread_skill_frames_{w}"] = spread_skill(idx - given).mean().item()
    return metrics | {"ens_samples": n}


@torch.no_grad()
def render(
    dit: WanDiT, cond: nn.Module, adapter: GroupedAdapter, vaes: list, loaders: tuple, cfg: DitStageConfig
) -> dict[str, tuple]:
    """{sim id: (sim id, truth, pred)} of every sim of `loaders` with the evaluation's noise, on this rank alone."""
    out, ids, device = {}, loaders[0].dataset.ids, dit.device
    for (latents, condition), (frames, _) in zip(*loaders, strict=True):
        batch = ids[len(out) : len(out) + len(latents)]
        z = rollout(dit, cond, latents.to(device), condition.to(device), cfg, noise_seeds(batch, cfg.sampling.seed))
        pred = decode(adapter, z, vaes)
        truth = adapter.select(frames.to(device).flatten(0, 1)).unflatten(0, (len(frames), -1))
        for sid, p, x in zip(batch, pred, truth, strict=True):
            out[sid] = (sid, x[: len(p)].cpu(), p.cpu())
    return out
