"""Evaluation of the VAE-stage round trip against the simulations: the per-field metrics per simulation, aggregated."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from windinet.data.specs import Spec
from windinet.experiment.distributed import all_sum, gather, world
from windinet.experiment.metrics import Aggregate, by_scalar
from windinet.field_adapter import GroupedAdapter
from windinet.wan.dit.benchmark import Benchmark
from windinet.wan.vae.roundtrip import decode, encode, round_trip


@torch.no_grad()
def evaluate(
    adapter: GroupedAdapter,
    loader: DataLoader,
    vaes: list,
    clip: int | None,
    stride: int = 1,
    latents: bool = False,
    *,
    spec: Spec,
    bench: Benchmark | None = None,
    variances: dict[str, torch.Tensor] | None = None,
) -> tuple[dict, list, dict]:
    """The per-field metrics (FIELD_METRICS) per simulation, averaged over simulations (the VRMSE with its std over
    simulations and, on whole trajectories, per frame and per value of each scalar); the adapter's RGB images of the
    first sim's last frame; on whole trajectories the first sim at the lowest and the highest value of the dataset's
    first scalar {value: (sim id, x, x_hat)} for the videos. `bench` (whole trajectories only): its metrics on the
    same reconstructions. `variances` (spec.uniform_start, clips): each sim's whole-trajectory variance per dataset
    field, the VRMSE's normaliser."""
    # the loader yields one simulation at a time, so the variance normalisation is that simulation's own;
    # clip None = the whole trajectory as one clip; under torchrun each rank sums its shard, then the sums are reduced
    device = next(adapter.parameters()).device
    # latents: the loader yields DiT rollout latents; frame 0 was given to the DiT, so it is not scored (as there)
    aggregate = Aggregate(adapter.fields, device, trajectory=clip is None, given=int(latents))
    first_rgbs, examples = None, {}
    ids = loader.dataset.ids
    first = {s: spec.first(s) for s in ids}
    ends = {min(first.values()), max(first.values())}  # as the DiT's videos: the same sims
    for sid, (frames, z) in zip(ids, loader, strict=True):
        x = adapter.select(frames.flatten(0, 1).to(device))
        if latents:
            x_hat, rgbs = decode(adapter, z.to(device), vaes).flatten(0, 1), adapter(x)
        else:
            x_hat, rgbs = round_trip(adapter, x, vaes, clip or len(x))
        var = None  # uniform start: a sample before the shock has ~0 variance, so the trajectory's normalises
        if variances:
            var = adapter.select(variances[sid][None])[0].to(x)
        elif spec.uniform_start and clip is None:  # the VRMSE's own, now also the per-frame curve's
            var = x[aggregate.given :].var((0, 2, 3), unbiased=False)
        aggregate.add(x_hat, x, var)
        if bench:
            bench.add(x_hat, x)
        first_rgbs = first_rgbs or [r[-1].cpu() for r in rgbs]
        if clip is None and first[sid] in ends and first[sid] not in examples:
            examples[first[sid]] = (sid, x.cpu(), x_hat.cpu())
    aggregate.reduce()
    metrics = aggregate.metrics(stride)
    if variances:  # the other metrics normalise by the sample alone, ~0 before the shock: whole trajectories only
        metrics = {k: v for k, v in metrics.items() if k.startswith("vrmse")}
    if clip is None:
        metrics |= by_scalar(aggregate.per_sim, [s for part in gather([spec.scalars(i) for i in ids]) for s in part])
    return metrics | (bench.metrics() if bench else {}), first_rgbs, examples


@torch.no_grad()
def latent_stats(adapter: GroupedAdapter, loader: DataLoader, vaes: list, start_vaes: list, sims: int = 20) -> dict:
    """Normalised latents of `sims` whole trajectories against N(0, 1) (natural video under Wan's latents_mean/std):
    per-channel mean and std, and their drift from the latents of the run's starting VAEs on the same input."""
    device = next(adapter.parameters()).device
    sums = torch.zeros(5, 16, device=device)  # count, sum, sum of squares, squared change, squared start latent
    for i, (frames, _) in enumerate(loader):
        if i >= -(-sims // world):  # this rank's share of the sims
            break
        x = adapter.select(frames.flatten(0, 1).to(device))
        z, z0 = (encode(adapter, x, v, len(x))[0].transpose(0, 2).flatten(1).double() for v in (vaes, start_vaes))
        sums += torch.stack(
            [
                torch.full_like(z[:, 0], z.shape[1]),
                z.sum(1),
                z.square().sum(1),
                (z - z0).square().sum(1),
                z0.square().sum(1),
            ]
        ).float()
    all_sum(sums)
    n, s, sq, change, ref = sums
    mean, std = s / n, (sq / n - (s / n).square()).clamp(min=0).sqrt()
    return {
        "latent_mean": mean.tolist(),
        "latent_std": std.tolist(),
        "latent_mean_abs": mean.abs().mean().item(),
        "latent_std_mean": std.mean().item(),
        "latent_drift": (change.sum() / ref.sum()).sqrt().item(),
    }
