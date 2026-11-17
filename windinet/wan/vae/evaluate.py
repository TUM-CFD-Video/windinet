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
    single: bool = False,
) -> tuple[dict, list, dict]:
    """The per-field metrics (FIELD_METRICS) per simulation, averaged over simulations (the VRMSE with its std over
    simulations and, on whole trajectories, per frame and per value of each scalar); the adapter's RGB images of the
    first sim's last frame; on whole trajectories the first sim at the lowest and the highest value of the dataset's
    first scalar {value: (sim id, x, x_hat)} for the videos. Whole trajectories also: per_sim (each sim's VRMSE) and,
    from spec.lola_from on, lola_vrmse_*. `bench` (whole trajectories only): its metrics on the same reconstructions.
    `variances` (spec.uniform_start, clips): each sim's whole-trajectory variance per dataset field, the VRMSE's
    normaliser. `single`: each frame through the VAE alone."""
    # the loader yields one simulation at a time, so the variance normalisation is that simulation's own;
    # clip None = the whole trajectory as one clip; under torchrun each rank sums its shard, then the sums are reduced
    device = next(adapter.parameters()).device
    # latents: the loader yields DiT rollout latents; frame 0 was given to the DiT, so it is not scored (as there)
    aggregate = Aggregate(adapter.fields, device, trajectory=clip is None, given=int(latents))
    # the frames LoLA's autoencoder test scores, normalised over them as the DiT stage's score_from
    lola = None
    if clip is None and not latents and spec.lola_from:
        lola = Aggregate(adapter.fields, device, trajectory=False, given=-(-spec.lola_from // stride))
    first_rgbs, examples = None, {}
    ids = loader.dataset.ids
    first = {s: spec.first(s) for s in ids}
    ends = {min(first.values()), max(first.values())}  # as the DiT's videos: the same sims
    for sid, (frames, z) in zip(ids, loader, strict=True):
        x, x_hat, rgbs = _reconstruct(adapter, frames, z, vaes, clip, latents, single)
        var = None  # uniform start: a sample before the shock has ~0 variance, so the trajectory's normalises
        if variances:
            var = adapter.select(variances[sid][None])[0].to(x)
        elif spec.uniform_start and clip is None:  # the VRMSE's own, now also the per-frame curve's
            var = x[aggregate.given :].var((0, 2, 3), unbiased=False)
        aggregate.add(x_hat, x, var)
        if lola and lola.given < len(x):
            lola.add(x_hat, x, x[lola.given :].var((0, 2, 3), unbiased=False) if spec.uniform_start else None)
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
        sids = [s for part in gather(list(ids)) for s in part]  # the order of aggregate.per_sim
        metrics["per_sim"] = {s: {"vrmse": v.mean().item()} for s, v in zip(sids, aggregate.per_sim, strict=True)}
    if lola:
        lola.reduce()  # on every rank, also one without sims or with trajectories too short for it
    if lola and lola.n:
        metrics |= {f"lola_{k}": v for k, v in lola.metrics().items() if k.startswith("vrmse")}
        metrics["lola_from"] = spec.lola_from
    return metrics | (bench.metrics() if bench else {}), first_rgbs, examples


def _reconstruct(adapter: GroupedAdapter, frames, z, vaes: list, clip: int | None, latents: bool, single: bool):
    """x, x_hat [N, C, H, W] and the RGB images of one simulation, as `evaluate` scores it."""
    x = adapter.select(frames.flatten(0, 1).to(next(adapter.parameters()).device))
    if latents:
        return x, decode(adapter, z.to(x.device), vaes).flatten(0, 1), adapter(x)
    return x, *round_trip(adapter, x, vaes, 1 if single else clip or len(x))


@torch.no_grad()
def render(adapter: GroupedAdapter, loader: DataLoader, vaes: list, latents: bool, single: bool) -> dict[str, tuple]:
    """{sim id: (sim id, x, x_hat)} of the whole trajectories of `loader` as the test reconstructs them, on this rank
    alone."""
    out = {}
    for sid, (frames, z) in zip(loader.dataset.ids, loader, strict=True):
        x, x_hat, _ = _reconstruct(adapter, frames, z, vaes, None, latents, single)
        out[sid] = (sid, x.cpu(), x_hat.cpu())
    return out


@torch.no_grad()
def latent_stats(
    adapter: GroupedAdapter, loader: DataLoader, vaes: list, start_vaes: list, sims: int = 20, single: bool = False
) -> dict:
    """Normalised latents of `sims` whole trajectories against N(0, 1) (natural video under Wan's latents_mean/std):
    per-channel mean and std, their drift from the latents of the run's starting VAEs on the same input, and the mean
    KL of the posteriors (WanVAE.posterior); `single`: each frame encoded alone."""
    device = next(adapter.parameters()).device
    channels = vaes[0].latents_mean.numel()
    sums = torch.zeros(5, channels, device=device)  # count, sum, sum of squares, squared change, squared start latent
    kl = torch.zeros(2, device=device)  # sum of the per-sim KL, sims
    for i, (frames, _) in enumerate(loader):
        if i >= -(-sims // world):  # this rank's share of the sims
            break
        x = adapter.select(frames.flatten(0, 1).to(device))
        (z, _, kl_sim), (z0, *_) = (encode(adapter, x, v, 1 if single else len(x)) for v in (vaes, start_vaes))
        kl += torch.stack([kl_sim, torch.ones_like(kl_sim)])
        z, z0 = (t.transpose(0, 2).flatten(1).double() for t in (z, z0))
        sums += torch.stack(
            [
                torch.full_like(z[:, 0], z.shape[1]),
                z.sum(1),
                z.square().sum(1),
                (z - z0).square().sum(1),
                z0.square().sum(1),
            ]
        ).float()
    all_sum(sums, kl)
    n, s, sq, change, ref = sums
    mean, std = s / n, (sq / n - (s / n).square()).clamp(min=0).sqrt()
    return {
        "latent_mean": mean.tolist(),
        "latent_std": std.tolist(),
        "latent_mean_abs": mean.abs().mean().item(),
        "latent_std_mean": std.mean().item(),
        "latent_drift": (change.sum() / ref.sum()).sqrt().item(),
        "latent_kl": (kl[0] / kl[1]).item(),
    }
