"""Evaluation of the VAE-stage round trip against the simulations: the per-field metrics per simulation, aggregated."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from windinet.eulermq.data import parse_gamma
from windinet.experiment.distributed import all_sum, world
from windinet.experiment.metrics import Aggregate
from windinet.field_adapter import GroupedAdapter
from windinet.wan.vae.roundtrip import decode, encode, round_trip


@torch.no_grad()
def evaluate(
    adapter: GroupedAdapter, loader: DataLoader, vaes: list, clip: int | None, stride: int = 1, latents: bool = False
) -> tuple[dict, tuple, dict]:
    """The per-field metrics (FIELD_METRICS) per simulation, averaged over simulations (the VRMSE with its std over
    simulations and, on whole trajectories, per frame); one example (x, x_hat, rgbs) for the panel; on whole
    trajectories the first sim at the lowest and the highest gamma {gamma: (sim id, x, x_hat)} for the videos."""
    # the loader yields one simulation at a time, so the variance normalisation is that simulation's own;
    # clip None = the whole trajectory as one clip; under torchrun each rank sums its shard, then the sums are reduced
    device = next(adapter.parameters()).device
    # latents: the loader yields DiT rollout latents; frame 0 was given to the DiT, so it is not scored (as there)
    aggregate = Aggregate(adapter.fields, device, trajectory=clip is None, given=int(latents))
    example, examples = None, {}
    ids = loader.dataset.ids
    ends = {f(round(parse_gamma(s), 3) for s in ids) for f in (min, max)}  # as the DiT's videos: the same sims
    for sid, (frames, z) in zip(ids, loader, strict=True):
        x = adapter.select(frames.flatten(0, 1).to(device))
        if latents:
            x_hat, rgbs = decode(adapter, z.to(device), vaes).flatten(0, 1), adapter(x)
        else:
            x_hat, rgbs = round_trip(adapter, x, vaes, clip or len(x))
        aggregate.add(x_hat, x)
        mid = len(x) // 2  # frame 0 is the piecewise-constant initial condition; the middle shows developed flow
        example = example or (x[mid].cpu(), x_hat[mid].cpu(), [r[mid].cpu() for r in rgbs])
        g = round(parse_gamma(sid), 3)
        if clip is None and g in ends and g not in examples:
            examples[g] = (sid, x.cpu(), x_hat.cpu())
    aggregate.reduce()
    return aggregate.metrics(stride), example, examples


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
