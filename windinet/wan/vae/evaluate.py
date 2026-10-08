"""Evaluation of the VAE-stage round trip against the simulations: the per-field metrics per simulation, aggregated."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from windinet.experiment.metrics import Aggregate
from windinet.field_adapter import GroupedAdapter
from windinet.wan.vae.roundtrip import round_trip


@torch.no_grad()
def evaluate(
    adapter: GroupedAdapter, loader: DataLoader, vaes: list, clip: int | None, stride: int = 1
) -> tuple[dict, tuple]:
    """The per-field metrics (FIELD_METRICS) per simulation, averaged over simulations (the VRMSE with its std over
    simulations and, on whole trajectories, per frame); plus one example (x, x_hat, rgbs) for the panel."""
    # the loader yields one simulation at a time, so the variance normalisation is that simulation's own;
    # clip None = the whole trajectory as one clip; under torchrun each rank sums its shard, then the sums are reduced
    device = next(adapter.parameters()).device
    aggregate = Aggregate(adapter.fields, device, trajectory=clip is None)  # the frame index is time only then
    example = None
    for frames, _ in loader:
        x = adapter.select(frames.flatten(0, 1).to(device))
        x_hat, rgbs = round_trip(adapter, x, vaes, clip or len(x))
        aggregate.add(x_hat, x)
        mid = len(x) // 2  # frame 0 is the piecewise-constant initial condition; the middle shows developed flow
        example = example or (x[mid].cpu(), x_hat[mid].cpu(), [r[mid].cpu() for r in rgbs])
    aggregate.reduce()
    return aggregate.metrics(stride), example
