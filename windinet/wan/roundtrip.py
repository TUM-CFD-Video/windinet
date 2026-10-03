"""The VAE-stage round trip fields -> adapter -> Wan VAE -> adapter^-1 -> fields: training loss and evaluation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.utils.data import DataLoader

from windinet.experiment.distributed import all_sum
from windinet.experiment.metrics import clipped, vrmse
from windinet.field_adapter import GroupedAdapter
from windinet.losses import h1_seminorm_loss, rmse_loss
from windinet.wan.vae_stage import LossConfig

if TYPE_CHECKING:
    from windinet.wan.vae import WanVAE


def round_trip(
    adapter: GroupedAdapter, x: torch.Tensor, vaes: list[WanVAE], clip: int = 1
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """x_hat and the RGB image per group. `x`: [N, F, H, W] frames, passed to the VAE as N / clip clips."""
    rgbs = adapter(x)
    if not vaes:  # adapter only
        return adapter.inverse(rgbs), rgbs
    per_group = [vaes[g % len(vaes)] for g in range(len(rgbs))]  # one shared VAE, or one per group
    decoded = [
        v.decode(v.encode(rgb.unflatten(0, (-1, clip)).movedim(1, 2))).movedim(2, 1).flatten(0, 1)
        for v, rgb in zip(per_group, rgbs, strict=True)
    ]
    return adapter.inverse(decoded), rgbs


def _as_clip(x: torch.Tensor) -> torch.Tensor:
    """[N, F, H, W] frames -> [1, F, N, H, W], the layout windinet.losses expect (frames pooled as time)."""
    return x.movedim(0, 1).unsqueeze(0)


def losses(
    adapter: GroupedAdapter, x: torch.Tensor, vaes: list[WanVAE], clip: int, weights: LossConfig
) -> dict[str, torch.Tensor]:
    """Reconstruction terms in z-scored field space and their weighted sum, "loss"."""
    x_hat, _ = round_trip(adapter, x, vaes, clip)
    z, z_hat = _as_clip(adapter.normalized(x)), _as_clip(adapter.normalized(x_hat))
    terms = {"rmse": rmse_loss(z_hat, z), "h1": h1_seminorm_loss(z_hat, z)}
    return {"loss": weights.rmse * terms["rmse"] + weights.h1 * terms["h1"], **terms}


@torch.no_grad()
def evaluate(
    adapter: GroupedAdapter,
    loader: DataLoader,
    vaes: list[WanVAE],
    norm: tuple[torch.Tensor, torch.Tensor],
    clip: int | None,
) -> tuple[dict[str, float], tuple]:
    """Per-field VRMSE per simulation, averaged over simulations, in physical units and in the LTX baseline's
    5-sigma clipped space (`norm` = field means and stds); plus one example (x, x_hat, rgbs) for the panel."""
    # the loader yields one simulation at a time, so the variance normalisation is that simulation's own;
    # clip None = the whole trajectory as one clip; under torchrun each rank sums its shard, then the sums are reduced
    mean, std = norm
    total = torch.zeros(2, len(adapter.fields), device=mean.device)  # physical, clipped
    n = torch.zeros((), device=mean.device)
    example = None
    for frames, _ in loader:
        x = adapter.select(frames.flatten(0, 1).to(mean.device))
        x_hat, rgbs = round_trip(adapter, x, vaes, clip or len(x))
        total[0] += vrmse(x_hat, x)
        total[1] += vrmse(clipped(x_hat, mean, std), clipped(x, mean, std))
        n += 1
        mid = len(x) // 2  # frame 0 is the piecewise-constant initial condition; the middle shows developed flow
        example = example or (x[mid].cpu(), x_hat[mid].cpu(), [r[mid].cpu() for r in rgbs])
    all_sum(total, n)
    metrics = {}
    for tag, per_field in zip(("vrmse", "vrmse_clipped"), total / n, strict=True):
        metrics |= {f"{tag}_{name}": v.item() for name, v in zip(adapter.fields, per_field, strict=True)}
        metrics[f"{tag}_mean"] = per_field.mean().item()
    return metrics, example
