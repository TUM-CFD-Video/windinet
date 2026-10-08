"""The VAE-stage objective: reconstruction terms of the round trip in z-scored field space."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from windinet.field_adapter import GroupedAdapter
from windinet.wan.vae.config import LossConfig
from windinet.wan.vae.roundtrip import round_trip

if TYPE_CHECKING:
    from windinet.wan.vae.model import WanVAE


def rmse(d: torch.Tensor) -> torch.Tensor:
    """Root mean square of the residual `d` (x_hat - x), eps inside the root so the gradient at zero is finite."""
    return (d.square().mean() + 1e-8).sqrt()


def h1(d: torch.Tensor) -> torch.Tensor:
    """H1 seminorm of the residual: its squared forward differences in x and in y, per pixel."""
    return (d.diff(dim=-1).square().sum() + d.diff(dim=-2).square().sum()) / d.numel()


def loss_terms(
    adapter: GroupedAdapter, x: torch.Tensor, vaes: list[WanVAE], clip: int, weights: LossConfig
) -> dict[str, torch.Tensor]:
    """The terms on the round trip of `x` in z-scored field space, and their weighted sum, "loss"."""
    x_hat, _ = round_trip(adapter, x, vaes, clip)
    d = adapter.normalized(x_hat) - adapter.normalized(x)
    terms = {"rmse": rmse(d), "h1": h1(d)}
    return {"loss": weights.rmse * terms["rmse"] + weights.h1 * terms["h1"], **terms}
