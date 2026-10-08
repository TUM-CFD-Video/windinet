"""The DiT's rollout: a simulation's latents from noise, conditioned on gamma and the clean first latent frame."""

from __future__ import annotations

import torch
from torch import nn

from windinet.wan.dit.config import DitStageConfig
from windinet.wan.dit.model import WanDiT, autocast


def rollout(
    dit: WanDiT, cond: nn.Module, latents: torch.Tensor, gamma: torch.Tensor, cfg: DitStageConfig
) -> torch.Tensor:
    """Every sample's trajectory from noise, conditioned on `cond(gamma)` and, with cfg.cond.first_frame, on the clean
    first latent frame of `latents` [B, G, 16, F', h, w] (the G latents stacked in the channels); same shape back."""
    z, first = latents.flatten(1, 2), cfg.cond.first_frame
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
    return out.unflatten(1, latents.shape[1:3])
