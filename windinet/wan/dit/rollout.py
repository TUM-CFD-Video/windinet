"""The DiT's rollout: a simulation's latents from noise, conditioned on gamma and the clean context frames; in one shot
or chunk by chunk with the predicted history fed back (sampling.chunk, sampling.history_noise)."""

from __future__ import annotations

import zlib

import torch
from torch import nn

from windinet.wan.dit.config import DitStageConfig
from windinet.wan.dit.delta import from_delta, to_delta
from windinet.wan.dit.model import WanDiT, autocast


def rollout(
    dit: WanDiT,
    cond: nn.Module,
    latents: torch.Tensor,
    gamma: torch.Tensor,
    cfg: DitStageConfig,
    seeds: list[int],
) -> torch.Tensor:
    """Every sample's trajectory from its own noise (`seeds`, see noise_seeds), conditioned on `cond(gamma)` and on the
    first cfg.cond.context latent frames of `latents` [B, G, 16, F', h, w] kept clean (the G latents stacked in the
    channels); same shape back. With sampling.chunk, the whole trajectory is generated again for every chunk, the
    history being the frames kept so far (predicted ones re-noised to sampling.history_noise)."""
    z, k, s = latents.flatten(1, 2), cfg.cond.context, cfg.sampling
    if dit.delta_scale is not None:  # the model generates scaled changes (latents.delta): sampled as such, summed up
        z = to_delta(z, dit.delta_scale)

    def sample(init: torch.Tensor, context: int, i: int) -> torch.Tensor:
        with autocast(cfg.model.precision, dit.device):
            return dit.sample(
                cond(gamma),
                cond.uncond(gamma),
                init.shape,
                s.steps,
                s.guidance,
                [seed + 1000 * i for seed in seeds],
                init=torch.cat([init[:, :, :context], init[:, :, context:] * 0], 2) if context else None,
                context=context,
                history_noise=s.history_noise,
                stochastic=s.stochastic,
            )

    out, chunk = sample(z, k, 0), s.chunk or z.shape[2]
    for i, n in enumerate(range(k + chunk, z.shape[2], chunk), 1):  # history: the first n frames kept so far
        out = torch.cat([out[:, :, :n], sample(torch.cat([out[:, :, :n], z[:, :, n:]], 2), n, i)[:, :, n:]], 2)
    if dit.delta_scale is not None:
        out = from_delta(out, dit.delta_scale)
    return out.unflatten(1, latents.shape[1:3])


def noise_seeds(ids: list[str], seed: int, member: int = 0) -> list[int]:
    """One noise seed per simulation, from its id: every run and batching draws the same noise for the same sim."""
    return [zlib.crc32(sid.encode()) + 1_000_003 * (seed + member) for sid in ids]
