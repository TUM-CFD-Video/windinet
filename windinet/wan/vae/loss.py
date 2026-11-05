"""The VAE-stage objective: reconstruction terms of the round trip in z-scored field space."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from windinet.experiment.metrics import shells
from windinet.field_adapter import GroupedAdapter
from windinet.wan.vae.config import LossConfig
from windinet.wan.vae.roundtrip import decode, round_trip

if TYPE_CHECKING:
    from windinet.wan.vae.model import WanVAE


def rmse(d: torch.Tensor) -> torch.Tensor:
    """Root mean square of the residual `d` (x_hat - x), eps inside the root so the gradient at zero is finite."""
    return (d.square().mean() + 1e-8).sqrt()


def h1(d: torch.Tensor) -> torch.Tensor:
    """H1 seminorm of the residual: its squared forward differences in x and in y, per pixel."""
    return (d.diff(dim=-1).square().sum() + d.diff(dim=-2).square().sum()) / d.numel()


def l1_gdl(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """L1 of the residual plus the gradient-difference term of Mathieu et al. (2016, α = 1): L1 of the residual's
    forward differences in x and in y, each over the std of the target's differences so both terms weigh alike."""
    d = x_hat - x
    return d.abs().mean() + sum(d.diff(dim=k).abs().mean() / x.diff(dim=k).std() for k in (-1, -2))


def amse(x_hat: torch.Tensor, x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Adjusted MSE of Subich et al. (2025) on [N, C, H, W]: per frame, field and integer wavenumber shell
    (sqrt P_hat - sqrt P)^2 + 2 max(P_hat, P) (1 - coherence), in MSE units. MSE has sqrt(P_hat P) where this has the
    max, so damping a scale whose position is uncertain no longer lowers the loss."""
    shell, _ = shells(x)

    def per_shell(t: torch.Tensor) -> torch.Tensor:
        return t.new_zeros(*t.shape[:-1], int(shell.max()) + 1).index_add_(-1, shell, t)

    a, b = (torch.fft.fft2(t, norm="ortho").flatten(-2) for t in (x_hat, x))
    p_hat, p, cross = (per_shell(t) for t in (a.abs().square(), b.abs().square(), (a * b.conj()).real))
    coherence = cross / (p_hat * p + eps).sqrt()
    terms = ((p_hat + eps).sqrt() - (p + eps).sqrt()).square() + 2 * torch.maximum(p_hat, p) * (1 - coherence)
    return terms.sum(-1).mean() / x[0, 0].numel()


def front_distance(
    x_hat: torch.Tensor, x: torch.Tensor, tau: float = 0.2, radius: int = 16, temp: float = 0.5
) -> torch.Tensor:
    """Symmetric soft Chamfer distance between the fronts of [N, C, H, W] fields, as a fraction of `radius` px: the
    mean distance of predicted fronts to the true ones plus that of true fronts to the predicted ones. Fronts: gradient
    magnitude above `tau` times the truth's 99th percentile per frame and field (soft for the prediction)."""

    def grad(t: torch.Tensor) -> torch.Tensor:
        return (t.diff(dim=-1)[..., :-1, :].square() + t.diff(dim=-2)[..., :-1].square() + 1e-12).sqrt()

    g_hat, g = grad(x_hat), grad(x)
    scale = g.flatten(2).quantile(0.99, dim=2)[..., None, None] + 1e-8
    pred, true = torch.sigmoid((g_hat / scale - tau) / 0.02), (g / scale > tau).float()
    r = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    kernel = torch.exp(-(r[:, None] ** 2 + r[None] ** 2).sqrt() / temp).view(1, 1, 2 * radius + 1, -1)

    def distance(fronts: torch.Tensor) -> torch.Tensor:  # soft-min distance to the fronts (px), saturating at radius
        near = F.conv2d(fronts.flatten(0, 1)[:, None], kernel, padding=radius).view_as(fronts)
        return (-temp * (near + math.exp(-radius / temp)).log()).clamp(min=0)

    extra = (pred * distance(true)).sum() / (pred.sum() + 1e-8)
    missing = (true * distance(pred)).sum() / (true.sum() + 1e-8)
    return (extra + missing) / radius


def window(z: torch.Tensor, frames: int | None, generator: torch.Generator | None = None) -> slice:
    """Latent frames the pixel loss decodes: all, or `frames` from a start s >= 1 drawn with probability ~ s."""
    if frames is None or frames >= z.shape[-3]:
        return slice(None)
    starts = torch.arange(1, z.shape[-3] - frames + 1, dtype=torch.float)
    s = int(starts[torch.multinomial(starts, 1, generator=generator)])
    return slice(s, s + frames)


def pixel_loss(
    adapter: GroupedAdapter,
    vaes: list[WanVAE],
    x0_hat: torch.Tensor,
    z: torch.Tensor,
    frames: int | None = None,
    front: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """`l1_gdl` (plus `front` x `front_distance` on density and pressure) in z-scored field space between the decoded
    clean-latent estimate and the decoded clean latent, both [n, G, 16, F', h, w]: over the predicted frames, or over
    a window of `frames` latent frames decoded on their own (see `window`). Returns the loss and the front term."""
    w = window(z, frames)
    skip = 1 if w.start is None else 0  # pixel frame 0 is the clean context, identical in both
    pred = adapter.normalized(decode(adapter, x0_hat[..., w, :, :], vaes)[:, skip:].flatten(0, 1))
    with torch.no_grad():
        ref = adapter.normalized(decode(adapter, z[..., w, :, :], vaes)[:, skip:].flatten(0, 1))
    fronts = [adapter.fields.index(f) for f in ("density", "pressure") if f in adapter.fields]
    term = front_distance(pred[:, fronts], ref[:, fronts]) if front else pred.new_zeros(())
    return l1_gdl(pred, ref) + front * term, term


def loss_terms(
    adapter: GroupedAdapter,
    x: torch.Tensor,
    vaes: list[WanVAE],
    clip: int,
    weights: LossConfig,
    latents: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """The terms on the round trip of `x` (or on decoding `latents` [n, G, 16, F', h, w] against it) in z-scored field
    space, and the loss weights.kind makes of them, "loss"."""
    x_hat = (
        decode(adapter, latents, vaes).flatten(0, 1) if latents is not None else round_trip(adapter, x, vaes, clip)[0]
    )
    x_hat, x = adapter.normalized(x_hat), adapter.normalized(x)
    terms = {"rmse": rmse(x_hat - x), "h1": h1(x_hat - x)}
    match weights.kind:
        case "l1_gdl":
            loss = l1_gdl(x_hat, x)
        case "amse":
            loss = amse(x_hat, x)
        case _:
            loss = weights.rmse * terms["rmse"] + weights.h1 * terms["h1"]
    return {"loss": loss, **terms}
