"""Where a simulation has fine detail: fronts of any strength (Loehner's indicator on density and pressure) and
vortex cores (Q-criterion), as the share of each latent cell's pixels, for weighting the flow loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F

FILTER = 0.01  # Loehner's noise filter (FLASH default): ignores jumps below ~1 % of the local value


def lohner(u: torch.Tensor) -> torch.Tensor:
    """Loehner's (1987) error indicator in [0, 1] per pixel of u [T, H, W], edge padded."""
    v = F.pad(u[:, None], (1, 1, 1, 1), mode="replicate")[:, 0]
    c, xp, xm, yp, ym = v[:, 1:-1, 1:-1], v[:, 2:, 1:-1], v[:, :-2, 1:-1], v[:, 1:-1, 2:], v[:, 1:-1, :-2]
    pp, pm, mp, mm = v[:, 2:, 2:], v[:, 2:, :-2], v[:, :-2, 2:], v[:, :-2, :-2]
    num = (xp - 2 * c + xm) ** 2 + (yp - 2 * c + ym) ** 2 + 2 * ((pp - pm - mp + mm) / 4) ** 2
    dxx = (xp - c).abs() + (c - xm).abs() + FILTER * (xp.abs() + 2 * c.abs() + xm.abs())
    dyy = (yp - c).abs() + (c - ym).abs() + FILTER * (yp.abs() + 2 * c.abs() + ym.abs())
    corners = FILTER * (pp.abs() + pm.abs() + mp.abs() + mm.abs())
    dxy = ((pp - pm).abs() + (mp - mm).abs() + corners) / 2  # d/dy at x +- 1
    dyx = ((pp - mp).abs() + (pm - mm).abs() + corners) / 2  # d/dx at y +- 1
    return (num / (dxx**2 + dyy**2 + dxy**2 + dyx**2)).sqrt()


def vortex_q(rho: torch.Tensor, mx: torch.Tensor, my: torch.Tensor) -> torch.Tensor:
    """Q = (|W|^2 - |S|^2) / 2 of the velocity m / rho [T, H, W] (axis H = x), over the frame's mean |grad u|^2."""
    a, b = torch.gradient(mx / rho, dim=(1, 2))
    c, d = torch.gradient(my / rho, dim=(1, 2))
    q = ((c - b) ** 2 / 2 - a**2 - d**2 - (b + c) ** 2 / 2) / 2
    return q / (a**2 + b**2 + c**2 + d**2).mean((1, 2), keepdim=True)


def detail_share(fields: torch.Tensor, threshold: float = 0.2, q_threshold: float = 1.0, cell: int = 8) -> torch.Tensor:
    """Fields [T, 4, H, W] (density, momenta, pressure; T = 1 + 4k frames as encoded) -> per latent frame and cell
    the share of pixels with Loehner(density or pressure) > threshold or Q > q_threshold [1 + k, H / cell, W / cell];
    a latent frame's share is the mean over its 4 frames, the first frame's its own."""
    rho, mx, my, p = fields.double().unbind(1)
    mask = (lohner(rho) > threshold) | (lohner(p) > threshold) | (vortex_q(rho, mx, my) > q_threshold)
    share = F.avg_pool2d(mask[:, None].float(), cell)[:, 0]
    return torch.cat([share[:1], share[1:].unflatten(0, (-1, 4)).mean(1)])
