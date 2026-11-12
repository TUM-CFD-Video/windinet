"""
Total-energy consistency (Harish, 2026-11-07): the energy implied by the predicted fields against the
ground truth's,

    E = p / (gamma - 1) + (m_x^2 + m_y^2) / (2 rho)        (= p/(gamma-1) + rho (u^2 + v^2) / 2)

with our channels density rho, momentum m = rho u and pressure p, and gamma per simulation (Euler_MQ's
equation of state; the HDF5's own energy field is this identity of the solver). The four channels are
scored one by one everywhere else; E couples them, so a prediction whose rho, m and p are each
slightly off in inconsistent ways shows up here.

Two uses:
- `energy_vrmse_from_normalized`: VAE training loss "energy" (opt-in weight; always logged).
  Works on the network's normalized [-1, 1] fields, undoes the z-score, scores E.
- `total_energy` / `energy_vrmse`: physical-unit metric in scripts/eval_dit_vrmse.py.

The score is a VRMSE of E, sqrt(mean((E_pred - E_true)^2) / var(E_true)) per sample (and per frame if
asked), the same normalization as every field's VRMSE, so it needs no energy statistics and does not
scale with gamma or the flow's energy level.
"""

import torch

ENERGY_CHANNELS = ("density", "momentum_x", "momentum_y", "pressure")
# Density floor in the kinetic term's denominator: the data's minimum is ~0.02, a decoded field can
# undershoot to ~0 or below early in training.
DENSITY_FLOOR = 1e-3


def has_energy_channels(channel_order: list[str]) -> bool:
    """True if channel_order holds every field E needs (Euler_MQ's 4 channels, any order)."""
    return all(name in channel_order for name in ENERGY_CHANNELS)


def total_energy(
    fields: torch.Tensor,  # [B, C, ...] physical units
    channel_order: list[str],
    gamma: torch.Tensor | float,  # [B] or scalar
    density_floor: float = DENSITY_FLOOR,
) -> torch.Tensor:
    """E = p/(gamma-1) + (m_x^2 + m_y^2)/(2 rho), shape [B, ...]."""
    rho, mx, my, p = (fields[:, channel_order.index(name)] for name in ENERGY_CHANNELS)
    gamma = torch.as_tensor(gamma, dtype=fields.dtype, device=fields.device)
    if gamma.dim() > 0:
        gamma = gamma.view(-1, *([1] * (rho.dim() - 1)))
    return p / (gamma - 1.0) + (mx.square() + my.square()) / (2.0 * rho.clamp(min=density_floor))


def energy_vrmse(
    e_pred: torch.Tensor,  # [B, T, H, W] (or [B, ...])
    e_true: torch.Tensor,
    per_frame: bool = False,
    eps: float = 1e-8,
) -> torch.Tensor:
    """VRMSE of E: [B] (over all frames), or [B, T] with per_frame (each frame its own variance)."""
    dims = tuple(range(2, e_true.dim())) if per_frame else tuple(range(1, e_true.dim()))
    mse = (e_pred - e_true).square().mean(dim=dims)
    return torch.sqrt(mse / (e_true.var(dim=dims, unbiased=False) + eps))


def energy_vrmse_from_normalized(
    pred: torch.Tensor,  # [B, C, T, H, W], normalized as build_shockwave_video
    target: torch.Tensor,
    *,
    gamma: torch.Tensor,  # [B]
    channel_mean: list[float],
    channel_std: list[float],
    normalization_clip: float,
    channel_order: list[str],
    log_transform_channels: list[str] | None = None,
) -> torch.Tensor:
    """VAE loss: batch-mean VRMSE of E between pred and target after undoing the normalization.

    The target is the clamped normalized ground truth the VAE is trained on, so E_true is the energy of
    the fields as the VAE sees them (identical to the true E wherever no channel hits the ±clip·σ
    clamp). Zero (no graph) if channel_order lacks one of the four fields (3-channel runs, SRBA).
    """
    if not has_energy_channels(channel_order):
        return pred.new_zeros(())
    shape = (1, len(channel_order)) + (1,) * (pred.dim() - 2)
    mean = pred.new_tensor(channel_mean).view(shape)
    scale = pred.new_tensor(channel_std).view(shape) * normalization_clip

    def physical(x: torch.Tensor) -> torch.Tensor:
        x = x.float() * scale + mean
        for name in log_transform_channels or []:
            c = channel_order.index(name)
            x = torch.cat([x[:, :c], x[:, c:c + 1].exp(), x[:, c + 1:]], dim=1)
        return x

    gamma = gamma.to(device=pred.device, dtype=torch.float32)
    e_pred = total_energy(physical(pred), channel_order, gamma)
    e_true = total_energy(physical(target), channel_order, gamma)
    return energy_vrmse(e_pred, e_true).mean()
