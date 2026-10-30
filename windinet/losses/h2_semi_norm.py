"""
H2 semi-norm loss.

One order past h1_seminorm_loss's first derivatives: measures the
difference between second-order spatial derivatives (curvature) of
prediction and target. Where H1 penalizes gradient (edge/slope) mismatches,
H2 additionally penalizes curvature mismatches -- relevant for shock
fronts, where the second derivative is large and a pure first-derivative
penalty can still miss over-smoothed peaks.

Expected input:
    pred, target: [B, C, T, H, W]

The H2 seminorm is:

    |u|_H2^2 = ||del^2 u||^2

Here we minimize:

    ||del^2 u_pred - del^2 u_target||^2

h2_seminorm_per_channel is the benchmark metric beside it: the relative error
of the full Hessian (mixed derivative included) per channel.
"""

import torch
import torch.nn.functional as F

from .h1_semi_norm import _relative_seminorm


def spatial_second_derivatives(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute second-order spatial derivatives via central differences.

    Args:
        x:
            Tensor with shape [B, C, T, H, W]

    Returns:
        dxx:
            Second derivative along width direction.
        dyy:
            Second derivative along height direction.
    """

    # Central second difference: x[i-1] - 2*x[i] + x[i+1]
    dxx = x[..., :, :-2] - 2 * x[..., :, 1:-1] + x[..., :, 2:]
    dyy = x[..., :-2, :] - 2 * x[..., 1:-1, :] + x[..., 2:, :]

    # Keep original tensor size
    dxx = F.pad(
        dxx,
        (1, 1, 0, 0),
    )

    dyy = F.pad(
        dyy,
        (0, 0, 1, 1),
    )

    return dxx, dyy


def h2_seminorm_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Compute H2 semi-norm loss.

    Penalizes differences between second-order spatial derivatives
    (curvature) -- complements h1_seminorm_loss's first derivatives.

    Args:
        pred:
            Predicted fields.
            Shape [B, C, T, H, W]

        target:
            Ground truth fields.
            Shape [B, C, T, H, W]

    Returns:
        Scalar H2 loss.
    """

    pred_dxx, pred_dyy = spatial_second_derivatives(pred)
    target_dxx, target_dyy = spatial_second_derivatives(target)

    loss_dxx = F.mse_loss(
        pred_dxx,
        target_dxx,
    )

    loss_dyy = F.mse_loss(
        pred_dyy,
        target_dyy,
    )

    return loss_dxx + loss_dyy


def _hessian(x: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Second differences between real cells only (H, W >= 3): d_hh, d_ww and the mixed d_hw twice, so the sum of
    squares is the Hessian's Frobenius norm (unchanged under the grid's rotations and reflections)."""
    d_hh = x[..., 2:, :] - 2 * x[..., 1:-1, :] + x[..., :-2, :]
    d_ww = x[..., :, 2:] - 2 * x[..., :, 1:-1] + x[..., :, :-2]
    d_hw = x[..., 1:, 1:] - x[..., 1:, :-1] - x[..., :-1, 1:] + x[..., :-1, :-1]
    return d_hh, d_ww, d_hw, d_hw


def h2_seminorm_per_channel(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Relative H2 seminorm error per channel, |pred - target|_H2 / |target|_H2 with the full Hessian: the benchmark
    metric beside the training loss above. pred, target: [B, C, T, H, W]; returns [C], averaged over the batch."""
    return _relative_seminorm(_hessian, pred, target, eps)
