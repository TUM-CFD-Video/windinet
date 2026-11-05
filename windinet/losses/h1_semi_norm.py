"""
H1 semi-norm loss.

Measures the difference between spatial gradients
of prediction and target.

Expected input:
    pred, target: [B, C, T, H, W]

The H1 seminorm is:

    |u|_H1^2 = ||∇u||^2

Here we minimize:

    ||∇u_pred - ∇u_target||^2

h1_seminorm_per_channel is the benchmark metric beside it: the relative error
||∇(u_pred - u_target)|| / ||∇u_target|| per channel.
"""

from collections.abc import Callable

import torch
import torch.nn.functional as F


def spatial_gradients(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute first-order spatial gradients.

    Args:
        x:
            Tensor with shape [B, C, T, H, W]

    Returns:
        dx:
            Gradient along width direction.
        dy:
            Gradient along height direction.
    """

    # Difference along W dimension
    dx = x[..., :, 1:] - x[..., :, :-1]

    # Difference along H dimension
    dy = x[..., 1:, :] - x[..., :-1, :]

    # Keep original tensor size
    dx = F.pad(
        dx,
        (0, 1, 0, 0),
    )

    dy = F.pad(
        dy,
        (0, 0, 0, 1),
    )

    return dx, dy


def h1_seminorm_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Compute H1 semi-norm loss.

    Penalizes differences between spatial derivatives.

    Args:
        pred:
            Predicted fields.
            Shape [B, C, T, H, W]

        target:
            Ground truth fields.
            Shape [B, C, T, H, W]

    Returns:
        Scalar H1 loss.
    """

    pred_dx, pred_dy = spatial_gradients(pred)
    target_dx, target_dy = spatial_gradients(target)

    loss_dx = F.mse_loss(
        pred_dx,
        target_dx,
    )

    loss_dy = F.mse_loss(
        pred_dy,
        target_dy,
    )

    return loss_dx + loss_dy


def _relative_seminorm(
    derivatives: Callable[[torch.Tensor], tuple[torch.Tensor, ...]],
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """||D(pred - target)|| / ||D target|| per channel for the derivatives D, over every dim but batch and channel,
    averaged over the batch -- reduced like vrms_per_channel. pred, target: [B, C, T, H, W]; returns [C]."""
    pred, target = pred.float(), target.float()
    dims = tuple(range(2, pred.dim()))
    error = sum(d.square().mean(dims) for d in derivatives(pred - target))
    norm = sum(d.square().mean(dims) for d in derivatives(target))
    return torch.sqrt(error / (norm + eps)).mean(dim=0)


def _first_differences(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward differences along H and W between neighbouring cells only (no padding: open boundaries)."""
    return x[..., 1:, :] - x[..., :-1, :], x[..., :, 1:] - x[..., :, :-1]


def h1_seminorm_per_channel(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Relative H1 seminorm error per channel, |pred - target|_H1 / |target|_H1: the benchmark metric beside the
    training loss above. Scale-free and the grid spacing cancels; 1 = as bad as a flat prediction.
    pred, target: [B, C, T, H, W]; returns [C], averaged over the batch."""
    return _relative_seminorm(_first_differences, pred, target, eps)
