"""
Variance-normalized RMSE (VRMS) loss.

Per channel, then averaged over channels (The Well's convention; the
"val_vrmse_chmean" eval metric, exposed here as a differentiable loss so it can
optionally be optimized directly). Normalizing by each field's own variance
means a channel with small dynamic range isn't automatically "cheap" to get
right the way plain RMSE would let it be.

The pooled variant (one variance over all channels together) was removed on
2026-10-24 and is not used anywhere any more: it also counts the spread between
the channels' means, so it reads lower than every single channel.

Expected input:
    pred, target: [B, C, T, H, W]
"""

import torch


def vrms_per_channel(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Variance-normalized RMSE per channel: sqrt(mean((pred - target)^2) / var(target)).

    Reduces over every dim except batch and channel, so each physical field
    gets its own variance-normalized RMSE.

    Args:
        pred, target: [B, C, T, H, W]
        eps: numerical stability term.

    Returns:
        [C] tensor, one VRMSE per channel, averaged over the batch.
    """
    dims = tuple(range(2, pred.dim()))
    diff = pred - target
    mse = diff.square().mean(dim=dims)
    variance = target.var(dim=dims, unbiased=False)

    return torch.sqrt(mse / (variance + eps)).mean(dim=0)


def vrms_chmean(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Mean over channels of vrms_per_channel -- the scalar VRMSE everywhere in
    this repo. pred, target: [B, C, T, H, W]."""
    return vrms_per_channel(pred, target, eps=eps).mean()
