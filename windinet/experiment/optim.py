"""Optimizer and learning-rate schedule shared by the stage scripts."""

from __future__ import annotations

import math
from collections.abc import Callable

import torch


def warmup_cosine(warmup_steps: int, steps: int, floor: float) -> Callable[[int], float]:
    """Per-step lr multiplier: linear warm-up from 1 % of the peak, then cosine decay to `floor`; 1 without warm-up."""

    def factor(step: int) -> float:
        if not warmup_steps:
            return 1.0
        if step < warmup_steps:
            return 0.01 + 0.99 * step / warmup_steps
        progress = (step - warmup_steps) / max(1, steps - warmup_steps)
        return floor + (1 - floor) * (1 + math.cos(math.pi * progress)) / 2

    return factor


def adamw(
    groups: list[dict], steps: int, warmup_steps: int, lr_floor: float, weight_decay: float = 0.0
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    """AdamW over `groups` (each {"params", "lr"}) with the warm-up + cosine schedule of the LTX recipe."""
    opt = torch.optim.AdamW(groups, weight_decay=weight_decay)
    return opt, torch.optim.lr_scheduler.LambdaLR(opt, warmup_cosine(warmup_steps, steps, lr_floor))
