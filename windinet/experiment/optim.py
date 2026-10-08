"""Optimizer, learning-rate schedule, weight EMA and weight-change measure shared by the stage scripts."""

from __future__ import annotations

import math
from collections.abc import Callable

import torch


def warmup_cosine(warmup_steps: int, steps: int, floor: float) -> Callable[[int], float]:
    """Per-step lr multiplier: linear warm-up from 1 % of the peak (none if warmup_steps is 0), then cosine decay to
    `floor` at the last step."""

    def factor(step: int) -> float:
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


class EMA:
    """Exponential moving average of `params` (fp32), decay warmed up as min(decay, (1 + n) / (10 + n))."""

    def __init__(self, params: list[torch.nn.Parameter], decay: float):
        if any(p.dtype != torch.float32 for p in params):
            raise ValueError("EMA needs fp32 weights: a bf16 average stops moving at decay 0.999")
        self.params, self.decay, self.n = params, decay, 0
        self.average = [p.detach().clone() for p in params]

    @torch.no_grad()
    def update(self) -> None:
        self.n += 1
        decay = min(self.decay, (1 + self.n) / (10 + self.n))
        torch._foreach_lerp_(self.average, [p.detach() for p in self.params], 1 - decay)

    def swap(self) -> None:
        """Exchange the weights and the average in place; twice restores the weights."""
        for p, a in zip(self.params, self.average, strict=True):
            p.data, a.data = a.data, p.data


def rel_change(params, ref) -> float:
    """||p - ref|| / ||ref|| over a list of tensors: how far weights moved. Computed where `ref` lives."""
    num = sum((p.detach().to(r.device) - r).square().sum() for p, r in zip(params, ref, strict=True))
    return (num / sum(r.square().sum() for r in ref)).sqrt().item()
