"""Learning-rate schedule of the LTX fine-tuning recipe."""

from __future__ import annotations

import math
from collections.abc import Callable


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
