"""cfd_bench's metrics beside ours (Harish's scripts/eval_dit_vrmse.py --score_frames --clamp_prediction): RMSE, VRMSE,
1 - SSIM, MLW and the relative H1 and H2 seminorm errors (the last three per field, then mean) once per simulation on
frames 10, 20, ..., 100 in Li's clipped normalised space (thesis eq. 3.14), prediction and truth clamped to [-1, 1],
plain mean over simulations. The functions of windinet.losses, not copies."""

from __future__ import annotations

import torch

from windinet.eulermq.data import load_field_stats
from windinet.experiment.distributed import all_sum
from windinet.losses import (
    SSIMLoss,
    h1_seminorm_per_channel,
    h2_seminorm_per_channel,
    mlw_loss,
    rmse_loss,
    vrms_per_channel,
)
from windinet.training.shockwave_data import normalize_fields

FRAMES = range(10, 101, 10)  # Harish's frames (2026-10-26); before: 40, 50, ..., 100
SHARED = (20, 40, 60, 80)  # the FRAMES that our stride-2 rollouts and LoLA's stride-4 ones both reach
CLIP = 5.0  # Li's normalisation: (x - mean) / (CLIP std), clamped to [-1, 1]
NAMES = ("rmse", "vrmse", "ssim", "mlw", "h1", "h2")


def field_mean(per_channel):
    """A per-field metric of windinet.losses as one number: per field, then the mean (Li eq. 5.11, The Well)."""
    return lambda x_hat, x: per_channel(x_hat, x).mean()


class Benchmark:
    """`add` one simulation [N, C, H, W] in physical units (every `stride`-th original frame), `metrics` at the end;
    frames the trajectory does not reach are left out. `prefix`: of the metric keys (bench_vae for the round trip)."""

    def __init__(self, stats_json: str, fields: list[str], stride: int, device, prefix: str = "bench", frames=FRAMES):
        stats = load_field_stats(stats_json)
        self.mean, self.std = ([stats[f][k] for f in fields] for k in ("mean", "std"))
        self.stride, self.scored = stride, [f // stride for f in frames if f % stride == 0]
        vrmse, h1, h2 = (field_mean(f) for f in (vrms_per_channel, h1_seminorm_per_channel, h2_seminorm_per_channel))
        self.fns = (rmse_loss, vrmse, SSIMLoss(len(fields)).to(device), mlw_loss, h1, h2)
        self.prefix, self.n = prefix, torch.zeros((), device=device)
        self.sum, self.sum_sq = torch.zeros(2, len(NAMES), device=device)  # over sims: their mean and std

    def add(self, x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Scores one simulation; returns its metrics in the order of NAMES."""
        self.scored = [f for f in self.scored if f < len(x)]
        x_hat, x = (normalize_fields(t[self.scored], self.mean, self.std, CLIP).movedim(0, 1)[None] for t in (x_hat, x))
        m = torch.stack([fn(x_hat, x) for fn in self.fns])
        self.sum, self.sum_sq, self.n = self.sum + m, self.sum_sq + m**2, self.n + 1
        return m

    def metrics(self) -> dict:
        """<prefix>_<name> (mean over sims), <prefix>_<name>_std and <prefix>_frames (original frame numbers)."""
        all_sum(self.sum, self.sum_sq, self.n)
        mean, std = self.sum / self.n, (self.sum_sq / self.n - (self.sum / self.n) ** 2).clamp(min=0).sqrt()
        out = {f"{self.prefix}_{k}": v.item() for k, v in zip(NAMES, mean, strict=True)}
        out |= {f"{self.prefix}_{k}_std": v.item() for k, v in zip(NAMES, std, strict=True)}
        return out | {f"{self.prefix}_frames": [f * self.stride for f in self.scored]}
