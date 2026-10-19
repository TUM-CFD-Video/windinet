"""cfd_bench's metrics beside ours (Harish's scripts/eval_dit_vrmse.py --score_frames --clamp_prediction): RMSE, VRMSE
(per field, then mean), 1 - SSIM and MLW once per simulation on frames 40, 50, ..., 100 in Li's clipped normalised
space (thesis eq. 3.14), prediction and truth clamped to [-1, 1], plain mean over simulations. Weihao's metric
functions, not copies."""

from __future__ import annotations

import torch

from windinet.eulermq.data import load_field_stats
from windinet.experiment.distributed import all_sum
from windinet.losses import SSIMLoss, mlw_loss, rmse_loss, vrms_per_channel
from windinet.training.shockwave_data import normalize_fields

FRAMES = range(40, 101, 10)  # what cfd_bench's autoregressive baseline predicts: 4 input frames, stride 10
CLIP = 5.0  # Li's normalisation: (x - mean) / (CLIP std), clamped to [-1, 1]
NAMES = ("rmse", "vrmse", "ssim", "mlw")


def vrmse(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Per field, then the mean (Li's thesis eq. 5.11, The Well), not Weihao's pooled vrms_loss."""
    return vrms_per_channel(x_hat, x).mean()


class Benchmark:
    """`add` one simulation [N, C, H, W] in physical units (every `stride`-th original frame), `metrics` at the end;
    frames the trajectory does not reach are left out."""

    def __init__(self, stats_json: str, fields: list[str], stride: int, device):
        stats = load_field_stats(stats_json)
        self.mean, self.std = ([stats[f][k] for f in fields] for k in ("mean", "std"))
        self.stride, self.scored = stride, [f // stride for f in FRAMES if f % stride == 0]
        self.fns = (rmse_loss, vrmse, SSIMLoss(len(fields)).to(device), mlw_loss)
        self.sum, self.n = torch.zeros(len(NAMES), device=device), torch.zeros((), device=device)

    def add(self, x_hat: torch.Tensor, x: torch.Tensor) -> None:
        self.scored = [f for f in self.scored if f < len(x)]
        x_hat, x = (normalize_fields(t[self.scored], self.mean, self.std, CLIP).movedim(0, 1)[None] for t in (x_hat, x))
        self.sum += torch.stack([fn(x_hat, x) for fn in self.fns])
        self.n += 1

    def metrics(self) -> dict:
        """bench_<name> and bench_frames (original frame numbers)."""
        all_sum(self.sum, self.n)
        out = {f"bench_{k}": v.item() for k, v in zip(NAMES, self.sum / self.n, strict=True)}
        return out | {"bench_frames": [f * self.stride for f in self.scored]}
