"""Configuration of the corrector stage."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from windinet.wan.vae.config import DataConfig, _Base, _Stage


class CorrectorModelConfig(_Base):
    history: int = Field(
        default=1,
        ge=0,
        description="own previous frames C(t-k..t-1) as input (C(0) = true frame 0, the earliest repeated while fewer "
        "exist); 0 = Wan's frames t-1, t, t+1 alone",
    )
    first_frame: bool = Field(default=False, description="the true frame 0 as input at every step")
    channels: tuple[int, ...] = Field(default=(64, 128, 128, 256), description="U-Net width per level, no attention")
    layers_per_block: int = Field(default=2, description="residual blocks per U-Net level")
    precision: Literal["mixed", "fp32"] = Field(default="mixed", description="mixed = fp32 weights, bf16 autocast")


class CorrectorTrainConfig(_Base):
    steps: int = Field(default=10000, description="updates; 0 = evaluate only")
    batch_sims: int = Field(default=32, description="windows per update (under torchrun: global, split over ranks)")
    window: int = Field(
        default=4,
        ge=1,
        description="frames per window with gradient; loss = their VRMSE^2",
    )
    teacher_forcing: bool = Field(
        default=False, description="own history = the true past frames (one step each) instead of the burn-in rollout"
    )
    history_noise: float = Field(
        default=0.0, ge=0, description="with teacher_forcing: Gaussian noise on the true history, in field stds"
    )
    loss: Literal["vrmse2", "mse"] = Field(
        default="vrmse2",
        description="vrmse2: VRMSE^2 against each window's own variance; mse: MSE of the fields z-scored with the "
        "dataset stats (The Well, PDEBench)",
    )
    ema: float = Field(default=0.0, description="EMA decay of the weights, evaluated and saved; 0 = off")
    lr: float = Field(default=2e-4, description="peak lr")
    weight_decay: float = Field(default=0.01, description="AdamW weight decay")
    warmup_steps: int = Field(default=500, description="linear warm-up from 1 % of the peak; 0 = none")
    lr_floor: float = Field(default=1e-2, description="end of the cosine decay as a fraction of the peak lr")
    max_grad_norm: float = Field(default=1.0, description="gradient clipping; 0 = off")
    eval_every: int = Field(default=1000, description="validation rollouts (and weight save) every n updates")
    log_every: int = Field(default=50, description="training-progress line every n updates")
    seed: int = Field(default=0, description="torch and window seed (+ rank under torchrun)")


class CorrectorStageConfig(_Stage):
    stage = "dit/corrector"
    name: str = Field(description="short run name: the results folder")
    description: str = Field(min_length=1, description="one sentence: what this run tests; shown in the results index")
    model: CorrectorModelConfig = CorrectorModelConfig()
    data: DataConfig = Field(description="the simulations: split, ground truth, stride, augment (D4 on the pairs)")
    train: CorrectorTrainConfig = CorrectorTrainConfig()
    rollouts: list[str] = Field(
        min_length=1,
        description="decoded Wan rollouts (decode_rollouts.py), one folder per sampling seed; the first is evaluated",
    )
    load: str | None = Field(default=None, description="corrector run folder to evaluate or continue (corrector.pt)")
    results_dir: str | None = Field(
        default=None, description="one folder per run is created here; None = results/wan/<data.dataset>/dit/corrector"
    )
    weights_dir: str | None = Field(
        default=None, description="where corrector.pt and the mp4 go, symlinked from the run; None = the run folder"
    )
