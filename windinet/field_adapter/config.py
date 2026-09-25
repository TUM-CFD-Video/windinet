"""Configuration for field-adapter experiments. One YAML, variants via CLI overrides."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DataConfig(_Base):
    h5_path: str
    stats_json: str = Field(description="output of scripts/compute_channel_stats.py on h5_path")
    test_every: int = Field(default=5, description="every n-th sim per gamma is held out")
    frames_per_sim: int = Field(default=2, description="random frames read per simulation per step")
    num_workers: int = 4


class AdapterConfig(_Base):
    groups: list[list[str]] = Field(
        default=[["density", "pressure"], ["momentum_x", "momentum_y"]],
        description="field names per colour image, at most three each; one VAE pass per group",
    )
    load: str | None = Field(default=None, description="state_dict (.pt) of an earlier run")


class LossConfig(_Base):
    """Weights of the reconstruction terms, measured in z-scored field space."""

    rmse: float = 1.0
    h1: float = 1.0


class TrainConfig(_Base):
    steps: int = Field(default=600, description="0 = evaluate only")
    batch_sims: int = 1
    lr: float = 3e-3
    eval_every: int = 300
    seed: int = 0


class FieldAdapterConfig(_Base):
    name: str
    vae: Literal["none", "wan"] = Field(default="wan", description="'none' skips the VAE: adapter round trip only")
    data: DataConfig
    adapter: AdapterConfig = AdapterConfig()
    loss: LossConfig = LossConfig()
    train: TrainConfig = TrainConfig()
    ref_stats: str | None = Field(default=None, description="natural-video latent statistics from ref_latents.py; enables the Fréchet metric")
