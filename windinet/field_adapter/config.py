"""Configuration for field-adapter experiments. One YAML, sweeps via CLI overrides."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DataConfig(_Base):
    h5_path: str = Field(description="euler_mq HDF5 with one group per simulation")
    stats_json: str = Field(description="output of scripts/compute_channel_stats.py on the same file")
    test_every: int = Field(default=5, ge=2, description="every n-th sim per gamma is held out")
    frames_per_sim: int = Field(default=4, ge=1, description="random frames read per simulation per step")
    num_workers: int = 4


class AdapterConfig(_Base):
    grouping: Literal["single", "triplets", "pairs"] = Field(default="single", description="'single': one 4->3 adapter; 'triplets': (m_x, m_y, p) and (rho, p, E), one exact 3->3 each; 'pairs': (rho, p) and (m_x, m_y), one exact 2->3 each")
    warp: bool = Field(default=True, description="learn per-channel monotone splines (False: PreNorm -> Mix -> SoftClip only)")
    bins: int = Field(default=8, ge=2)
    init: Literal["pca", "naive"] = Field(default="pca", description="Mix init: PCA -> luma/opponent axes, or naive = first three z-scored fields straight to RGB (4th dropped)")
    load: str | None = Field(default=None, description="state_dict (.pt) of an earlier run; skips init_from_data")


class LossConfig(_Base):
    """Weights; 0 disables the term. Reconstruction terms act in z-scored field space."""

    rmse: float = 1.0
    h1: float = 1.0
    slope_penalty: float = 1e-3
    rgb_moment: float = Field(default=0.0, description="match RGB per-channel mean/var to natural images")
    latent_moment: float = 0.0
    latent_cov: float = 0.0
    latent_psd: float = 0.0


class TrainConfig(_Base):
    steps: int = 2000
    batch_sims: int = 1
    lr: float = 1e-3
    eval_every: int = 250
    seed: int = 0


class FieldAdapterConfig(_Base):
    name: str = Field(description="run name; results land in <results_dir>/<name>.json")
    vae: Literal["none", "wan"] = Field(default="none", description="'none': A^-1(A(x)) only; 'wan': A -> frozen Wan VAE -> A^-1")
    data: DataConfig
    adapter: AdapterConfig = AdapterConfig()
    loss: LossConfig = LossConfig()
    train: TrainConfig = TrainConfig()
    results_dir: str = "results/field_adapter"
    ref_stats: str | None = Field(default=None, description="reference latent/RGB stats (.pt from scripts/field_adapter/ref_latents.py); enables Fréchet/moment metrics and latent losses")
    ablations_md: str = "docs/field_adapter/ablations.md"
