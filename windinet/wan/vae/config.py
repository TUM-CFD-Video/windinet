"""Configuration of the VAE stage: the field adapter and the Wan VAE behind it. One YAML, variants via CLI overrides."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from windinet.data import frames


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DataConfig(_Base):
    h5_path: str = Field(description="training file; also the evaluation file unless test_h5 is set")
    test_h5: str | None = Field(
        default=None, description="separate test file; the training file is then split into train + val"
    )
    stats_json: str = Field(description="output of scripts/compute_channel_stats.py: field means, stds and ranges")
    val_sims: int = Field(default=500, description="with test_h5: sims held out of the training file for validation")
    split_seed: int = Field(default=42, description="with test_h5: seed of the train/val shuffle; 42 = LTX baseline")
    eval_sims: int = Field(default=100, description="validation sims evaluated mid-run (the first n)")
    test_sims: int | None = Field(default=None, description="test sims in the final eval (the first n); None = all")
    test_every: int = Field(default=5, description="without test_h5: every n-th sim per gamma is held out")
    test_gamma: float | None = Field(
        default=None, description="without test_h5: hold out this whole gamma instead (extrapolation test)"
    )
    frames_per_sim: int = Field(
        default=2,
        description="frames read per simulation: random ones in a training step, fixed ones in validation (clip: "
        "consecutive, one clip, 4k + 1)",
    )
    clip: bool = Field(
        default=False, description="the frames are consecutive and pass the VAE as one clip, not as single frames"
    )
    frame_stride: int = Field(
        default=1,
        description="every n-th frame: in clips, and in whole trajectories (evaluation, latent encoding; 4k + 1 kept)",
    )
    augment: bool = Field(
        default=False,
        description="training data under the dataset's symmetries (data.frames.symmetry): VAE-stage frames and clips "
        "under a random one, latent encoding stores every one, corrector: the Wan/truth pairs under one random one; "
        "evaluation never",
    )
    num_workers: int = Field(default=4, description="data-loader worker processes per training process")
    latents: str | None = Field(
        default=None,
        description="folder of DiT rollout latents (train_dit's rollouts): train and evaluate the decoders on decoding "
        "them against the same sims' whole trajectories (frame_stride), not on the round trip; the sims it holds",
    )

    @property
    def clip_len(self) -> int:
        """Frames per VAE pass in training: the clip, or 1 for single frames."""
        return self.frames_per_sim if self.clip else 1

    def splits(self) -> dict[str, tuple[str, list[str]]]:
        """{'train' | 'val' | 'test': (h5 file, sim ids)}, see data.frames.splits."""
        return frames.splits(
            self.h5_path, self.test_h5, self.val_sims, self.split_seed, self.test_every, self.test_gamma
        )


class AdapterConfig(_Base):
    groups: list[list[str]] = Field(
        default=[["density", "pressure"], ["momentum_x", "momentum_y"]],
        description="field names per image (up to three: a colour image; more widen the VAE's in and out convolutions "
        "to that many channels); one VAE pass per group; other fields dropped",
    )


class LossConfig(_Base):
    rmse: float = Field(default=1.0, description="weight of the RMSE term, measured in z-scored field space")
    h1: float = Field(default=1.0, description="weight of the H1 seminorm term (spatial gradients), same space")
    latent: float = Field(
        default=0.0,
        description="weight of the latent anchor: per channel the squared batch mean and (std - 1)^2 of the normalised "
        "latents, toward the N(0, 1) of natural video that Wan's DiT was trained on; 0 = off",
    )
    kind: Literal["rmse_h1", "l1_gdl", "amse"] = Field(
        default="rmse_h1",
        description="rmse_h1: the weighted RMSE + H1; l1_gdl: L1 + gradient-difference L1 (Mathieu et al. "
        "2016); amse: adjusted MSE per wavenumber shell (Subich et al. 2025), does not reward blur; same space",
    )


class TrainConfig(_Base):
    steps: int = Field(default=600, description="0 = evaluate only")
    batch_sims: int = Field(default=1, description="simulations per update (under torchrun: global, split over ranks)")
    micro_batch: int = Field(default=1, description="clips (frames without data.clip) per backward pass")
    adapter_lr: float = Field(default=3e-3, description="peak lr of the adapter; 0 = adapter fixed")
    vae_parts: Literal["none", "io", "decoder", "all"] = Field(
        default="none",
        description="which part of the Wan VAE to fine-tune alongside the adapter; io: its in and out convolutions",
    )
    precision: Literal["fp32", "mixed"] = Field(
        default="fp32", description="mixed: bf16 autocast in the VAE passes of training, fp32 weights; evaluation fp32"
    )
    checkpointing: bool = Field(
        default=True, description="recompute the VAE blocks in backward: less memory, about a third more time"
    )
    vae_per_group: bool = Field(
        default=False, description="one VAE (and so one fine-tuned decoder) per image group instead of one shared"
    )
    vae_lr: float = Field(default=5e-5, description="peak lr of the VAE parameters")
    io_lr: float | None = Field(
        default=None, description="peak lr of the VAE's in and out convolutions (the field interface); None = vae_lr"
    )
    warmup_steps: int = Field(
        default=0,
        description="linear warm-up from 1 % of the peak, then cosine decay (the LTX recipe); 0 = constant lr",
    )
    lr_floor: float = Field(
        default=0.02, description="end of the cosine decay as a fraction of each peak lr (LTX: 1e-6 / 5e-5)"
    )
    max_grad_norm: float = Field(default=0.0, description="gradient clipping; 0 = off")
    eval_every: int = Field(default=300, description="validation (and weight save) every n updates")
    log_every: int = Field(default=10, description="training-progress line every n updates")
    seed: int = Field(default=0, description="torch and frame-sampling seed (+ rank under torchrun)")


class VaeStageConfig(_Base):
    name: str = Field(description="short run name: the results folder")
    description: str = Field(min_length=1, description="one sentence: what this run tests; shown in the results index")
    vae: Literal["none", "wan"] = Field(default="wan", description="'none' skips the VAE: adapter round trip only")
    data: DataConfig
    adapter: AdapterConfig = AdapterConfig()
    loss: LossConfig = LossConfig()
    train: TrainConfig = TrainConfig()
    load: str | None = Field(
        default=None,
        description="run folder to continue from: its adapter.pt and, if present, vae.pt or one vae_<group>.pt per VAE",
    )
    results_dir: str = Field(default="results/wan/vae", description="one folder per run is created here")
    weights_dir: str | None = Field(
        default=None,
        description="where vae.pt (280+ MB) is written, with a symlink in the run folder; None = the run folder itself",
    )
