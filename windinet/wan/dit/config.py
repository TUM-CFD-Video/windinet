"""Configuration of the DiT stage: the latents it trains on, the Wan transformer, its conditioning, training,
sampling and evaluation. One schema per script: EncodeConfig, SampleConfig, DitStageConfig."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from windinet.wan import REPO
from windinet.wan.vae.config import DataConfig, _Base


class EncodeConfig(_Base):
    """scripts/wan/encode_latents.py: every simulation of every split through adapter + encoder, once."""

    data: DataConfig
    load: str = Field(description="VAE-stage run folder: its adapter.pt and adapter groups (config.yaml)")
    out: str = Field(description="latent folder: <out>/<split>/<sim id>.pt and <out>/encode.yaml")
    sims: int | None = Field(default=None, description="first n sims per split (smoke); None = all")
    augmentations: int = Field(
        default=1,
        description="copies of every train simulation under the square's symmetries (eulermq.data.symmetry, exact for "
        "the Euler equations): 1 = as recorded, 8 = every rotation and reflection; files <sim id>_s<k>.pt",
    )


class SampleConfig(_Base):
    """scripts/wan/sample_dit.py: one video from the pretrained or a fine-tuned Wan DiT, text-conditioned."""

    name: str = Field(description="short run name: the results folder")
    description: str = Field(min_length=1, description="one sentence: what this sample shows")
    prompt: str = Field(description="text conditioning")
    negative_prompt: str = Field(default="", description="what classifier-free guidance steers away from")
    height: int = Field(default=480, description="pixels, multiple of 16; Wan 1.3B was trained at 832x480")
    width: int = Field(default=832, description="pixels, multiple of 16")
    frames: int = Field(default=81, description="video frames, 4k + 1")
    steps: int = Field(default=50, description="denoising steps (UniPC)")
    guidance: float = Field(default=5.0, description="classifier-free guidance scale; 1 = off")
    seed: int = Field(default=0, description="noise seed")
    fps: int = Field(default=16, description="playback rate of the mp4")
    source: str | None = Field(
        default=None,
        description="'<h5 file>:<sim id>': start from this simulation's latents (its adapter image A); size follows it",
    )
    adapter: str | None = Field(default=None, description="VAE-stage run folder whose adapter renders `source`")
    strength: float = Field(
        default=1.0, description="how far `source` is noised before denoising: 1 = ignored, 0 = reconstructed only"
    )
    keep_first_frame: bool = Field(
        default=False, description="keep `source`'s first latent frame clean (timestep 0), generate the rest"
    )
    load: str | None = Field(default=None, description="DiT run folder (dit.pt); None = pretrained Wan 2.1")
    results_dir: str = Field(default="results/wan/dit/zero_shot", description="one folder per sample is created here")
    video_dir: str | None = Field(
        default=None, description="where the mp4 is written, symlinked from the run folder; None = the run folder"
    )


class ModelConfig(_Base):
    repo: str = Field(default=REPO, description="Hugging Face repo of the Wan 2.1 pipeline")
    precision: Literal["mixed", "fp32", "bf16"] = Field(
        default="mixed", description="mixed = fp32 weights, bf16 matmuls (autocast); bf16 = weights in bf16 too"
    )
    checkpointing: bool = Field(default=True, description="recompute each block in backward: memory for speed")


class CondConfig(_Base):
    kind: Literal["gamma", "gamma_adaln", "text"] = Field(
        default="gamma",
        description="gamma as cross-attention tokens (Li), gamma added to the time embedding that drives Wan's adaLN "
        "(cross-attention gets one zero token), or a fixed text prompt",
    )
    first_frame: bool = Field(
        default=True, description="the first latent frame stays clean (timestep 0, no loss) and conditions the rest"
    )
    tokens: int = Field(default=4, description="gamma: tokens produced by the embedding")
    gamma_range: tuple[float, float] = Field(default=(1.0, 2.0), description="gamma: scaled to [0, 1] over this range")
    prompt: str = Field(default="", description="text: the prompt of every sample")
    negative_prompt: str = Field(default="", description="text: what guidance steers away from")


class LatentConfig(_Base):
    dir: str = Field(description="latent folder written by scripts/wan/encode_latents.py")
    frames: int | None = Field(
        default=None, description="latent frames per sample (the first n); None = whole trajectories (26)"
    )


class DitTrainConfig(_Base):
    steps: int = Field(default=8000, description="updates; 0 = evaluate only")
    batch_sims: int = Field(default=32, description="simulations per update (under torchrun: global, split over ranks)")
    micro_batch: int = Field(default=4, description="simulations per backward pass per process")
    lr: float = Field(default=1e-5, description="peak lr of the transformer and the conditioning")
    patchify_lr: float | None = Field(
        default=None,
        description="peak lr of the patchify and unpatchify layers when widened for several latents (widen_patchify); "
        "None = lr. Zero-initialised streams in a fresh run grow in faster with more than the backbone lr (1e-4)",
    )
    weight_decay: float = Field(default=0.01, description="AdamW weight decay")
    warmup_steps: int = Field(default=0, description="linear warm-up from 1 % of the peak; 0 = none")
    lr_floor: float = Field(default=1e-3, description="final lr as a fraction of the peak (Li: 1e-8 / 1e-5)")
    max_grad_norm: float = Field(default=1.0, description="gradient clipping; 0 = off")
    ema: float = Field(
        default=0.0,
        description="decay of the weight EMA that is evaluated and saved instead of the raw weights (warmed up as "
        "min(ema, (1 + n) / (10 + n))); the raw weights are evaluated beside it (raw_* metrics); 0 = off",
    )
    sigma_mean: float = Field(default=0.0, description="logit-normal timestep draw: mean of the logit")
    sigma_std: float = Field(default=1.0, description="logit-normal timestep draw: std of the logit")
    sigma_shift: float = Field(
        default=3.0, description="time shift of the draw, as in the sampler: Wan's scheduler default (3 at 480p)"
    )
    eval_every: int = Field(default=1000, description="validation rollouts (and weight save) every n updates")
    log_every: int = Field(default=10, description="training-progress line every n updates")
    seed: int = Field(default=0, description="torch seed (+ rank under torchrun)")


class SamplingConfig(_Base):
    steps: int = Field(default=20, description="denoising steps of a rollout (UniPC)")
    shift: float = Field(
        default=3.0,
        description="time shift of the sampler sigmas (Wan: 3 at 480p, 5 at 720p); training draw: train.sigma_shift",
    )
    guidance: float = Field(default=1.0, description="classifier-free guidance scale; 1 = off")
    batch: int = Field(default=4, description="simulations per rollout batch in evaluation")


class DitStageConfig(_Base):
    name: str = Field(description="short run name: the results folder")
    description: str = Field(min_length=1, description="one sentence: what this run tests; shown in the results index")
    model: ModelConfig = ModelConfig()
    cond: CondConfig = CondConfig()
    latents: LatentConfig
    data: DataConfig = Field(description="the simulations: split and ground truth of the evaluation (as in vae.yaml)")
    train: DitTrainConfig = DitTrainConfig()
    sampling: SamplingConfig = SamplingConfig()
    vae: str = Field(description="VAE-stage run folder: the adapter and the (fine-tuned) decoders behind the latents")
    load: str | None = Field(default=None, description="DiT run folder to continue from: its dit.pt and cond.pt")
    results_dir: str = Field(default="results/wan/dit", description="one folder per run is created here")
    weights_dir: str | None = Field(
        default=None, description="where dit.pt (5.7 GB) and cond.pt go, symlinked from the run folder; None = the run"
    )
