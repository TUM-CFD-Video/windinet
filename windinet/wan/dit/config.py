"""Configuration of the DiT stage: the latents it trains on, the Wan transformer, its conditioning, training,
sampling and evaluation. One schema per script: EncodeConfig, SampleConfig, DitStageConfig."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from windinet.wan import REPO
from windinet.wan.vae.config import DataConfig, _Base, _Stage


class EncodeConfig(_Base):
    """scripts/wan/encode_latents.py: every simulation of every split through adapter + encoder, once."""

    data: DataConfig
    load: str = Field(description="VAE-stage run folder: its adapter.pt and adapter groups (config.yaml)")
    out: str = Field(description="latent folder: <out>/<split>/<sim id>.pt and <out>/encode.yaml")
    sims: int | None = Field(default=None, description="first n sims per split (smoke); None = all")
    all_starts: bool = Field(
        default=False,
        description="also encode every train sim from each chunk boundary of its first half (<sim id>_t<frame>.pt): "
        "mid-trajectory single-frame starts for autoregressive training",
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
    results_dir: str = Field(
        default="results/wan/eulermq/dit/zero_shot", description="one folder per sample is created here"
    )
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
        description="gamma: the dataset's scalars (and direction) as cross-attention tokens (Li); gamma_adaln: its one "
        "scalar added to the time embedding that drives Wan's adaLN (cross-attention gets one zero token); text: a "
        "fixed text prompt",
    )
    context: int = Field(
        default=1,
        ge=0,
        description="clean latent frames at the start (timestep 0, no loss) that condition the rest: k = pixel frames "
        "0 to 4 (k - 1) x stride; 1 = the first frame alone, 0 = none",
    )
    tokens: int = Field(default=4, description="gamma: tokens per input")
    prompt: str = Field(default="", description="text: the prompt of every sample")
    negative_prompt: str = Field(default="", description="text: what guidance steers away from")

    @model_validator(mode="before")
    @classmethod
    def _range_is_a_dataset_fact(cls, data):
        if isinstance(data, dict) and "gamma_range" in data:  # old run configs, always (1, 2): now data.specs' ranges
            if list(data.pop("gamma_range")) != [1, 2]:
                raise ValueError("cond.gamma_range is gone; the range is data.specs' (1, 2)")
        return data


class LatentConfig(_Base):
    dir: str = Field(description="latent folder written by scripts/wan/encode_latents.py")
    frames: int | None = Field(
        default=None, description="latent frames per sample (the first n); None = whole trajectories (26)"
    )
    ar_window: int | None = Field(
        default=None,
        description="training on random windows of this many latent frames instead of whole trajectories; None = whole",
    )
    delta: bool = Field(
        default=False,
        description="the DiT generates every latent frame after the first as its change from the one before, scaled "
        "by the train std of such changes (measured at the start, delta.pt in the run); rollouts are summed back up",
    )


class DitTrainConfig(_Base):
    steps: int = Field(default=8000, description="updates; 0 = evaluate only")
    batch_sims: int = Field(default=32, description="simulations per update (under torchrun: global, split over ranks)")
    micro_batch: int = Field(default=4, description="simulations per backward pass per process")
    lr: float = Field(default=1e-5, description="peak lr of the transformer and the conditioning")
    noise_lr: float = Field(default=1e-3, description="peak lr of the shared noise with noise: learned")
    patchify_lr: float | None = Field(
        default=None,
        description="peak lr of the patchify and unpatchify layers when widened for several latents (widen_patchify); "
        "None = lr. Zero-initialised streams in a fresh run grow in faster with more than the backbone lr (1e-4)",
    )
    weight_decay: float = Field(default=0.01, description="AdamW weight decay")
    warmup_steps: int = Field(default=0, description="linear warm-up from 1 % of the peak; 0 = none")
    lr_floor: float = Field(
        default=1e-3, description="end of the cosine decay as a fraction of each peak lr (Li: 1e-8 / 1e-5)"
    )
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
    history_noise: float | None = Field(
        default=None,
        ge=0,
        description="Diffusion Forcing: per-frame timesteps, a random history of at least cond.context frames at one "
        "sigma in [0, this] (0 for a fifth of the samples, frame 0 clean) and out of the loss; None = one timestep, "
        "the context clean",
    )
    pixel_loss: float = Field(
        default=0.0,
        ge=0,
        description="weight of the pixel loss: L1 + gradient-difference L1 (Mathieu et al. 2016) in z-scored field "
        "space between the decoded clean-latent estimate x_σ - σ v and the decoded clean latent, for the "
        "pixel_gate_samples lowest-sigma latents under pixel_gate of each micro-batch; 0 = off",
    )
    pixel_gate: float = Field(
        default=0.8,
        gt=0,
        le=1,
        description="pixel loss only for latents noised at sigma <= this: the 2-step sampler evaluates at 1 and 0.75, "
        "and the estimate at sigma 1 is the prior",
    )
    pixel_gate_samples: int = Field(
        default=1,
        ge=1,
        description="latents per micro-batch and process that get the pixel loss: each decodes its 49 frames forward "
        "and backward, about two DiT steps of compute",
    )
    pixel_window: int | None = Field(
        default=None,
        ge=2,
        description="latent frames decoded for the pixel loss: a window starting at latent s >= 1 with probability "
        "proportional to s (late frames, where fronts drift, more often); None = all, about 4x the cost of 6",
    )
    front_loss: float = Field(
        default=0.0,
        ge=0,
        description="weight of the front-distance term added to L1 + GDL in the pixel loss: symmetric soft Chamfer "
        "between the density and pressure fronts of the decoded estimate and of the decoded clean latent; 0 = off",
    )
    unrolled: float | None = Field(
        default=None,
        gt=0,
        le=1,
        description="share of each micro-batch trained on its own state, as the sampler sees it: call k of an n-step "
        "chain from noise (the steps before it without gradient, Euler), its clean estimate against the clean latent "
        "(MSE); None = off",
    )
    unrolled_steps: tuple[int, int] = Field(
        default=(2, 5), description="range of the chain length n (uniform); the trained call k is uniform in 2..n"
    )
    late_weight: float = Field(
        default=0.0,
        ge=0,
        description="flow loss of latent frame f weighted 1 + late_weight * f / (F - 1), normalised to mean 1 over the "
        "target frames: later frames count more; 0 = all equal",
    )
    detail_weight: float = Field(
        default=0.0,
        ge=0,
        description="flow loss of each latent cell weighted 1 + detail_weight * its detail share (detail_masks), "
        "normalised to mean 1 like late_weight: fronts and vortices count more; 0 = all equal",
    )
    detail_masks: str | None = Field(
        default=None, description="folder of per-sim detail shares [F', h, w] (scripts/wan/detail_masks.py)"
    )
    sim_weights: str | None = Field(
        default=None,
        description="JSON {sim id: weight}: training sims drawn with these weights (their symmetric copies too) "
        "instead of uniformly; None = uniform",
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
    samples: int = Field(default=1, ge=1, description="rollouts per simulation, averaged in field space: ensemble")
    seed: int = Field(
        default=0, description="noise seed: every simulation draws its own from its id, so all runs share their noise"
    )
    chunk: int | None = Field(
        default=None,
        ge=1,
        description="latent frames kept per sampler run: the trajectory is generated again with the history grown by "
        "this many predicted frames, from cond.context on; None = one shot",
    )
    history_noise: float = Field(
        default=0, ge=0, description="sigma the context frames after frame 0 are re-noised to and marked with"
    )
    stochastic: float = Field(
        default=0,
        ge=0,
        le=1,
        description="share of fresh noise when each step re-noises the clean estimate to the next sigma (DDIM eta, "
        "Euler steps): fresh detail at every step instead of the mean; 0 = UniPC, deterministic",
    )


class PruneConfig(_Base):
    """Structured pruning of the loaded transformer before training (windinet/wan/dit/prune.py); null = keep."""

    depth: int | None = Field(default=None, ge=2, description="blocks kept, the first and the last among them")
    blocks: Literal["spread", "scan"] = Field(
        default="spread",
        description="which blocks: evenly spaced (every other at half depth), or the lowest velocity error when "
        "dropped alone on the calibration states",
    )
    hidden: int | None = Field(
        default=None, description="residual-stream channels kept (Minitron embedding axis); heads and FFN keep theirs"
    )
    heads: int | None = Field(default=None, ge=1, description="self-attention heads kept per block")
    cross_heads: int | None = Field(default=None, ge=1, description="cross-attention heads kept per block")
    ffn: int | None = Field(default=None, ge=1, description="FFN channels kept per block")
    calib_sims: int = Field(
        default=64, description="train sims of the calibration pass (global): their GT-noised and sampler-chain states"
    )
    compensate: bool = Field(
        default=True,
        description="least-squares refit of every layer whose inputs were cut on the calibration activations; a "
        "kept block followed by dropped ones adds their mean residual",
    )


class DistillConfig(_Base):
    """Knowledge distillation from a frozen teacher run, evaluated on the fly without gradients."""

    teacher: str = Field(description="DiT run folder of the teacher: its dit.pt and cond.pt")
    flow: float = Field(default=1.0, ge=0, description="weight of the flow loss on the ground truth (the anchor)")
    velocity: float = Field(
        default=1.0, ge=0, description="weight of the MSE to the teacher's velocity at the same inputs and states"
    )
    features: float = Field(
        default=0.0,
        ge=0,
        description="weight of the relative MSE between each block's output and the teacher's at the matching depth "
        "(the end of its group of dropped blocks; the kept channels when the stream is pruned)",
    )
    probe_sims: int = Field(default=8, description="val sims of the fixed batch on which the closeness is tracked")


class DitStageConfig(_Stage):
    stage = "dit"
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
    load_step: int | None = Field(
        default=None, description="with load: this eval point (dit_<step>.pt, cond_<step>.pt), not the final dit.pt"
    )
    prune: PruneConfig | None = Field(default=None, description="shrink the loaded transformer before training")
    distill: DistillConfig | None = Field(default=None, description="train against a frozen teacher run")
    benchmark: bool = Field(
        default=False,
        description="also cfd_bench's metrics (Harish): RMSE, VRMSE, 1-SSIM, MLW on frames 10, 20, ..., 100 in Li's "
        "clipped normalised space, prediction and truth clamped to [-1, 1]; bench_shared on the frames LoLA reaches",
    )
    score_from: int = Field(
        default=0,
        ge=0,
        description="first original frame the per-field metrics and frame windows score (never the context frames): "
        "the same frames for every context size; 0 = all predicted frames",
    )
    examples: list[Literal["gamma", "worst", "best"]] = Field(
        default=["gamma", "worst", "best"],
        description="test videos: the first sim at the lowest and the highest first scalar, the sim with the highest "
        "and the lowest VRMSE (mean over fields)",
    )
    noise: Literal["random", "fixed", "learned"] = Field(
        default="random",
        description="random: fresh N(0, 1) per sample; fixed: one seeded N(0, 1) draw (sampling.seed) shared by every "
        "sample in training and sampling, a regression model; learned: that draw trained with train.noise_lr. The run "
        "saves it as noise.pt and a later run loads it from `load`",
    )
    rollouts: str | None = Field(
        default=None,
        description="write the rollout latents of every train sim and of the evaluated val and test sims to "
        "<this>/<split>/<sim id>.pt (the latent folder layout) and stop: the VAE stage's data.latents",
    )
    results_dir: str | None = Field(
        default=None, description="one folder per run is created here; None = results/wan/<data.dataset>/dit"
    )
    weights_dir: str | None = Field(
        default=None,
        description="where dit.pt (5.7 GB), cond.pt and the mp4 go (the results tree mirrored), symlinked from the "
        "run; None = the run folder",
    )

    @model_validator(mode="after")
    def _windows_continue_from_context(self):
        if self.latents.ar_window is not None and not 0 < self.cond.context < self.latents.ar_window:
            raise ValueError("latents.ar_window needs 0 < cond.context < ar_window: windows continue from the last")
        if self.train.history_noise is not None and self.cond.context < 1:
            raise ValueError("train.history_noise needs cond.context >= 1: frame 0 stays clean")
        if self.train.front_loss and not self.train.pixel_loss:
            raise ValueError("train.front_loss is part of the pixel loss: needs train.pixel_loss > 0")
        if self.latents.delta and self.latents.ar_window is not None:
            raise ValueError("latents.delta scales the change into latent 1 apart: windows would start mid-trajectory")
        if self.latents.delta and self.train.pixel_loss:
            raise ValueError("latents.delta: the pixel loss would decode changes, not states; set train.pixel_loss 0")
        if self.train.pixel_loss and self.train.history_noise is not None:
            raise ValueError("train.pixel_loss needs one sigma per latent: train.history_noise None")
        return self
