"""MAGI-1 4.5B DiT (official code) with the gamma embedding in the T5 slot, its flow matching, and sampling.

Flow matching as MAGI-1 defines it: t = 0 is noise, t = 1 clean, x_t = t * x + (1 - t) * noise, velocity
= x - noise, one timestep per chunk of 6 latent frames (24 frames).
Latent layout (data.build_latent_sequence): [ image | body latents (frames 1..F) | padding ]. The image
latent (frame 0) is the condition: kept clean in the input, never scored.

Training: all chunks in one forward pass, each at its own t, earlier chunks cleaner. With --clean-chunks some
samples start with clean chunks (MAGI-1's video-continuation task, report sec. 2.2.3: up to 5% noise on them,
no loss on them), as often as the windowed sampler meets finished chunks (sample_num_clean_chunks).

Sampling (rollout), MAGI-1's pipeline (video_generate.py generate_sequences): up to `window` chunks are
denoised at once, each num_steps / window Euler steps further than the next; a new chunk starts from pure
noise (t = 0) whenever the oldest one in the window is finished (t = 1), and finished chunks are fed back as
clean context (block-causal attention lets chunk k see chunks 0..k). window = 1 is one chunk at a time. In
chunk 0 the first latent frame is overwritten with the clean image latent before every step (MAGI-1's
image-to-video "prefix inpainting", video_generate.py try_pad_prefix_video).
"""

import torch
import torch.nn as nn
from inference.common import MagiConfig
from inference.model.dit import get_dit
from peft import LoraConfig, inject_adapter_in_model
from safetensors.torch import load_file
from torch.utils.checkpoint import checkpoint

from magi_finetuning.settings import CHUNK_WIDTH, CKPT, LORA_TARGETS, MAGI_CONFIG, SAMPLER_WINDOW
from windinet.config import ScalarConditioningConfig
from windinet.scalar_embeddings import ScalarEmbedding


# ---------------------------------------------------------------- model


def load_magi_config():
    """MAGI-1 4.5B config, pointed at our weights, without context parallelism."""
    cfg = MagiConfig.from_json(str(MAGI_CONFIG))
    cfg.runtime_config.load = str(CKPT / "magi" / "4.5B_base")
    cfg.engine_config.cp_strategy = "none"
    return cfg


def load_dit():
    """MAGI's own config + get_dit (loads weights, fp32 promotion, eval()). Returns (dit, config)."""
    cfg = load_magi_config()
    # eval() stays on during training: MAGI's only train-mode behaviour is T5 caption dropout,
    # which breaks on 4 scalar tokens (the DiT has no dropout layers).
    return get_dit(cfg), cfg


def enable_gradient_checkpointing(dit) -> None:
    """Recompute each transformer layer in backward instead of storing its activations."""
    for layer in dit.videodit_blocks.layers:
        layer.forward = lambda *a, _f=layer.forward, **k: checkpoint(_f, *a, use_reentrant=False, **k)


def gamma_embedding():
    return ScalarEmbedding(ScalarConditioningConfig(
        enabled=True, scalar_names=["gamma"], scalar_ranges={"gamma": (1.0, 2.0)}, embedding_dim=4096,
    )).cuda()


class Model(nn.Module):
    """DiT + scalar embedding as one module so DDP syncs both."""

    def __init__(self, dit, scalar_emb):
        super().__init__()
        self.dit, self.scalar_emb = dit, scalar_emb

    def forward(self, x, t, gamma, n_chunks, tok_per_chunk):
        y = self.scalar_emb(gamma)[:, None].expand(n_chunks, 1, -1, -1)  # (chunks, 1, L, 4096): T5 slot
        mask = torch.ones(n_chunks, 1, y.shape[2], device=x.device)
        # Block-causal: chunk i attends to chunks 0..i (MAGI's generate_default_kvrange layout).
        kv_range = torch.tensor([[0, (i + 1) * tok_per_chunk] for i in range(n_chunks)], device=x.device, dtype=torch.int32)
        return self.dit(x, t, y, caption_dropout_mask=torch.zeros(1, dtype=torch.bool, device=x.device),
                        xattn_mask=mask, kv_range=kv_range, inference_params=None,
                        range_num=n_chunks, denoising_range_num=n_chunks, slice_point=0, fwd_extra_1st_chunk=False)


# ---------------------------------------------------------------- flow matching (training)


CLEAN_CHUNK_NOISE = 0.05  # clean context chunks get up to 5% noise against exposure bias (MAGI-1 report sec. 2.2.3)


def sample_num_clean_chunks(n_chunks: int, window: int = SAMPLER_WINDOW) -> int:
    """How many leading chunks are clean context in one training sample, drawn as often as the windowed
    sampler (rollout) meets each count: of its n_chunks + window - 1 stages, the first `window` have no
    finished chunk, then one more chunk is finished per stage. 5 chunks, window 4: 0 clean in 4 of the 8
    stages, 1, 2, 3 and 4 clean in one stage each.
    Drawn on the GPU like the noise: the CPU generator is also used by the DataLoader when a resumed job
    creates its iterator, which would shift every later draw (resume would no longer be exact)."""
    stage = int(torch.randint(n_chunks + window - 1, (), device="cuda"))
    return max(0, stage - window + 1)


def sample_chunk_t(n_chunks: int, shift: float, num_clean: int = 0) -> torch.Tensor:
    """(1, n_chunks): MAGI's sd3-shifted t, sorted so earlier chunks are cleaner. The first num_clean chunks
    are clean context: t in [1 - CLEAN_CHUNK_NOISE, 1]."""
    u = torch.rand(1, n_chunks - num_clean, device="cuda")
    noisy = (u / (shift - (shift - 1) * u)).sort(dim=1, descending=True).values
    if num_clean == 0:
        return noisy
    clean = 1 - CLEAN_CHUNK_NOISE * torch.rand(1, num_clean, device="cuda")
    return torch.cat([clean, noisy], dim=1)


def add_noise(x1, t, noise):
    """x_t = t * x1 + (1 - t) * noise, with each chunk's own t."""
    tf = t.repeat_interleave(CHUNK_WIDTH, dim=1)[:, None, :, None, None]
    return tf * x1 + (1 - tf) * noise


def keep_image_clean(x, latents):
    """Overwrite latent frame 0 with the clean image latent (MAGI-1's image-to-video prefix inpainting)."""
    x = x.clone()
    x[:, :, :1] = latents[:, :, :1]
    return x


def velocity_loss(v, x1, noise, num_body: int, num_clean: int = 0):
    """MSE of the predicted velocity on the body latents (frames 1..F) of the noisy chunks only: not the
    given image latent (frame 0), not the padding, not the first num_clean chunks (clean context)."""
    body = slice(max(1, num_clean * CHUNK_WIDTH), 1 + num_body)
    return (v.float() - (x1 - noise))[:, :, body].pow(2).mean()


# ---------------------------------------------------------------- sampling (evaluation)


def time_schedule(num_steps: int, shift: float, device) -> torch.Tensor:
    """num_steps + 1 times from 0 to 1, crowded toward the noisy end. MAGI-1's sd3 schedule, the same
    shift as training's timestep sampling (sample_chunk_t)."""
    u = torch.linspace(0, 1, num_steps + 1, device=device)
    return u / (shift - (shift - 1) * u)


@torch.no_grad()
def rollout(model, image_latent, gamma, num_chunks: int, tokens_per_chunk: int, num_steps: int = 64,
            shift: float = 3.0, clean_t: float = 0.9999, window: int = SAMPLER_WINDOW, use_autocast: bool = False):
    """image_latent: (1, 16, 1, h, w) clean latent of frame 0 (already scaled).
    Returns all generated latents (1, 16, 6 * num_chunks, h, w); latent 0 is the image itself.

    Stage s (of num_chunks + window - 1) starts chunk s (if any) from noise and runs num_steps / window Euler
    steps on every unfinished chunk k <= s, chunk k being at step (s - k) * steps_per_stage + i. Window 4,
    64 steps: the newest chunk is at t 0 -> 0.10 while the one before is at 0.10 -> 0.25, then 0.25 -> 0.5
    and 0.5 -> 1 (the staircase training sees); older chunks are finished, clean context."""
    if num_steps % window:
        raise ValueError(f"num_steps ({num_steps}) must be a multiple of window ({window})")
    steps_per_stage = num_steps // window
    times = time_schedule(num_steps, shift, image_latent.device)
    chunk_shape = image_latent.shape[:2] + (CHUNK_WIDTH,) + image_latent.shape[3:]
    chunks = []  # every chunk started so far, finished or not

    for stage in range(num_chunks + window - 1):
        if stage < num_chunks:
            chunks.append(torch.randn(chunk_shape, device=image_latent.device))
        first_active = max(0, stage - window + 1)  # chunks before it are finished
        active = range(first_active, len(chunks))
        for i in range(steps_per_stage):
            chunks[0][:, :, :1] = image_latent  # the given frame 0 stays clean (also as context)
            steps = torch.tensor([(stage - k) * steps_per_stage + i for k in active], device=image_latent.device)
            chunk_times = torch.full((1, len(chunks)), clean_t, device=image_latent.device)  # finished: clean
            chunk_times[0, first_active:] = times[steps]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_autocast):
                velocity = model(torch.cat(chunks, dim=2), chunk_times, gamma, len(chunks), tokens_per_chunk)
            for k, step in zip(active, steps):
                v = velocity[:, :, k * CHUNK_WIDTH:(k + 1) * CHUNK_WIDTH].float()
                chunks[k] = chunks[k] + v * (times[step + 1] - times[step])
    chunks[0][:, :, :1] = image_latent

    return torch.cat(chunks, dim=2)


# ---------------------------------------------------------------- loading a trained DiT (train_lora.py / train_full.py checkpoints)


def load_trained_dit(checkpoint: str):
    """DiT + gamma embedding from a train_lora.py or train_full.py checkpoint (type detected from its keys).

    Returns (model, magi_config, is_full_finetune).
    """
    state = load_file(checkpoint)
    is_lora = any("lora_" in name for name in state)
    dit, magi_config = load_dit()

    if is_lora:
        # Rank from the saved adapter shapes, so it always matches the run.
        rank = next(t.shape[0] for name, t in state.items() if "lora_A" in name)
        inject_adapter_in_model(LoraConfig(r=rank, lora_alpha=rank, target_modules=LORA_TARGETS), dit)
    else:
        dit = dit.float()  # full-FT checkpoints hold every weight; compute runs under bf16 autocast

    model = Model(dit, gamma_embedding())
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise ValueError(f"{checkpoint}: weights the model doesn't have: {unexpected[:5]}")
    # Trainers save parameters only; buffers (e.g. the gamma embedding's fixed range and Fourier
    # frequencies) are rebuilt from the config, so only missing *parameters* are an error.
    parameter_names = {name for name, _ in model.named_parameters()}
    missing = [name for name in missing if name in parameter_names]
    if is_lora:  # only adapters + gamma embedding are saved; everything else is the released weights
        missing = [name for name in missing if "lora_" in name or name.startswith("scalar_emb")]
    if missing:
        raise ValueError(f"{checkpoint}: trained weights missing: {missing[:5]}")

    model.cuda().eval().requires_grad_(False)
    return model, magi_config, not is_lora
