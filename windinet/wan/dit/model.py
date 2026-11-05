"""Wan 2.1 text-to-video transformer: loading, text embeddings, the velocity forward, and the sampling loop.

The transformer sees normalised latents [B, 16 x num_latents, F', H/8, W/8] (what WanVAE.encode returns, the latents of
a sample stacked in the channels) and a conditioning tensor [B, L, 4096] through cross-attention
(windinet.wan.dit.conditioning builds it). Timesteps can be given per latent frame, which is how a clean conditioning
frame is marked (timestep 0).
"""

from __future__ import annotations

from pathlib import Path

import torch
from diffusers import UniPCMultistepScheduler, WanTransformer3DModel
from torch import nn
from transformers import T5TokenizerFast, UMT5EncoderModel

from windinet.wan import REPO


def encode_text(prompts: list[str], device: torch.device | str, repo: str = REPO, max_len: int = 512) -> torch.Tensor:
    """UMT5 embeddings [len(prompts), max_len, 4096], padding zeroed as in the Wan pipeline. Loads and frees the
    11 GB encoder: call once, keep the result."""
    tokenizer = T5TokenizerFast.from_pretrained(repo, subfolder="tokenizer")
    encoder = UMT5EncoderModel.from_pretrained(  # shard by shard onto the device: the host never holds the 22 GB
        repo, subfolder="text_encoder", torch_dtype=torch.bfloat16, device_map={"": str(device)}
    )
    tokens = tokenizer(prompts, padding="max_length", max_length=max_len, truncation=True, return_tensors="pt")
    mask = tokens.attention_mask.to(device)
    with torch.no_grad():
        embeds = encoder(tokens.input_ids.to(device), attention_mask=mask).last_hidden_state * mask[..., None]
    del encoder
    torch.cuda.empty_cache()
    return embeds


@torch.no_grad()
def widen_patchify(transformer: WanTransformer3DModel, num_latents: int) -> list[nn.Parameter]:
    """`num_latents` latents stacked in the channels: the patch embedding gets zero weights for the extra channels, the
    output head zero rows for the extra outputs (position-major, channel-minor, as the unpatchify reads them). Latent 0
    starts exactly as the pretrained model, the others are learned in (Wan's own I2V recipe). Returns the parameters
    of the two widened layers: zero-initialised, they need a higher lr than the backbone."""
    emb, head = transformer.patch_embedding, transformer.proj_out
    c, positions = emb.in_channels, head.out_features // emb.in_channels
    new_emb = nn.Conv3d(c * num_latents, emb.out_channels, emb.kernel_size, emb.stride).to(emb.weight)
    new_emb.weight.zero_()
    new_emb.weight[:, :c] = emb.weight
    new_emb.bias.copy_(emb.bias)

    # Wan's head is a Linear that diffusers reshapes back into patches, not a ConvTranspose3d
    new_head = nn.Linear(head.in_features, head.out_features * num_latents).to(head.weight)
    new_head.weight.zero_()
    new_head.bias.zero_()
    new_head.weight.view(positions, c * num_latents, -1)[:, :c] = head.weight.view(positions, c, -1)
    new_head.bias.view(positions, c * num_latents)[:, :c] = head.bias.view(positions, c)
    transformer.patch_embedding, transformer.proj_out = new_emb, new_head
    return list(new_emb.parameters()) + list(new_head.parameters())


class Modulation:
    """Forward hook on Wan's time embedding: adds `offset` [B, dim] (gamma through adaLN) when one is set; the
    embedding is [B, dim] or, with per-token timesteps, [B, L, dim]."""

    offset: torch.Tensor | None = None

    def __call__(self, module: nn.Module, inputs: tuple, out: torch.Tensor) -> torch.Tensor:
        if self.offset is None:
            return out
        return out + (self.offset[:, None] if out.ndim == 3 else self.offset).to(out)


def autocast(precision: str, device: torch.device) -> torch.autocast:
    """'mixed': bf16 matmuls over fp32 weights; 'fp32' and 'bf16' (weights already bf16) run as they are."""
    return torch.autocast(device.type, torch.bfloat16, enabled=precision == "mixed" and device.type == "cuda")


class WanDiT:
    """The pretrained transformer (or fine-tuned weights from a state file) with Wan's scheduler at time shift `shift`;
    `num_latents` latents per sample enter stacked in the channels (`widen_patchify`)."""

    def __init__(
        self,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        repo: str = REPO,
        load: str | Path | None = None,
        train: bool = False,
        checkpointing: bool = False,
        num_latents: int = 1,
        shift: float = 3.0,
    ):
        # local_files_only: the sharded loader would otherwise ask the Hub even in offline mode
        self.transformer = WanTransformer3DModel.from_pretrained(
            repo, subfolder="transformer", torch_dtype=dtype, local_files_only=True
        )
        self.patchify_params = widen_patchify(self.transformer, num_latents) if num_latents > 1 else []
        self.transformer.to(device).requires_grad_(train)
        if load:
            self.transformer.load_state_dict(torch.load(load, map_location=device))
        if checkpointing:
            self.transformer.enable_gradient_checkpointing()
        widened = {id(p) for p in self.patchify_params}
        self.trainable_params = [p for p in self.transformer.parameters() if p.requires_grad and id(p) not in widened]
        self.scheduler = UniPCMultistepScheduler.from_pretrained(
            repo, subfolder="scheduler", local_files_only=True, flow_shift=shift
        )
        self.modulation = Modulation()
        self.transformer.condition_embedder.time_embedder.register_forward_hook(self.modulation)
        self.device, self.dtype = torch.device(device), dtype

    def velocity(self, latents: torch.Tensor, timestep: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """The transformer's prediction (noise - x0) at `timestep` in 0..1000, one per sample [B] or per latent frame
        [B, F'] (spread over the frame's tokens). `cond`: tokens [B, L, 4096] for cross-attention, or a vector [B, dim]
        added to the time embedding (adaLN), with one zero token for cross-attention."""
        if timestep.ndim == 2:
            timestep = timestep.repeat_interleave(latents[0, 0, 0].numel() // 4, dim=1)
        self.modulation.offset = cond if cond.ndim == 2 else None
        tokens = cond if cond.ndim == 3 else cond.new_zeros(len(cond), 1, self.transformer.config.text_dim)
        out = self.transformer(latents.to(self.dtype), timestep, tokens.to(self.dtype), return_dict=False)[0]
        return out.float()

    @torch.no_grad()
    def sample(
        self,
        cond: torch.Tensor,
        uncond: torch.Tensor,
        shape: tuple[int, ...],
        steps: int = 50,
        guidance: float = 5.0,
        seed: int | list[int] = 0,
        init: torch.Tensor | None = None,
        strength: float = 1.0,
        context: int = 0,
        history_noise: float = 0,
        stochastic: float = 0,
    ) -> torch.Tensor:
        """Latents of `shape` from noise; with `init`, from that latent noised to `strength` (SDEdit: 1 = from
        noise, 0 = `init` untouched). `context`: `init`'s first latent frames stay fixed and only the rest is
        generated (the in-context conditioning of the fine-tune): frame 0 clean at timestep 0, the others re-noised
        to `history_noise` and marked with it. Classifier-free guidance between `cond` and `uncond`. `seed`: one for
        the batch or one per sample. `stochastic` > 0: Euler steps that re-noise the clean estimate to the next sigma,
        this share of the noise fresh (DDIM eta); 0 = UniPC."""
        seeds = seed if isinstance(seed, list) else [seed]
        per = (shape[0] // len(seeds), *shape[1:])
        gens = [torch.Generator(self.device).manual_seed(s) for s in seeds]
        noise = torch.cat([torch.randn(per, generator=g, device=self.device) for g in gens])
        self.scheduler.set_timesteps(steps, device=self.device)
        timesteps, start = self.scheduler.timesteps, 0
        latents = noise
        if init is not None:
            start = min(round(steps * (1 - strength)), steps)
            timesteps = timesteps[start:]
            self.scheduler.set_begin_index(start)
            latents = self.scheduler.add_noise(init, noise, timesteps[:1]) if len(timesteps) else init
        if context:  # what the model sees of the context: frame 0 clean, the history re-noised
            seen = torch.cat([init[:, :, :1], init[:, :, 1:context].lerp(noise[:, :, 1:context], history_noise)], 2)
        sigmas = self.scheduler.sigmas[start:].tolist()
        for i, t in enumerate(timesteps):
            if context:
                latents[:, :, :context] = seen
            t_frames = t.expand(shape[0], shape[2]).float()
            t_frames[:, :context], t_frames[:, 1:context] = 0, history_noise * 1000
            v = self.velocity(latents, t_frames, cond)
            if guidance != 1:
                v_uncond = self.velocity(latents, t_frames, uncond)
                v = v_uncond + guidance * (v - v_uncond)
            if stochastic:  # x = (1 - s) x0 + s noise and v = noise - x0 give both estimates
                sigma, nxt = sigmas[i : i + 2]
                fresh = torch.cat([torch.randn(per, generator=g, device=self.device) for g in gens])
                x0, eps = latents - sigma * v, latents + (1 - sigma) * v
                latents = (1 - nxt) * x0 + nxt * ((1 - stochastic**2) ** 0.5 * eps + stochastic * fresh)
            else:
                latents = self.scheduler.step(v, t, latents, return_dict=False)[0]
        if context:
            latents[:, :, :context] = init[:, :, :context]
        return latents
