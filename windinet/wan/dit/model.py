"""Wan 2.1 text-to-video transformer: loading, text embeddings, the velocity forward, and the sampling loop.

The transformer sees normalised latents [B, 16, F', H/8, W/8] (what WanVAE.encode returns) and a conditioning
tensor [B, L, 4096] through cross-attention (windinet.wan.dit.conditioning builds it). Timesteps can be given per token,
which is how a clean conditioning frame is marked (timestep 0).
"""

from __future__ import annotations

from pathlib import Path

import torch
from diffusers import UniPCMultistepScheduler, WanTransformer3DModel
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


def autocast(precision: str, device: torch.device) -> torch.autocast:
    """'mixed': bf16 matmuls over fp32 weights; 'fp32' and 'bf16' (weights already bf16) run as they are."""
    return torch.autocast(device.type, torch.bfloat16, enabled=precision == "mixed" and device.type == "cuda")


class WanDiT:
    """The pretrained transformer (or a fine-tuned one from a run folder) with Wan's scheduler."""

    def __init__(
        self,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        repo: str = REPO,
        load: str | None = None,
        train: bool = False,
        checkpointing: bool = False,
    ):
        # local_files_only: the sharded loader would otherwise ask the Hub even in offline mode
        self.transformer = WanTransformer3DModel.from_pretrained(
            repo, subfolder="transformer", torch_dtype=dtype, local_files_only=True
        )
        self.transformer.to(device).requires_grad_(train)
        if load:
            self.transformer.load_state_dict(torch.load(Path(load) / "dit.pt", map_location=device))
        if checkpointing:
            self.transformer.enable_gradient_checkpointing()
        self.trainable_params = [p for p in self.transformer.parameters() if p.requires_grad]
        self.scheduler = UniPCMultistepScheduler.from_pretrained(repo, subfolder="scheduler", local_files_only=True)
        self.device, self.dtype = torch.device(device), dtype

    def velocity(
        self, latents: torch.Tensor, timestep: torch.Tensor, cond: torch.Tensor, clean: int = 0
    ) -> torch.Tensor:
        """The transformer's prediction (noise - x0) at `timestep` [B] in 0..1000; the first `clean` tokens (the
        frame-major 2x2 patches of latent frame 0) get timestep 0."""
        if clean:
            timestep = timestep[:, None].expand(-1, latents[0, 0].numel() // 4).clone()
            timestep[:, :clean] = 0
        out = self.transformer(latents.to(self.dtype), timestep, cond.to(self.dtype), return_dict=False)[0]
        return out.float()

    @torch.no_grad()
    def sample(
        self,
        cond: torch.Tensor,
        uncond: torch.Tensor,
        shape: tuple[int, ...],
        steps: int = 50,
        guidance: float = 5.0,
        seed: int = 0,
        init: torch.Tensor | None = None,
        strength: float = 1.0,
        keep_first_frame: bool = False,
    ) -> torch.Tensor:
        """Latents of `shape` from noise; with `init`, from that latent noised to `strength` (SDEdit: 1 = from
        noise, 0 = `init` untouched). `keep_first_frame`: `init`'s first latent frame stays clean at timestep 0
        and only the rest is generated (the in-context conditioning of the fine-tune). Classifier-free guidance
        between `cond` and `uncond`."""
        generator = torch.Generator(self.device).manual_seed(seed)
        noise = torch.randn(shape, generator=generator, device=self.device)
        self.scheduler.set_timesteps(steps, device=self.device)
        timesteps = self.scheduler.timesteps
        latents = noise
        if init is not None:
            start = min(round(steps * (1 - strength)), steps)
            timesteps = timesteps[start:]
            self.scheduler.set_begin_index(start)
            latents = self.scheduler.add_noise(init, noise, timesteps[:1]) if len(timesteps) else init
        clean = shape[-2] * shape[-1] // 4 if keep_first_frame else 0
        for t in timesteps:
            if clean:
                latents[:, :, :1] = init[:, :, :1]
            v = self.velocity(latents, t.expand(shape[0]), cond, clean)
            if guidance != 1:
                v_uncond = self.velocity(latents, t.expand(shape[0]), uncond, clean)
                v = v_uncond + guidance * (v - v_uncond)
            latents = self.scheduler.step(v, t, latents, return_dict=False)[0]
        if clean:
            latents[:, :, :1] = init[:, :, :1]
        return latents
