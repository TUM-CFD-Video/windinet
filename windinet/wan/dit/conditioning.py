"""What conditions the DiT on gamma: tokens [B, L, 4096] it cross-attends to, or a vector [B, dim] added to the time
embedding behind Wan's adaLN (WanDiT.velocity tells the two apart by shape).

A conditioning is a module with forward(gamma) -> condition and uncond(gamma) -> the empty condition (classifier-free
guidance). To add one: a class here and a branch in `build`; `cond.kind` in the YAML selects it. The clean first frame
is not a conditioning here: it enters through per-token timesteps (WanDiT.velocity).
"""

from __future__ import annotations

import torch
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from torch import nn

from windinet.config import ScalarConditioningConfig
from windinet.scalar_embeddings import ScalarEmbedding
from windinet.wan.dit.config import CondConfig
from windinet.wan.dit.model import encode_text


class TextTokens(nn.Module):
    """One fixed prompt for every sample: the UMT5 embeddings of `prompt` and of `negative_prompt`, computed once.
    The zero-shot and text baselines; nothing to train."""

    def __init__(self, prompt: str, negative_prompt: str, device: torch.device):
        super().__init__()
        self.register_buffer("tokens", encode_text([prompt, negative_prompt], device))  # [2, 512, 4096]

    def forward(self, gamma: torch.Tensor) -> torch.Tensor:
        return self.tokens[:1].expand(len(gamma), -1, -1)

    def uncond(self, gamma: torch.Tensor) -> torch.Tensor:
        return self.tokens[1:].expand(len(gamma), -1, -1)


class GammaTokens(nn.Module):
    """Li's conditioning: gamma scaled to [0, 1] -> 64 Fourier features -> MLP -> `tokens` tokens of width 4096 that
    replace the text; trained with the DiT. Weihao's ScalarEmbedding (windinet/scalar_embeddings.py) for now."""

    def __init__(self, tokens: int, gamma_range: tuple[float, float]):
        super().__init__()
        spec = ScalarConditioningConfig(
            enabled=True, scalar_names=["gamma"], scalar_ranges={"gamma": gamma_range}, num_tokens_per_scalar=tokens
        )
        self.embed = ScalarEmbedding(spec)

    def forward(self, gamma: torch.Tensor) -> torch.Tensor:
        return self.embed(gamma.float()[:, None])

    def uncond(self, gamma: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(self(gamma))  # Li: zeros as the unconditional tokens


class GammaModulation(nn.Module):
    """Gamma into Wan's adaLN path, embedded like the timestep: scaled to 0..1000, Wan's sinusoidal features, a fresh
    copy of its time MLP whose last layer starts at zero (the pretrained model at step 0). `embedder` is the
    transformer's condition_embedder (feature width and embedding width are read from it)."""

    def __init__(self, gamma_range: tuple[float, float], embedder: nn.Module):
        super().__init__()
        self.lo, self.hi = gamma_range
        self.features = Timesteps(embedder.timesteps_proj.num_channels, flip_sin_to_cos=True, downscale_freq_shift=0)
        time_mlp = embedder.time_embedder
        self.mlp = TimestepEmbedding(time_mlp.linear_1.in_features, time_mlp.linear_2.out_features)
        nn.init.zeros_(self.mlp.linear_2.weight)
        nn.init.zeros_(self.mlp.linear_2.bias)

    def forward(self, gamma: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.features((gamma.float() - self.lo) / (self.hi - self.lo) * 1000))

    def uncond(self, gamma: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(self(gamma))


def build(cfg: CondConfig, device: torch.device, embedder: nn.Module | None = None) -> nn.Module:
    """The conditioning of `cfg.kind`; gamma_adaln needs the transformer's condition_embedder."""
    if cfg.kind == "text":
        return TextTokens(cfg.prompt, cfg.negative_prompt, device).to(device)
    if cfg.kind == "gamma_adaln":
        return GammaModulation(cfg.gamma_range, embedder).to(device)
    return GammaTokens(cfg.tokens, cfg.gamma_range).to(device)
