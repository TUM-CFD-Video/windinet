"""What the DiT cross-attends to: a token tensor [B, L, 4096] per sample, built from gamma.

A conditioning is a module with forward(gamma) -> tokens and uncond(gamma) -> the tokens of the empty condition
(classifier-free guidance). To add one: a class here and a branch in `build`; `cond.kind` in the YAML selects it.
The clean first frame is not a token conditioning: it enters through per-token timesteps (WanDiT.velocity).
"""

from __future__ import annotations

import torch
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


def build(cfg: CondConfig, device: torch.device) -> nn.Module:
    if cfg.kind == "text":
        return TextTokens(cfg.prompt, cfg.negative_prompt, device).to(device)
    return GammaTokens(cfg.tokens, cfg.gamma_range).to(device)
