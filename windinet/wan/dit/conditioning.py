"""What conditions the DiT on the dataset's scalars: tokens [B, L, 4096] it cross-attends to, or a vector [B, dim]
added to the time embedding behind Wan's adaLN (WanDiT.velocity tells the two apart by shape).

A conditioning is a module with forward(condition [B, S]) -> condition and uncond(condition) -> the empty condition
(classifier-free guidance). To add one: a class here and a branch in `build`; `cond.kind` in the YAML selects it. The
clean first frame is not a conditioning here: it enters through per-token timesteps (WanDiT.velocity).
"""

from __future__ import annotations

import torch
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from torch import nn

from windinet.config import ScalarConditioningConfig
from windinet.data.specs import Spec
from windinet.scalar_embeddings import ScalarEmbedding
from windinet.wan.dit.config import CondConfig
from windinet.wan.dit.model import encode_text


class TextTokens(nn.Module):
    """One fixed prompt for every sample: the UMT5 embeddings of `prompt` and of `negative_prompt`, computed once.
    The zero-shot and text baselines; nothing to train."""

    def __init__(self, prompt: str, negative_prompt: str, device: torch.device):
        super().__init__()
        self.register_buffer("tokens", encode_text([prompt, negative_prompt], device))  # [2, 512, 4096]

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        return self.tokens[:1].expand(len(condition), -1, -1)

    def uncond(self, condition: torch.Tensor) -> torch.Tensor:
        return self.tokens[1:].expand(len(condition), -1, -1)


class ScalarTokens(nn.Module):
    """Li's conditioning, one embedding per input: scaled to [0, 1] over its range -> 64 Fourier features -> its own
    MLP -> `tokens` tokens of width 4096; concatenated, they replace the text. Wan's cross-attention has no key
    positions, so a shared MLP could not tell the inputs apart. The first is `embed`, so one input (eulermq's gamma)
    keeps the module and state dict of the former GammaTokens."""

    def __init__(self, tokens: int, ranges: dict[str, tuple[float, float]]):
        super().__init__()

        def embedding(name: str) -> ScalarEmbedding:
            config = ScalarConditioningConfig(
                enabled=True, scalar_names=[name], scalar_ranges={name: ranges[name]}, num_tokens_per_scalar=tokens
            )
            return ScalarEmbedding(config)

        self.embed, *rest = (embedding(n) for n in ranges)
        self.rest = nn.ModuleList(rest)

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        c = condition.float().view(len(condition), -1)
        return torch.cat([e(c[:, i : i + 1]) for i, e in enumerate((self.embed, *self.rest))], 1)

    def uncond(self, condition: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(self(condition))  # Li: zeros as the unconditional tokens


class GammaModulation(nn.Module):
    """Gamma into Wan's adaLN path, embedded like the timestep: scaled to 0..1000, Wan's sinusoidal features, a fresh
    copy of its time MLP whose last layer starts at zero (the pretrained model at step 0); input [B, 1]. `embedder` is
    the transformer's condition_embedder (feature width and embedding width are read from it)."""

    def __init__(self, gamma_range: tuple[float, float], embedder: nn.Module):
        super().__init__()
        self.lo, self.hi = gamma_range
        self.features = Timesteps(embedder.timesteps_proj.num_channels, flip_sin_to_cos=True, downscale_freq_shift=0)
        time_mlp = embedder.time_embedder
        self.mlp = TimestepEmbedding(time_mlp.linear_1.in_features, time_mlp.linear_2.out_features)
        nn.init.zeros_(self.mlp.linear_2.weight)
        nn.init.zeros_(self.mlp.linear_2.bias)

    def forward(self, gamma: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.features((gamma.float()[:, 0] - self.lo) / (self.hi - self.lo) * 1000))

    def uncond(self, gamma: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(self(gamma))


def build(cfg: CondConfig, spec: Spec, device: torch.device, embedder: nn.Module | None = None) -> nn.Module:
    """The conditioning of `cfg.kind` on the dataset's scalars and direction; gamma_adaln (one scalar) needs the
    transformer's condition_embedder."""
    ranges = spec.ranges | ({"dir_h": (-1.0, 1.0), "dir_w": (-1.0, 1.0)} if spec.direction else {})
    if cfg.kind == "text":
        return TextTokens(cfg.prompt, cfg.negative_prompt, device).to(device)
    if cfg.kind == "gamma_adaln":
        if len(ranges) != 1:
            raise ValueError(f"gamma_adaln needs one scalar, {spec.name} has {list(ranges)}")
        (single,) = ranges.values()
        return GammaModulation(single, embedder).to(device)
    return ScalarTokens(cfg.tokens, ranges).to(device)
