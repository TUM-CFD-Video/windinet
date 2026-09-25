"""Deterministic per-pixel maps between physical fields and 3-channel RGB.

Model-agnostic: nothing here imports diffusers or knows which video VAE
consumes the RGB. See `bijections.py` for the adapter and `latent_stats.py`
for the distribution-matching terms (usable on RGB pixels or VAE latents).
"""

from .bijections import FieldAdapter, Mix, MonotoneWarp, PreNorm, SoftClip

__all__ = ["FieldAdapter", "Mix", "MonotoneWarp", "PreNorm", "SoftClip"]
