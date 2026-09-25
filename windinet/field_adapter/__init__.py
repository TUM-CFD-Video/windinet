"""Deterministic per-pixel maps between physical fields and 3-channel RGB.

Model-agnostic: nothing here imports diffusers or knows which video VAE
consumes the RGB. See `bijections.py` for the adapter, `grouping.py` for the
fields-to-images assignment and `latent_stats.py` for the Fréchet diagnostic.
"""

from .bijections import FieldAdapter, Mix, PreNorm, SoftClip
from .grouping import GroupedAdapter

__all__ = ["FieldAdapter", "GroupedAdapter", "Mix", "PreNorm", "SoftClip"]
