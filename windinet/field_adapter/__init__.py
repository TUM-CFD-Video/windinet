"""Deterministic per-pixel maps between physical fields and 3-channel RGB; nothing here knows which video VAE
consumes the RGB. `bijections.py` is the adapter, `grouping.py` the fields-to-images assignment."""

from .bijections import FieldAdapter
from .grouping import GroupedAdapter, group_tag

__all__ = ["FieldAdapter", "GroupedAdapter", "group_tag"]
