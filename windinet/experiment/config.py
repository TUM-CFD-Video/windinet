"""A YAML config plus `--set a.b=value` overrides: how every stage script takes its settings."""

from __future__ import annotations

from pathlib import Path

import yaml


def load_config(path: Path, overrides: list[str]) -> dict:
    """The YAML as a dict with each dotted override applied; values are parsed as YAML (3e-3, true, [a, b])."""
    raw = yaml.safe_load(path.read_text())
    for item in overrides:
        key, _, value = item.partition("=")
        *parents, leaf = key.split(".")
        node = raw
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(value)
    return raw
