#!/usr/bin/env python
"""Regenerate the README.md index of every results folder that holds runs (train_vae.py does this after each run).

python scripts/wan/index_runs.py [results/wan/vae results/field_adapter/20_wan_roundtrip ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

from windinet.experiment.runs import write_index

if __name__ == "__main__":
    found = {p.parent.parent for p in Path("results").rglob("metrics.json")}  # a run with stages is not a stage dir
    stages = [Path(p) for p in sys.argv[1:]] or sorted(d for d in found if not (d / "config.yaml").exists())
    for stage in stages:
        print("wrote", write_index(stage))
