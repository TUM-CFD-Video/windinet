#!/usr/bin/env python
"""Regenerate the README.md index of every results folder that holds runs (train_vae.py does this after each run).

python scripts/wan/index_runs.py [results/wan/vae results/field_adapter/20_wan_roundtrip ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

from windinet.experiment.runs import write_index

if __name__ == "__main__":
    stages = [Path(p) for p in sys.argv[1:]] or sorted({p.parent.parent for p in Path("results").rglob("metrics.json")})
    for stage in stages:
        print("wrote", write_index(stage))
