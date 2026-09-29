#!/usr/bin/env python
"""Regenerate the README.md index of every results stage (train_adapter.py does this after each run).

    python scripts/field_adapter/index_runs.py [results/field_adapter/20_wan_roundtrip ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

from windinet.field_adapter.runs import write_index

if __name__ == "__main__":
    stages = [Path(p) for p in sys.argv[1:]] or [d for d in Path("results/field_adapter").iterdir() if any(d.glob("*/metrics.json"))]
    for stage in stages:
        print("wrote", write_index(stage))
