"""scripts/wan/field_stats.py against numpy on tiny synthetic files."""

import importlib.util
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from windinet.data.frames import load_field_stats
from windinet.data.specs import SPECS

spec = importlib.util.spec_from_file_location("field_stats", Path(__file__).parents[1] / "scripts/wan/field_stats.py")
# registered so the pool can pickle the script's functions by module
field_stats = sys.modules["field_stats"] = importlib.util.module_from_spec(spec)
spec.loader.exec_module(field_stats)


SIMS = 4


def run(dataset: str, h5: Path, out: Path) -> dict:
    field_stats.main(dataset, h5, out, workers=2, sims=SIMS)
    return json.loads(out.read_text())


def pooled(h5: Path, name: str) -> np.ndarray:
    with h5py.File(h5, "r") as f:
        x = np.concatenate([f[sid][name.removeprefix("log_")][:].ravel() for sid in sorted(f)[:SIMS]])
    return np.log(x) if name.startswith("log_") else x


def test_srba(srba_files, tmp_path):
    h5 = srba_files[0]
    out = run("srba_oooo", h5, tmp_path / "stats.json")
    stats = out["data_normalization_stats"]
    assert out["num_simulations"] == SIMS
    assert "log_pressure" in stats and "log_log_density" not in stats and "log_velocityX" not in stats
    assert set(SPECS["srba_oooo"].fields) <= set(stats)
    for name, s in stats.items():
        x = pooled(h5, name)
        for key, ref in (("mean", x.mean()), ("std", x.std()), ("min", x.min()), ("max", x.max())):
            assert s[key] == pytest.approx(ref, rel=1e-9, abs=1e-12), (name, key)
        bin_width = (x.max() - x.min()) / field_stats.BINS
        assert abs(s["p50"] - np.percentile(x, 50, method="inverted_cdf")) <= bin_width, name
    assert load_field_stats(tmp_path / "stats.json") == stats


def test_eulermq(eulermq_file, tmp_path):
    stats = run("eulermq", eulermq_file, tmp_path / "stats.json")["data_normalization_stats"]
    assert list(stats) == [*SPECS["eulermq"].fields, "log_density", "log_pressure"]
