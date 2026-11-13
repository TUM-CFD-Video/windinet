import json
import math

import pytest
from torch.utils.data import DataLoader

from windinet.data.frames import FrameSampler, load_field_stats
from windinet.data.specs import EULERMQ, SRBA_OOOO
from windinet.field_adapter import GroupedAdapter
from windinet.wan.dit.benchmark import Benchmark
from windinet.wan.figures import _shown, save_examples, save_panel
from windinet.wan.vae.config import DataConfig
from windinet.wan.vae.evaluate import evaluate

POSITIVE = {"mean": 1.6, "std": 2.0, "min": 0.01, "max": 30.0, "p0_1": 0.02, "p99_9": 20.0}
SIGNED = {"mean": 0.1, "std": 1.0, "min": -4.0, "max": 4.0, "p0_1": -3.0, "p99_9": 2.5}
SRBA_STATS = {"pressure": POSITIVE, "velocityX": SIGNED, "velocityY": SIGNED, "log_density": SIGNED | {"min": -3.5}}


def _data(tmp_path, dataset, stats, h5="unused.h5"):
    path = tmp_path / "stats.json"
    path.write_text(json.dumps({"data_normalization_stats": stats}))
    return DataConfig(dataset=dataset, h5_path=str(h5), stats_json=str(path))


def test_srba_display_bounds(tmp_path):
    shown = dict(zip(SRBA_OOOO.fields, _shown(_data(tmp_path, "srba_oooo", SRBA_STATS), SRBA_OOOO.fields), strict=True))
    for v in ("velocityX", "velocityY"):
        assert (shown[v]["lo"], shown[v]["hi"], shown[v]["signed"]) == (-3.0, 3.0, True)
    p = shown["pressure"]
    assert (p["lo"], p["hi"], p["centre"], p["width"], p["signed"]) == (0.01, 20.0, 0.01, 0.5, False)
    assert shown["log_density"]["centre"] == shown["log_density"]["lo"] == -3.5
    assert (shown["log_density"]["width"], shown["velocityY"]["width"], shown["velocityX"]["width"]) == (100, 1, 100)


def test_eulermq_display_unchanged(tmp_path):
    stats = {f: SIGNED for f in EULERMQ.fields}
    data = _data(tmp_path, "eulermq", stats)
    today = [{**stats[n], "signed": n.startswith("momentum")} for n in EULERMQ.fields]
    assert _shown(data, list(EULERMQ.fields)) == today


@pytest.fixture
def srba_eval(make_h5, tmp_path):
    """The adapter-only round trip of srba sims at three Mach numbers, whole trajectories; 16×16, since SSIM's window
    is 11 pixels."""
    ids = [
        f"{i:03d}_Mas{m:.2f}_sb{b}_Ax0.0129_Ay0.0081_Ar0.0035_BxNaN_ByNaN_BrNaN"
        for i, (m, b) in enumerate((m, b) for m in (2.1, 1.3, 1.7) for b in (1, 2))
    ]
    test = make_h5(tmp_path / "test.h5", SRBA_OOOO, ids, size=16)
    data = _data(tmp_path, "srba_oooo", SRBA_STATS, test)
    adapter = GroupedAdapter([list(g) for g in SRBA_OOOO.groups], load_field_stats(data.stats_json), SRBA_OOOO)
    sims = FrameSampler(test, ids, None, 0, spec=SRBA_OOOO)
    adapter.init_from_data(adapter.select(sims[0][0]))
    bench = Benchmark(data.stats_json, adapter.fields, 1, "cpu", prefix="bench_vae", frames=(4, 8))
    return data, adapter, *evaluate(adapter, DataLoader(sims, 1), [], None, spec=SRBA_OOOO, bench=bench)


def test_evaluate_examples_at_lowest_and_highest_mach(srba_eval):
    _, adapter, metrics, _, examples = srba_eval
    assert set(examples) == {1.3, 2.1}
    assert metrics["bench_vae_frames"] == [4, 8]
    assert all(math.isfinite(metrics[k]) for k in ("bench_vae_vrmse", "bench_vae_ssim", "vrmse_mean"))
    assert examples[1.3][1].shape == (9, len(adapter.fields), 16, 16)


def test_evaluate_breaks_the_vrmse_down_per_scalar(srba_eval):
    metrics = srba_eval[2]
    for name, counts in (("mach", {1.3: 2, 1.7: 2, 2.1: 2}), ("bubbles", {1: 3, 2: 3})):
        means = {v: metrics[f"vrmse_mean_{name}{v:g}"] for v in counts}
        assert math.isclose(sum(counts[v] * m for v, m in means.items()) / 6, metrics["vrmse_mean"], rel_tol=1e-5)


def test_panel_and_videos_written(srba_eval, tmp_path):
    data, adapter, _, _, examples = srba_eval
    short = {m: (sid, x[:3], x_hat[:3]) for m, (sid, x, x_hat) in examples.items()}
    save_panel(short, adapter.fields, data, tmp_path / "panel.png")
    save_examples({2.1: short[2.1]}, adapter.fields, data, tmp_path, None, "recon")
    assert {"panel.png", "recon_mach2.1.mp4", "example_mach2.1.pt"} <= {p.name for p in tmp_path.iterdir()}
