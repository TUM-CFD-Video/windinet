import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch
import yaml
from torch.utils.data import DataLoader

from windinet.data.frames import SYMMETRIES, FrameSampler, load_field_stats, symmetry, val_split
from windinet.data.specs import EULERMQ, SRBA_OOOO
from windinet.experiment.config import load_config
from windinet.experiment.log import log_metrics
from windinet.experiment.metrics import Aggregate, vrmse
from windinet.experiment.runs import run_dataset, write_index
from windinet.field_adapter import FieldAdapter, GroupedAdapter, load_adapter
from windinet.field_adapter.bijections import PreNorm
from windinet.field_adapter.grouping import shared_vector_stats
from windinet.losses import vrms_per_channel
from windinet.wan.corrector.config import CorrectorStageConfig
from windinet.wan.dit.config import DitStageConfig
from windinet.wan.vae.config import VaeStageConfig
from windinet.wan.vae.evaluate import evaluate
from windinet.wan.weights import load_weights

FIELDS = ["density", "momentum_x", "momentum_y", "pressure", "log_density", "log_pressure"]
STATS = {k: {"mean": 0.0, "std": 1.0, "min": -3.0, "max": 3.0} for k in FIELDS}
SRBA_STATS = {k: {"mean": 0.1, "std": 2.0, "min": -9.0, "max": 9.0} for k in SRBA_OOOO.fields}
SRBA_STATS["log_pressure"] = {"mean": 0.1, "std": 2.0, "min": -5.0, "max": 5.0}  # own bounds: log_fields matter
NAMED = ["name=t", "description=t"]


def _adapter(n):
    return FieldAdapter(mean=[0.0] * n, std=[1.0] * n, log_channels=([True, False, False, True])[:n])


def test_two_and_three_field_round_trips_are_exact():
    for n in (2, 3):
        a = _adapter(n)
        x = torch.rand(2, n, 8, 8) + 0.1  # positive, so the log channels are fine
        a.init_from_data(x)
        with torch.no_grad():
            a.mix.weight.add_(torch.randn_like(a.mix.weight) * 0.3)
        assert torch.allclose(a.inverse(a(x)), x, atol=1e-4, rtol=1e-4)
        assert a(x).abs().max() < 1.0


def test_four_to_three_round_trip_is_pseudo_inverse():
    a = _adapter(4)
    x = torch.rand(2, 4, 8, 8) + 0.1
    z = a.mix(a.prenorm(x))
    assert torch.allclose(a.mix(a.mix.inverse(z)), z, atol=1e-5)  # W W^+ W = W: exact on the row space
    assert a.inverse(a(x)).shape == x.shape


def test_grouped_adapter_drops_fields_in_no_group():
    a = GroupedAdapter([["density", "momentum_x", "momentum_y"]], STATS, EULERMQ)
    x = a.select(torch.rand(2, 4, 8, 8) + 0.1)
    a.init_from_data(x)
    assert a.fields == ["density", "momentum_x", "momentum_y"] and x.shape[1] == 3
    assert torch.allclose(a.inverse(a(x)), x, atol=1e-4)


def test_widened_vae_starts_as_the_pretrained_one():
    from diffusers import AutoencoderKLWan

    from windinet.wan.vae.model import widen_io

    vae = AutoencoderKLWan(base_dim=8, num_res_blocks=1, dim_mult=[1, 1, 1, 1], z_dim=4).eval()
    x = torch.randn(1, 4, 5, 32, 32)
    with torch.no_grad():
        z3 = vae.encode(x[:, :3]).latent_dist.mode()
        widen_io(vae, 4)
        z = vae.encode(x).latent_dist.mode()
        y = vae.decode(z).sample
    assert torch.allclose(z, z3) and y.shape[1] == 4 and y[:, 3].abs().max() == 0
    a = GroupedAdapter([FIELDS[:4]], STATS, EULERMQ)
    assert a.channels == 4 and a(torch.rand(2, 4, 8, 8) + 0.1)[0].shape[1] == 4


def test_vrmse_equals_the_ltx_baselines_metric():
    x, x_hat = torch.rand(7, 4, 8, 8), torch.rand(7, 4, 8, 8)
    his = vrms_per_channel(x_hat.movedim(0, 1)[None], x.movedim(0, 1)[None])  # his layout: [B, C, T, H, W]
    assert torch.allclose(vrmse(x_hat, x), his, atol=1e-6)


def _uniform_start(tmp_path):
    """An srba-like sim of 9 frames, exactly uniform up to frame 4, written as the dataset stores it; its
    FrameSampler (whole trajectories) and a prediction with 1e-3 noise."""
    gen = torch.Generator().manual_seed(0)
    x = torch.ones(9, 4, 16, 16, dtype=torch.float64)
    x[4:] += torch.randn(5, 4, 16, 16, generator=gen, dtype=torch.float64)
    sid = "000_Mas1.30_sb1_Ax0.0129_Ay0.0081_Ar0.0035_BxNaN_ByNaN_BrNaN"
    with h5py.File(tmp_path / "s.h5", "w") as f:
        for i, n in enumerate(SRBA_OOOO.fields):
            v = x[:, i : i + 1].numpy()
            f.create_dataset(f"{sid}/{n.removeprefix('log_')}", data=np.exp(v) if n.startswith("log_") else v)
    sims = FrameSampler(tmp_path / "s.h5", [sid], None, 0, spec=SRBA_OOOO)
    x = sims[0][0]
    return sims, x, x + 1e-3 * torch.randn(x.shape, generator=gen)


def test_uniform_start_scored_over_the_trajectory_variance(tmp_path):
    sims, x, x_hat = _uniform_start(tmp_path)
    var = sims.variance(0).float()
    assert vrmse(x_hat[:2], x[:2]).min() > 1  # the sample's own variance is 0: the error explodes
    assert vrmse(x_hat[:2], x[:2], var).max() < 1e-2
    torch.testing.assert_close(vrmse(x_hat, x, var), vrmse(x_hat, x), rtol=1e-6, atol=0)  # whole trajectory: equal
    curves = [Aggregate(SRBA_OOOO.fields, "cpu") for _ in range(2)]
    curves[0].add(x_hat, x)
    curves[1].add(x_hat, x, var)
    assert max(curves[0].metrics()["vrmse_per_frame"]) > 1 and max(curves[1].metrics()["vrmse_per_frame"]) < 1e-2
    adapter = _srba_adapter()
    val, *_ = evaluate(adapter, DataLoader(sims, 1), [], 1, spec=SRBA_OOOO, variances={sims.ids[0]: var})
    assert val and all(k.startswith("vrmse") for k in val) and val["vrmse_mean"] < 1e-2


def test_eulermq_scores_unchanged(make_h5, tmp_path):
    """No trajectory variance on eulermq: the numbers of HEAD 6707466's definitions, bit for bit."""
    stats = load_field_stats("results/field_adapter/00_stats/eulermq_256_train_stats.json")
    adapter = GroupedAdapter([list(g) for g in EULERMQ.groups], stats, EULERMQ)
    ids = [f"{i:04d}_gamma{g}" for i, g in enumerate((1.13, 1.4, 1.76, 1.4))]
    sims = FrameSampler(make_h5(tmp_path / "e.h5", EULERMQ, ids, size=16), ids, None, 0, spec=EULERMQ)
    metrics, *_ = evaluate(adapter, DataLoader(sims, 1), [], None, spec=EULERMQ)
    x = adapter.select(sims[0][0])
    x_hat = x + 0.1 * torch.randn(x.shape, generator=torch.Generator().manual_seed(0))
    one = Aggregate(adapter.fields, "cpu")
    one.add(x_hat, x)
    m, err = one.metrics(), (x_hat - x).square()
    head = (err.mean((0, 2, 3)) / (x.var((0, 2, 3), unbiased=False) + 1e-8)).sqrt()
    assert [m[f"vrmse_{f}"] for f in adapter.fields] == head.tolist()
    head_curve = (err.mean((2, 3)) / (x.var((2, 3), unbiased=False) + 1e-8)).sqrt().mean(1)
    assert m["vrmse_per_frame"] == head_curve.tolist()
    assert {f"vrmse_mean_gamma{g}" for g in (1.13, 1.4, 1.76)} <= set(metrics) and "h1_mean" in metrics


def test_val_split_is_the_ltx_baselines_rule():
    ids = [f"{i:04d}_gamma1.3" for i in range(20)]
    train, val = val_split(ids, n_val=5, seed=42)
    perm = torch.randperm(20, generator=torch.Generator().manual_seed(42)).tolist()
    assert val == [ids[i] for i in perm[15:]] and len(train) == 15 and not set(train) & set(val)


def test_old_run_configs_load_unchanged():
    vae = VaeStageConfig(**load_config(Path("results/wan/eulermq/vae/res256/pairs/adapter/config.yaml"), []))
    assert vae.data.dataset == "eulermq" and vae.results_dir == "results/wan/eulermq/vae"
    assert vae.adapter.groups == [["density", "pressure"], ["momentum_x", "momentum_y"]]
    dit = DitStageConfig(**load_config(Path("results/wan/eulermq/dit/res256/pairs/unrolled/unrolled2/config.yaml"), []))
    assert dit.results_dir == "results/wan/eulermq/dit"
    corrector = CorrectorStageConfig(**load_config(Path("results/wan/eulermq/dit/corrector/ar/config.yaml"), []))
    assert corrector.results_dir == "results/wan/eulermq/dit/corrector"


def test_stage_defaults_follow_the_dataset():
    srba = VaeStageConfig(**load_config(Path("configs/wan/srba_oooo/vae.yaml"), NAMED))
    assert (
        srba.adapter.groups == [list(g) for g in SRBA_OOOO.groups] and srba.results_dir == "results/wan/srba_oooo/vae"
    )
    assert VaeStageConfig(**load_config(Path("configs/wan/vae.yaml"), NAMED)).results_dir == "results/wan/eulermq/vae"
    for schema, path, stage in ((DitStageConfig, "dit", "dit"), (CorrectorStageConfig, "corrector", "dit/corrector")):
        for dataset in ("eulermq", "srba_oooo"):
            cfg = schema(**load_config(Path(f"configs/wan/{path}.yaml"), [*NAMED, f"data.dataset={dataset}"]))
            assert cfg.results_dir == f"results/wan/{dataset}/{stage}"


def test_adapter_fields_must_be_in_the_dataset():
    srba = {"name": "x", "description": "x", "data": {"dataset": "srba_oooo"}}
    for adapter in ({"groups": [["density", "pressure"]]}, {"log_fields": ["log_density"]}):
        with pytest.raises(ValueError, match="srba_oooo"):
            VaeStageConfig(**srba, adapter=adapter)


def _srba_adapter(log_fields=None, clip_quantile=0.98):
    groups = [list(g) for g in SRBA_OOOO.groups]
    return GroupedAdapter(groups, SRBA_STATS, SRBA_OOOO, log_fields, clip_quantile)


def test_srba_adapter_round_trip_and_loss_space():
    x = torch.randn(2, 4, 8, 8) / 2
    x[:, 0] = x[:, 0].exp()  # pressure > 0 for its log arm, inside the stats' range
    for log_fields, logged in (([], [False, False]), (["pressure"], [False, True])):  # group A: (ln ρ, p)
        a = _srba_adapter(log_fields)
        xs = a.select(x)
        a.init_from_data(xs)
        assert a.fields == ["pressure", "velocityX", "velocityY", "log_density"]
        assert torch.allclose(a.inverse(a(xs)), xs, atol=1e-4)
        assert not a.base_norm.log_mask.any()
        assert [bool(v) for v in a.adapters[0].prenorm.log_mask] == logged
    assert _srba_adapter(clip_quantile=0.999).adapters[1].clip.q == 0.999


SKEWED = SRBA_STATS | {  # srba_oooo-like: u large and one-signed, v small and centred
    "velocityX": {"mean": 147.0, "std": 184.0, "min": -143.0, "max": 1029.0},
    "velocityY": {"mean": 0.0, "std": 23.5, "min": -625.0, "max": 577.0},
}


def _skewed(shared: bool) -> GroupedAdapter:
    return GroupedAdapter([list(g) for g in SRBA_OOOO.groups], SKEWED, SRBA_OOOO, shared_vector_scale=shared)


def test_shared_vector_scale_makes_the_normalisation_d4_equivariant():
    std = ((147.0**2 + 184.0**2 + 23.5**2) / 2) ** 0.5
    assert shared_vector_stats(SKEWED, SRBA_OOOO.vector)["velocityY"] == {
        "mean": 0.0, "std": std, "min": -1029.0, "max": 1029.0
    }  # fmt: skip
    on, off = _skewed(True), _skewed(False)
    uv = [on.fields.index(n) for n in SRBA_OOOO.vector]
    assert on.fields == list(SRBA_OOOO.fields)  # so symmetry acts on the grouped fields as on the dataset's
    for pre, idx in ((on.base_norm, uv), (on.adapters[1].prenorm, [0, 1])):
        for buf, want in (("mean", 0.0), ("std", std), ("lo", -1029.0), ("hi", 1029.0)):
            assert torch.allclose(getattr(pre, buf)[idx], torch.tensor(want))
    x = torch.randn(2, 4, 8, 8) * 50
    x[:, 0], x[:, 3] = torch.rand(2, 8, 8) * 8 + 0.5, torch.randn(2, 8, 8)  # inside the scalars' bounds (+-9)
    x[:, 1] += 147.0
    for k in range(SYMMETRIES):
        rotated = on.normalized(symmetry(x.clone(), k, SRBA_OOOO))
        assert torch.allclose(rotated, symmetry(on.normalized(x), k, SRBA_OOOO), atol=1e-6)
    # per-component statistics: a transposed sim puts u into the channel normalised for v
    assert not torch.allclose(
        off.normalized(symmetry(x.clone(), 1, SRBA_OOOO)), symmetry(off.normalized(x), 1, SRBA_OOOO)
    )
    xs = on.select(x)
    on.init_from_data(xs)
    assert torch.allclose(on.inverse(on(xs)), xs, atol=1e-3, rtol=1e-4)


def test_inverse_gradient_finite_for_large_linear_fields():
    pre = PreNorm([0.0, 0.0], [1.0, 1.0], [False, True])
    z = torch.tensor([[1e5, 1.0]], requires_grad=True)  # pressure in Pa: exp overflows
    pre.inverse(z).sum().backward()
    assert z.grad.isfinite().all()


def _run(folder: Path, data: dict, stats: dict, adapter: GroupedAdapter, adapter_cfg: dict) -> Path:
    folder.mkdir(parents=True)
    (folder / "stats.json").write_text(json.dumps({"data_normalization_stats": stats}))
    cfg = {"data": data | {"stats_json": str(folder / "stats.json")}, "adapter": adapter_cfg}
    (folder / "config.yaml").write_text(yaml.safe_dump(cfg))
    with torch.no_grad():
        for p in adapter.parameters():
            p.add_(0.1)
    torch.save(adapter.state_dict(), folder / "adapter.pt")
    return folder


def test_load_adapter_reads_the_dataset(tmp_path):
    srba = _srba_adapter(["pressure"])
    groups = [list(g) for g in SRBA_OOOO.groups]
    run = _run(
        tmp_path / "srba", {"dataset": "srba_oooo"}, SRBA_STATS, srba, {"groups": groups, "log_fields": ["pressure"]}
    )
    loaded = load_adapter(run, "cpu")
    assert run_dataset(run) == "srba_oooo" and loaded.fields == srba.fields
    assert all(torch.equal(v, loaded.state_dict()[k]) for k, v in srba.state_dict().items())
    for a, b in zip(srba.adapters, loaded.adapters, strict=True):  # non-persistent: rebuilt from log_fields
        assert torch.equal(a.prenorm.lo, b.prenorm.lo) and torch.equal(a.prenorm.hi, b.prenorm.hi)
    eulermq = GroupedAdapter([["density", "pressure"]], STATS, EULERMQ)
    run = _run(tmp_path / "old", {}, STATS, eulermq, {"groups": [["density", "pressure"]]})
    loaded = load_adapter(run, "cpu")
    assert run_dataset(run) == "eulermq" and loaded.adapters[0].prenorm.log_mask.all()
    assert all(torch.equal(v, loaded.state_dict()[k]) for k, v in eulermq.state_dict().items())


def test_load_adapter_restores_the_shared_vector_scale(tmp_path):
    groups = [list(g) for g in SRBA_OOOO.groups]
    run = _run(
        tmp_path / "run",
        {"dataset": "srba_oooo"},
        SKEWED,
        _skewed(True),
        {"groups": groups, "shared_vector_scale": True},
    )
    loaded, fresh = load_adapter(run, "cpu"), _skewed(True)
    for a, b in zip(fresh.adapters, loaded.adapters, strict=True):
        assert torch.equal(a.prenorm.lo, b.prenorm.lo) and torch.equal(a.prenorm.hi, b.prenorm.hi)
    assert torch.equal(fresh.base_norm.std, loaded.base_norm.std)
    assert not torch.equal(_skewed(False).adapters[1].prenorm.hi, loaded.adapters[1].prenorm.hi)


def test_shared_vector_scale_refuses_log_components():
    sets = [*NAMED, "adapter.log_fields=[velocityX]", "adapter.shared_vector_scale=true"]
    with pytest.raises(ValueError, match="vector components"):
        VaeStageConfig(**load_config(Path("configs/wan/srba_oooo/vae.yaml"), sets))
    with pytest.raises(ValueError, match="log fields"):
        GroupedAdapter([["velocityY", "velocityX"]], SKEWED, SRBA_OOOO, ["velocityX"], shared_vector_scale=True)


def test_load_weights_without_adapter_takes_only_the_vae(tmp_path):
    vae = SimpleNamespace(vae=torch.nn.Linear(2, 2))
    state = {k: torch.full_like(v, 7.0) for k, v in vae.vae.state_dict().items()}
    torch.save(state, tmp_path / "vae.pt")  # no adapter.pt in the folder
    load_weights(tmp_path, None, [vae], torch.device("cpu"))
    assert (vae.vae.weight == 7).all()


def test_index_and_log_show_the_dataset_symbols(tmp_path, capsys):
    run = tmp_path / "results/wan/srba_oooo/vae/r"
    run.mkdir(parents=True)
    final = {f"vrmse_{f}": 0.1 * (i + 1) for i, f in enumerate(SRBA_OOOO.fields)} | {"vrmse_mean": 0.25}
    (run / "metrics.json").write_text(json.dumps({"final": final}))
    (run / "config.yaml").write_text(yaml.safe_dump({"description": "d", "train": {"steps": 0}}))
    head, _, row = write_index(run.parent).read_text().splitlines()[-3:]
    assert "| p | u | v | ln ρ |" in head and "| 0.100 | 0.200 | 0.300 | 0.400 |" in row
    log_metrics("val", final)
    assert "per field  p 0.100  u 0.200  v 0.300  ln ρ 0.400" in capsys.readouterr().out
