import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from windinet.data.frames import val_split
from windinet.data.specs import EULERMQ, SRBA_OOOO
from windinet.experiment.config import load_config
from windinet.experiment.log import log_metrics
from windinet.experiment.metrics import vrmse
from windinet.experiment.runs import run_dataset, write_index
from windinet.field_adapter import FieldAdapter, GroupedAdapter, load_adapter
from windinet.field_adapter.bijections import PreNorm
from windinet.losses import vrms_per_channel
from windinet.wan.corrector.config import CorrectorStageConfig
from windinet.wan.dit.config import DitStageConfig
from windinet.wan.vae.config import VaeStageConfig
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
