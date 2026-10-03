import torch

from windinet.eulermq.data import val_split
from windinet.experiment.metrics import vrmse
from windinet.field_adapter import FieldAdapter, GroupedAdapter
from windinet.losses import vrms_per_channel

FIELDS = ["density", "momentum_x", "momentum_y", "pressure", "log_density", "log_pressure"]
STATS = {k: {"mean": 0.0, "std": 1.0, "min": -3.0, "max": 3.0} for k in FIELDS}


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
    a = GroupedAdapter([["density", "momentum_x", "momentum_y"]], STATS)
    x = a.select(torch.rand(2, 4, 8, 8) + 0.1)
    a.init_from_data(x)
    assert a.fields == ["density", "momentum_x", "momentum_y"] and x.shape[1] == 3
    assert torch.allclose(a.inverse(a(x)), x, atol=1e-4)


def test_vrmse_equals_the_ltx_baselines_metric():
    x, x_hat = torch.rand(7, 4, 8, 8), torch.rand(7, 4, 8, 8)
    his = vrms_per_channel(x_hat.movedim(0, 1)[None], x.movedim(0, 1)[None])  # his layout: [B, C, T, H, W]
    assert torch.allclose(vrmse(x_hat, x), his, atol=1e-6)


def test_val_split_is_the_ltx_baselines_rule():
    ids = [f"{i:04d}_gamma1.3" for i in range(20)]
    train, val = val_split(ids, n_val=5, seed=42)
    perm = torch.randperm(20, generator=torch.Generator().manual_seed(42)).tolist()
    assert val == [ids[i] for i in perm[15:]] and len(train) == 15 and not set(train) & set(val)


def test_channel_moments_match_channel_stats():
    from windinet.experiment.latent_stats import ChannelMoments, channel_stats

    x = torch.randn(6, 5, 3, 4)
    moments = ChannelMoments(5)
    for part in x.split(2):
        moments.add(part)
    for k, v in channel_stats(x).items():
        torch.testing.assert_close(moments.stats()[k], v)
