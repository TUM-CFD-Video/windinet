import torch

from windinet.data.frames import val_split
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
    a = GroupedAdapter([FIELDS[:4]], STATS)
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
