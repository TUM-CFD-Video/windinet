import torch

from windinet.field_adapter import FieldAdapter, MonotoneWarp


def _adapter(n=4, warp=True):
    return FieldAdapter(mean=[0.0] * n, std=[1.0] * n, log_channels=[True, False, False, True][:n], warp=warp)


def test_warp_is_identity_at_init_and_invertible_after_perturbation():
    warp = MonotoneWarp(channels=2, bins=8)
    x = torch.randn(3, 2, 16, 16) * 3
    assert torch.allclose(warp(x), x, atol=1e-5)
    with torch.no_grad():
        for p in warp.parameters():
            p.add_(torch.randn_like(p) * 0.5)
    assert torch.allclose(warp.inverse(warp(x)), x, atol=1e-4)
    assert (warp(x + 1e-3) > warp(x)).all()  # monotone


def test_three_to_three_round_trip_is_exact():
    a = _adapter(n=3)
    x = torch.rand(2, 3, 8, 8) + 0.1  # positive, so log channels are fine
    a.init_from_data(x)
    with torch.no_grad():
        for p in a.warp.parameters():
            p.add_(torch.randn_like(p) * 0.3)
    assert torch.allclose(a.inverse(a(x)), x, atol=1e-4, rtol=1e-4)
    assert a(x).abs().max() < 1.0


def test_four_to_three_round_trip_is_pseudo_inverse():
    a = _adapter(n=4)
    x = torch.rand(2, 4, 8, 8) + 0.1
    a.init_from_data(x)
    z = a.mix(a.warp(a.prenorm(x)))
    back = a.mix.inverse(z)
    # W W^+ W = W: the round trip is exact on the row space of W.
    assert torch.allclose(a.mix(back), z, atol=1e-5)
    assert a.inverse(a(x)).shape == x.shape
