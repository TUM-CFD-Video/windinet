import torch

from windinet.field_adapter import FieldAdapter


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
