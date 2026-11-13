import numpy as np
import pytest
from matplotlib import pyplot as plt

from windinet.video import RolloutFigure, scale

DENSITY = {"mean": 1.0, "std": 0.5, "min": 0.1, "max": 5.0}
MOMENTUM = {"mean": 0.1, "std": 0.4, "min": -3.0, "max": 3.0, "signed": True}


def render(stats, gt, pred=None, **kwargs):
    figure = RolloutFigure([f"c{i}" for i in range(len(stats))], stats, gt.shape[1:], "run", **kwargs)
    image, numbers = figure.frame(0.25, gt, gt if pred is None else pred)
    texts = [figure.fig.texts[0].get_text(), figure.fig.get_suptitle()]  # scale line, title and time
    plt.close(figure.fig)
    return image, numbers, texts, figure


def test_default_scale_is_mean_2std_within_extremes():
    s = scale(DENSITY)
    assert (s.lo, s.hi, s.centre, s.width, s.cmap) == (0.1, 2.0, 1.0, 0.1, "viridis")
    s = scale(MOMENTUM)
    assert (s.lo, s.hi, s.centre) == (pytest.approx(-0.9), pytest.approx(0.9), 0.0)


def test_default_footer_and_time_unchanged():
    _, _, texts, _ = render([DENSITY, MOMENTUM], np.ones((2, 4, 6)))
    assert texts[0] == (
        "fields: mean ± 2σ, asinh width 0.1σ    error: 0 .. 2σ, asinh width 0.05σ    magenta/cyan: beyond the range"
    )
    assert texts[1] == "run, t = 0.25 s"


def test_given_bounds_centre_and_width():
    given = DENSITY | {"lo": 0.1, "hi": 9.0, "centre": 0.1, "width": 0.5}
    s = scale(given)
    assert (s.lo, s.hi, s.centre, s.width) == (0.1, 9.0, 0.1, 0.5)
    _, _, texts, figure = render([given], np.ones((1, 4, 6)))
    norm = figure.images[0][0].norm
    assert (norm.linear_width, norm.vmin, norm.vmax) == (0.25, 0.0, pytest.approx(8.9))
    assert texts[0].startswith("fields: given bounds, asinh width 0.5σ")
    assert render([given, DENSITY], np.ones((2, 4, 6)))[2][0].startswith("fields: per channel, asinh width per channel")


def test_bounds_must_be_ordered():
    with pytest.raises(ValueError):
        scale(DENSITY | {"lo": 2.0, "hi": 2.0})


def test_signed_bounds_symmetric():
    s = scale(MOMENTUM | {"lo": -2.0, "hi": 1.5})
    assert (s.lo, s.hi, s.centre) == (-2.0, 2.0, 0.0)
    s = scale(MOMENTUM | {"lo": -0.5, "centre": 1.0})  # one bound given: its magnitude; the centre stays 0
    assert (s.lo, s.hi, s.centre) == (-0.5, 0.5, 0.0)


def test_origin_lower_flips_rows():
    gt = np.ones((1, 8, 8))
    gt[0, 0] = 100.0  # beyond the range: magenta in the truth panel
    rows = []
    for origin in ("upper", "lower"):
        image = render([DENSITY], gt, origin=origin)[0]
        panel = image[:, : image.shape[1] // 3]
        rows.append(np.nonzero((panel == [255, 0, 255]).all(-1))[0].mean())
    assert rows[1] > rows[0]


def test_time_unit():
    assert render([DENSITY], np.ones((1, 4, 6)), time_unit="µs")[2][1] == "run, t = 0.25 µs"


def test_out_of_range_counted_with_given_bounds():
    pred = np.ones((1, 4, 5))
    pred[0, 0] = 9.5  # 5 of 20 pixels above hi
    numbers = render([DENSITY | {"lo": 0.1, "hi": 9.0}], np.ones((1, 4, 5)), pred)[1]
    assert numbers[0]["fraction_prediction_beyond"] == 0.25
