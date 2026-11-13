"""Tiny HDF5 files shaped like the real datasets: 9 frames of 8×8, fields (T, 1, H, W) float64."""

import h5py
import numpy as np
import pytest

from windinet.data.specs import EULERMQ, SRBA_OOOO

FRAMES, SIZE = 9, 8


def make_h5(path, spec, ids, size=SIZE):
    """One group per sim id with the spec's stored fields; non-vector fields are positive."""
    stored = [f.removeprefix("log_") for f in spec.fields]
    with h5py.File(path, "w") as f:
        for i, sid in enumerate(ids):
            rng = np.random.default_rng(i)
            for name in stored:
                x = rng.standard_normal((FRAMES, 1, size, size))
                f.create_dataset(f"{sid}/{name}", data=x if name in (spec.vector or ()) else np.exp(x))
    return path


def srba_id(i, mach, bubbles):
    """An id in the real format, e.g. 005_Mas1.30_sb1_Ax0.0129_Ay0.0081_Ar0.0035_BxNaN_ByNaN_BrNaN."""
    return f"{i:03d}_Mas{mach:.2f}_sb{bubbles}_Ax0.0129_Ay0.0081_Ar0.0035_BxNaN_ByNaN_BrNaN"


@pytest.fixture(name="make_h5")
def make_h5_fixture():
    return make_h5


@pytest.fixture
def srba_files(tmp_path):
    """(train.h5, test.h5): 15 cells (Mach × bubbles) with 4 sims each in train and 1 in test."""
    cells = [(m, b) for m in (1.3, 1.7, 2.1) for b in range(1, 6)]
    train = [srba_id(4 * c + j, m, b) for c, (m, b) in enumerate(cells) for j in range(4)]
    test = [srba_id(c, m, b) for c, (m, b) in enumerate(cells)]
    return make_h5(tmp_path / "train.h5", SRBA_OOOO, train), make_h5(tmp_path / "test.h5", SRBA_OOOO, test)


@pytest.fixture
def eulermq_file(tmp_path):
    """3 gammas × 5 sims."""
    ids = [f"{5 * g + j:04d}_gamma{gamma}" for g, gamma in enumerate((1.13, 1.4, 1.76)) for j in range(5)]
    return make_h5(tmp_path / "eulermq.h5", EULERMQ, ids)
