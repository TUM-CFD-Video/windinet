import h5py
import numpy as np
import pytest
import torch
from pydantic import BaseModel, ValidationError

from windinet.data.frames import SYMMETRIES, FrameSampler, sim_ids, splits, symmetry, turn_vector, val_split
from windinet.data.specs import EULERMQ, SPECS, SRBA_OOOO, Dataset

SRBA_ID = "005_Mas1.30_sb1_Ax0.0129_Ay0.0081_Ar0.0035_BxNaN_ByNaN_BrNaN"


@pytest.mark.parametrize("spec", SPECS.values(), ids=SPECS)
def test_spec_names_refer_to_its_fields(spec):
    fields = set(spec.fields)
    assert SPECS[spec.name] is spec and set(spec.vector) <= fields and spec.log_fields <= fields
    assert {f for g in spec.groups for f in g} <= fields and set(spec.display.symbols) == fields
    assert set(spec.display.widths or {}) <= fields


@pytest.mark.parametrize("spec, sid", [(EULERMQ, "0001_gamma1.4"), (SRBA_OOOO, SRBA_ID)], ids=SPECS)
def test_ranges_follow_the_scalars_order(spec, sid):
    assert list(spec.ranges) == list(spec.scalars(sid))


def test_dataset_literal_validates_the_spec_names():
    class Config(BaseModel):
        dataset: Dataset = "eulermq"

    assert [Config(dataset=n).dataset for n in SPECS] == list(SPECS)
    with pytest.raises(ValidationError):
        Config(dataset="euler_mq")


def test_scalars_parse_both_id_formats():
    assert SRBA_OOOO.scalars(SRBA_ID) == {"mach": 1.3, "bubbles": 1.0}
    assert SRBA_OOOO.display.condition(SRBA_OOOO.scalars(SRBA_ID)) == "Mach 1.3, 1 bubble"
    assert EULERMQ.scalars("0001_gamma1.4040000439") == {"gamma": 1.4040000439}
    assert EULERMQ.display.condition(EULERMQ.scalars("0001_gamma1.4040000439")) == "γ 1.404"


def test_frame_sampler_reads_log_density_and_returns_scalars(srba_files):
    train, _ = srba_files
    sid = sim_ids(train)[0]
    sampler = FrameSampler(train, [sid], None, 0, spec=SRBA_OOOO)
    fields, scalars = sampler[0]
    with h5py.File(train, "r", locking=False) as f:  # as the open sampler, else HDF5 refuses
        density, pressure = f[sid]["density"][:, 0], f[sid]["pressure"][:, 0]
    assert fields.shape == (9, 4, 8, 8) and fields.dtype == torch.float32
    assert torch.equal(fields[:, 3], torch.from_numpy(np.log(density)).float())
    assert torch.equal(fields[:, 0], torch.from_numpy(pressure).float())
    assert scalars.dtype == torch.float32 and scalars.tolist() == [pytest.approx(1.3), 1.0]
    augmented = FrameSampler(train, [sid], None, 0, augment=True, spec=SRBA_OOOO)[0][0]
    assert any(torch.equal(augmented, symmetry(fields, k, SRBA_OOOO)) for k in range(SYMMETRIES))


def test_srba_split_holds_out_two_sims_per_cell(srba_files):
    train_h5, test_h5 = srba_files
    s = splits(SRBA_OOOO, train_h5, test_h5)
    train, val = s["train"][1], s["val"][1]
    cells = [tuple(SRBA_OOOO.scalars(sid).values()) for sid in val]
    assert len(val) == 30 and all(cells.count(c) == 2 for c in cells) and len(set(cells)) == 15
    assert not set(train) & set(val) and sorted(train + val) == sim_ids(train_h5)
    assert s["test"] == (test_h5, sim_ids(test_h5)) and len(s["test"][1]) == 15
    assert splits(SRBA_OOOO, train_h5, test_h5) == s and splits(SRBA_OOOO, train_h5, test_h5, split_seed=0) != s
    with pytest.raises(ValueError, match="test_h5"):
        splits(SRBA_OOOO, train_h5)


def test_eulermq_splits_are_unchanged(eulermq_file):
    ids = sim_ids(eulermq_file)
    held_out = ["0000_gamma1.13", "0005_gamma1.4", "0010_gamma1.76"]  # the first sim of each gamma
    one_file = splits(EULERMQ, eulermq_file)
    assert one_file["val"] == one_file["test"] == (eulermq_file, held_out)
    assert one_file["train"][1] == [sid for sid in ids if sid not in held_out]
    two_files = splits(EULERMQ, eulermq_file, eulermq_file, val_sims=3)
    assert (two_files["train"][1], two_files["val"][1]) == val_split(ids, 3, 42) and two_files["test"][1] == ids


@pytest.mark.parametrize("spec", SPECS.values(), ids=SPECS)
def test_symmetry_is_d4_and_turns_the_vector(spec):
    h, w = (spec.fields.index(n) for n in spec.vector)
    r = torch.arange(6.0) - 2.5  # centred coordinate: a radial vector field points away from the centre
    fields = torch.rand(2, len(spec.fields), 6, 6)
    fields[:, h], fields[:, w] = r[:, None], r[None, :]
    before = fields.clone()
    out = {k: symmetry(fields, k, spec) for k in range(SYMMETRIES)}
    assert torch.equal(out[0], fields) and torch.equal(fields, before)  # identity; the input is left alone
    for k in range(SYMMETRIES):
        assert torch.equal(out[k][:, [h, w]], fields[:, [h, w]]), k  # a radial vector field is invariant
        for j in range(SYMMETRIES):  # closure: every composition is an element
            assert any(torch.equal(symmetry(out[k], j, spec), o) for o in out.values()), (k, j)
    along_w = torch.zeros(1, len(spec.fields), 6, 6)
    along_w[:, w] = 1.0  # e.g. velocityX = 1: under the transpose it points along H
    swapped = symmetry(along_w, 1, spec)
    assert torch.equal(swapped[:, h], torch.ones(1, 6, 6)) and not swapped[:, w].any()


def test_turn_vector_turns_the_flow_as_symmetry_turns_the_fields():
    p, u, v = (SRBA_OOOO.fields.index(n) for n in ("pressure", "velocityX", "velocityY"))
    fields = torch.zeros(1, 4, 8, 8)
    fields[0, p], fields[0, u] = torch.arange(8.0), 1.0  # a pressure ramp along the flow, +W
    table = [(0, 1), (1, 0), (0, 1), (-1, 0), (0, -1), (1, 0), (0, -1), (-1, 0)]
    assert [turn_vector(SRBA_OOOO.direction, k) for k in range(SYMMETRIES)] == table
    for k in range(SYMMETRIES):
        t = symmetry(fields, k, SRBA_OOOO)[0]
        assert (t[v, 0, 0].item(), t[u, 0, 0].item()) == table[k], k
        assert (t[p].diff(dim=-2).mean().item(), t[p].diff(dim=-1).mean().item()) == table[k], k
