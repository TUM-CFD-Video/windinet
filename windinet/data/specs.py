"""The datasets the pipeline runs on: every fact the code needs about each, in one place."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from windinet.training.shockwave_data import parse_gamma


@dataclass(frozen=True)
class Display:
    """How videos, logs and the results index show a dataset."""

    symbols: dict[str, str]  # per field
    frame_dt: float  # time between stored frames, in time_unit
    time_unit: str
    slowdown: float  # video seconds per time_unit
    origin: str  # imshow's; 'lower' draws row 0 at the bottom
    condition: Callable[[dict[str, float]], str]  # scalars -> title text
    widths: dict[str, float] | None  # asinh width in σ per field (absent: linear); None: video.py defaults


@dataclass(frozen=True)
class Spec:
    """One dataset. A field log_<f> is read as ln of the stored f."""

    name: str
    fields: tuple[str, ...]  # FrameSampler's channel order
    vector: tuple[str, str] | None  # vector components along H (rows) and W (columns)
    groups: tuple[tuple[str, ...], ...]  # default adapter images
    log_fields: frozenset[str]  # default adapter log fields; also the loss space (base_norm)
    scalars: Callable[[str], dict[str, float]]  # sim id -> condition; the first one orders the example sims
    val_per_cell: int  # >0: this many val sims per cell of equal scalars; 0: data.val_sims at random
    display: Display


def _srba_scalars(sid: str) -> dict[str, float]:
    """Mach number and bubble count from an id like 005_Mas1.30_sb1_Ax..."""
    return {"mach": float(sid.split("_Mas")[1].split("_")[0]), "bubbles": float(sid.split("_sb")[1].split("_")[0])}


EULERMQ = Spec(
    name="eulermq",
    fields=("density", "momentum_x", "momentum_y", "pressure"),
    vector=("momentum_x", "momentum_y"),  # The Well: axis H = x
    groups=(("density", "pressure"), ("momentum_x", "momentum_y")),
    log_fields=frozenset({"density", "pressure"}),
    scalars=lambda sid: {"gamma": parse_gamma(sid)},
    val_per_cell=0,
    display=Display(
        symbols={"density": "ρ", "momentum_x": "m_x", "momentum_y": "m_y", "pressure": "p"},
        frame_dt=0.015,
        time_unit="s",
        slowdown=8.0,
        origin="upper",
        condition=lambda c: f"γ {round(c['gamma'], 3):g}",
        widths=None,
    ),
)
SRBA_OOOO = Spec(
    name="srba_oooo",
    fields=("pressure", "velocityX", "velocityY", "log_density"),
    vector=("velocityY", "velocityX"),  # rows are y, columns x; the shock runs in +x
    groups=(("log_density", "pressure"), ("velocityX", "velocityY")),
    log_fields=frozenset(),
    scalars=_srba_scalars,
    val_per_cell=2,
    display=Display(
        symbols={"pressure": "p", "velocityX": "u", "velocityY": "v", "log_density": "ln ρ"},
        frame_dt=0.7,
        time_unit="µs",
        slowdown=0.17,
        origin="lower",
        condition=lambda c: f"Mach {c['mach']:g}, {c['bubbles']:g} bubble{'s' * (c['bubbles'] != 1)}",
        widths={"pressure": 0.5, "velocityY": 1.0},
    ),
)
SPECS = {s.name: s for s in (EULERMQ, SRBA_OOOO)}
Dataset = Literal[tuple(SPECS)]  # pydantic validates it as Literal["eulermq", "srba_oooo"]
