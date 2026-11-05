"""Standardized CFD rollout videos (docs/video_spec.md): one row per channel, ground truth | prediction | signed error
on scales fixed by training statistics. ★ marks a change against the spec's reference code.

    python -m windinet.video rollout.npz stats.json out/model_0456 --title "model, trajectory 0456"

rollout.npz: gt and pred [T, C, H, W] in physical units, times [T], channels [C]. stats.json: {channel: Stats}.
Writes <out>.mp4 with the scales and statistics in its metadata (--json: also <out>.json with the per-frame numbers;
--per-channel: <out>_<channel>.mp4 each; --frame i: <out>_t<i>.png of time step i instead). From Python: `from windinet.video import write_video, write_frame`. Needs numpy, matplotlib and imageio-ffmpeg.
"""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import NotRequired, TypedDict

import imageio_ffmpeg
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import AsinhNorm

# ★ one row per channel and the height follows the rows (was one channel per video on a fixed 960 x 1120 px)
WIDTH, ROW, HEAD, LINE, DPI = 9.6, 2.5, 0.5, 0.2, 100  # inches
SPAN = 2.0  # ★ shown range in training std: mean ± 2σ (was min..max, set by rare shocks: fields looked flat)
FIELD_WIDTH = 0.1  # ★ asinh linear width of the fields in σ (was 1σ): fronts and vortices are 0.1-0.5σ
ERROR_WIDTH = 0.05  # ★ asinh linear width of the error in σ (was 1σ): typical errors are 0.01-0.5σ
TIME_UNIT = "s"  # The Well's euler_multi_quadrants: Δt 0.015 s
EXTREMES = {"under": "cyan", "over": "magenta", "bad": "gray"}


class Stats(TypedDict):
    """One channel's training ground truth, pooled over trajectories, points and times, in physical units."""

    mean: float
    std: float
    min: float
    max: float
    p0_5: NotRequired[float]  # 0.5th percentile, recorded in the .json only
    p99_5: NotRequired[float]  # 99.5th percentile, recorded in the .json only
    unit: NotRequired[str]
    signed: NotRequired[bool]  # ★ drawn diverging around 0 (momentum): shared, so every collaborator gets the same


@dataclass(frozen=True)
class Scale:
    """What a channel's colours mean, from its Stats alone: the same for every model and frame."""

    centre: float  # the asinh is linear around it
    lo: float  # shown range, physical units; beyond: magenta/cyan
    hi: float
    std: float  # unit of the error panel
    cmap: str


def scale(stats: Stats) -> Scale:
    mean, std = stats["mean"], stats["std"]
    if not (std > 0 and stats["max"] > stats["min"]):
        raise ValueError("constant or invalid channel: needs an explicit special case")
    if stats.get("signed"):  # ★ diverging around physical 0 with symmetric bounds, a channel policy the spec allows
        half = min(abs(mean) + SPAN * std, max(-stats["min"], stats["max"]))
        return Scale(0.0, -half, half, std, "RdBu_r")
    # ★ within the training extremes (density and pressure are positive)
    return Scale(mean, max(mean - SPAN * std, stats["min"]), min(mean + SPAN * std, stats["max"]), std, "viridis")


class RolloutFigure:
    """The frame of a rollout, one row per channel: built once, then filled per time step."""

    def __init__(
        self, names: list[str], stats: list[Stats], shape: tuple[int, int], title: str, signed_error: bool = False
    ):
        n, self.title, self.signed_error = len(names), title, signed_error
        self.scales = [scale(st) for st in stats]
        foot = 2 * LINE  # the scale line with a gap above and below
        height = 2 * round((HEAD + ROW * n + foot) * DPI / 2) / DPI  # even pixels for the video encoder
        self.fig, axes = plt.subplots(n, 3, figsize=(WIDTH, height), dpi=DPI, squeeze=False)
        self.fig.subplots_adjust(
            left=0.01, right=0.93, top=1 - HEAD / height, bottom=foot / height, wspace=0.25, hspace=0.2
        )
        # ★ the error shown up to 2σ (was (max - min)/σ, where errors stayed black), asinh width 0.05σ (was 1σ);
        # signed on request: too high or too low, a front ahead or behind
        if signed_error:
            error_norm, error_cmap = (
                AsinhNorm(ERROR_WIDTH, -SPAN, SPAN),
                plt.get_cmap("RdBu_r").with_extremes(**EXTREMES),
            )
            error_label, error_ticks = "(prediction − truth) / σ", [-2, -0.5, 0, 0.5, 2]
        else:  # magma holds magenta itself: cyan flags beyond the range
            error_norm, error_cmap = (
                AsinhNorm(ERROR_WIDTH, 0, SPAN),
                plt.get_cmap("magma").with_extremes(over="cyan", bad="gray"),
            )
            error_label, error_ticks = "|prediction − truth| / σ", [0, 0.1, 0.5, 2]
        self.images = []
        for row, name, st, s in zip(axes, names, stats, self.scales, strict=True):
            field_norm = AsinhNorm(FIELD_WIDTH * s.std, s.lo - s.centre, s.hi - s.centre)
            field_cmap = plt.get_cmap(s.cmap).with_extremes(**EXTREMES)
            images = [
                ax.imshow(np.zeros(shape), norm=norm, cmap=cmap, interpolation="nearest")  # sharp cells
                for ax, norm, cmap in zip(
                    row, (field_norm, field_norm, error_norm), (field_cmap, field_cmap, error_cmap), strict=True
                )
            ]
            for ax, label in zip(row, (f"{name}: ground truth", "prediction", error_label), strict=True):
                ax.set_title(label, fontsize=9)
                ax.set_axis_off()
            ticks = []
            for t in (s.lo, s.centre - s.std / 2, s.centre, s.centre + s.std / 2, s.hi):
                if s.lo <= t <= s.hi and all(
                    abs(field_norm(t - s.centre) - field_norm(u - s.centre)) > 0.15 for u in ticks
                ):
                    ticks.append(t)  # labels apart on the bar
            bar = self.fig.colorbar(images[1], ax=row[1], fraction=0.046, pad=0.03, extend="both")
            bar.set_ticks([t - s.centre for t in ticks], labels=[f"{t:.3g}" for t in ticks])  # physical values
            extend = "both" if signed_error else "max"
            bar_err = self.fig.colorbar(images[2], ax=row[2], fraction=0.046, pad=0.03, extend=extend)
            bar_err.set_ticks(error_ticks, labels=[f"{t:g}σ" if abs(t) == SPAN else f"{t:g}" for t in error_ticks])
            for b in (bar, bar_err):
                b.minorticks_off()
                b.ax.tick_params(labelsize=7)
            self.images.append(images)
        # ★ title and time on top, one scale line under the panels (was nine formula lines and the statistics: the
        # colorbars show physical values, the numbers are in the .json)
        self.fig.text(
            0.5,
            foot / (2 * height),
            f"fields: mean ± {SPAN:g}σ, asinh width {FIELD_WIDTH}σ    error: {'±' if signed_error else '0 .. '}{SPAN:g}σ, "
            f"asinh width {ERROR_WIDTH}σ    magenta/cyan: beyond the range",
            fontsize=8.5,
            ha="center",
            va="center",
        )

    def frame(self, t: float, gt: np.ndarray, pred: np.ndarray) -> tuple[np.ndarray, list[dict]]:
        """RGB image of time t (gt, pred [C, H, W]) and per channel its RMSE and out-of-range fractions."""
        numbers = []
        for images, s, g, p in zip(self.images, self.scales, gt, pred, strict=True):
            error = (p - g) / s.std
            shown = error if self.signed_error else np.abs(error)
            for image, data in zip(images, (g - s.centre, p - s.centre, shown), strict=True):
                image.set_data(data)
            finite = np.isfinite(g) & np.isfinite(p)  # non-finite pixels are counted, never scored
            numbers.append(
                {
                    "rmse": float(np.sqrt(np.mean((p - g) ** 2))) if finite.all() else None,
                    "fraction_error_beyond": float(np.mean(np.abs(error[finite]) > SPAN)),
                    "fraction_prediction_beyond": float(np.mean((p[finite] < s.lo) | (p[finite] > s.hi))),
                    "fraction_nonfinite": float(1 - finite.mean()),
                }
            )
        self.fig.suptitle(f"{self.title}, t = {t:.3g} {TIME_UNIT}", fontsize=11, y=1 - 0.15 / self.fig.get_figheight())
        self.fig.canvas.draw()
        return np.asarray(self.fig.canvas.buffer_rgba())[..., :3].copy(), numbers


def write_frame(gt, pred, t: float, names, stats, path: Path, title: str = "", signed_error: bool = False) -> None:
    """One time step (gt, pred [C, H, W]) as a png."""
    figure = RolloutFigure(names, stats, gt.shape[1:], title, signed_error)
    plt.imsave(path, figure.frame(t, gt, pred)[0])
    plt.close(figure.fig)


def write_video(
    gt: np.ndarray,  # [T, C, H, W], physical units
    pred: np.ndarray,  # [T, C, H, W]
    times: np.ndarray,  # [T], physical time
    names: list[str],  # [C]
    stats: list[Stats],  # [C], training ground truth
    path: Path,  # <path>.mp4, <path>.json
    title: str = "",
    slowdown: float = 8.0,  # video seconds per physical second
    save_json: bool = False,
    signed_error: bool = False,
) -> list[dict]:
    """The rollout's video, its scales, statistics and mapping in the mp4's comment tag (ffprobe shows them); with
    `save_json` also a .json with them and the per-frame numbers. Returns the per-frame numbers."""
    figure, frames = RolloutFigure(names, stats, gt.shape[2:], title, signed_error), []
    size = figure.fig.canvas.get_width_height()
    # ★ playback follows physical time (was a fixed fps): rollouts with any time step play in step side by side
    fps = (len(times) - 1) / ((times[-1] - times[0]) * slowdown)
    meta = {
        "title": title,
        "fps": fps,
        "slowdown": slowdown,
        "frame_size": size,
        "stats": dict(zip(names, stats, strict=True)),
    }
    meta["scales"] = {name: asdict(s) for name, s in zip(names, figure.scales, strict=True)}
    meta["mapping"] = {
        "fields": f"AsinhNorm(linear_width={FIELD_WIDTH} * std, vmin=lo - centre, vmax=hi - centre)(value - centre)",
        "error": f"AsinhNorm(linear_width={ERROR_WIDTH}, vmin=-{SPAN}, vmax={SPAN})((pred - truth) / std)"
        if signed_error
        else f"AsinhNorm(linear_width={ERROR_WIDTH}, vmin=0, vmax={SPAN})(|pred - truth| / std)",
    }
    writer = imageio_ffmpeg.write_frames(
        str(path.with_suffix(".mp4")),
        size,
        fps=fps,
        macro_block_size=1,
        output_params=["-metadata", f"comment={json.dumps(meta)}"],
    )
    writer.send(None)
    for t, g, p in zip(times, gt, pred, strict=True):
        image, numbers = figure.frame(t, g, p)
        writer.send(image)
        frames.append({"time": float(t), **dict(zip(names, numbers, strict=True))})
    writer.close()
    plt.close(figure.fig)
    if save_json:
        path.with_suffix(".json").write_text(json.dumps(meta | {"frames": frames}, indent=1))
    return frames


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="video (or one frame) of a rollout against its ground truth")
    parser.add_argument("rollout", type=Path, help="npz: gt, pred [T, C, H, W], times [T], channels [C]")
    parser.add_argument("stats", type=Path, help="json: {channel: Stats} of the training ground truth")
    parser.add_argument("out", type=Path, help="output path without extension, e.g. videos/model_0456")
    parser.add_argument("--title", default="", help="shown before the time, e.g. model and trajectory")
    parser.add_argument("--slowdown", type=float, default=8.0, help="video seconds per physical second")
    parser.add_argument("--json", action="store_true", help="also a .json with the per-frame RMSE and fractions")
    parser.add_argument("--signed-error", action="store_true", help="signed error (pred - truth) instead of |error|")
    parser.add_argument("--per-channel", action="store_true", help="one video per channel")
    parser.add_argument("--frame", type=int, help="only this time step, as a png")
    args = parser.parse_args()
    data, all_stats = np.load(args.rollout), json.loads(args.stats.read_text())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    names = [str(c) for c in data["channels"]]
    groups = [[i] for i in range(len(names))] if args.per_channel else [list(range(len(names)))]
    for idx in groups:
        sel = [names[i] for i in idx]
        common = (sel, [all_stats[c] for c in sel])
        out = args.out.with_name(f"{args.out.name}_{sel[0]}") if args.per_channel else args.out
        if args.frame is None:
            write_video(
                data["gt"][:, idx],
                data["pred"][:, idx],
                data["times"],
                *common,
                out,
                args.title,
                args.slowdown,
                args.json,
                args.signed_error,
            )
        else:
            t = args.frame
            write_frame(
                data["gt"][t, idx],
                data["pred"][t, idx],
                data["times"][t],
                *common,
                out.with_name(f"{out.name}_t{t}.png"),
                args.title,
                args.signed_error,
            )
