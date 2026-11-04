"""Standardized rollout videos, comparable across MAGI-1, LTX and Wan (CFD_Rollout_Visualization_Spec, Harish).

One video per (trajectory, model, channel): ground truth | prediction | |prediction - truth| / sigma, all in
physical units. Every colour limit comes from the training-set statistics of the channel
(utils/rollout_video_stats.py -> rollout_video_stats.json, shared by every collaborator), never from the clip
or the model, so the same physical value has the same colour in every model's video and every frame:

    fields  Phi_c(v) = [asinh((v - mu)/sigma) - asinh((a - mu)/sigma)] / D_c   = AsinhNorm(sigma, a - mu, b - mu)
            on v - mu, viridis; below a cyan, above b magenta, non-finite gray
    error   e_c = |u_hat - u| / sigma,  Psi_c(e) = asinh(e) / asinh(E_max),  E_max = (b - a) / sigma
            = AsinhNorm(1, 0, E_max), inferno; above E_max magenta, non-finite gray

Each frame is a 9.6 x 11.2 in figure at 100 dpi = 960 x 1120 px (checked on the rendered frame and the encoded
video) with the complete formulas, the six numeric statistics, the frame and its RMSE / MAE / out-of-range
fractions burnt in. Next to each video, <video>.json holds the configuration and the per-frame numbers.

    python -m magi_finetuning.eval.rollout_video --check       (the spec's acceptance checks, CPU, seconds)
"""

import json
import os
from pathlib import Path

from magi_finetuning.compat import IG_WORK

os.environ.setdefault("MPLCONFIGDIR", str(IG_WORK / ".cache" / "matplotlib"))  # before matplotlib loads
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import AsinhNorm  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter  # noqa: E402

from magi_finetuning.settings import OUTPUTS  # noqa: E402

STATS_FILE = OUTPUTS / "rollout_video_stats.json"
FIGSIZE = (9.6, 11.2)  # inches (width, height)
DPI = 100
FRAME_SIZE = (960, 1120)  # pixels (width, height)
FIELD_CMAP = plt.get_cmap("viridis").with_extremes(under="cyan", over="magenta", bad="gray")
ERROR_CMAP = plt.get_cmap("inferno").with_extremes(over="magenta", bad="gray")
# Panel layout in figure fractions (left, bottom, width, height): fixed for every frame, video and model.
IMAGE_AXES = [(0.016 + k * 0.330, 0.640, 0.308, 0.264) for k in range(3)]
BAR_AXES = [(0.016 + k * 0.330 + 0.010, 0.597, 0.288, 0.014) for k in range(3)]
ANNOTATION_TOP = 0.515
NICE = [m * 10.0 ** e for e in range(-3, 4) for m in (1, 2, 5)]


def load_stats(path=STATS_FILE) -> dict:
    return json.loads(Path(path).read_text())


def norms(stats: dict):
    """(field norm on v - mu, error norm on e, E_max) of one channel."""
    mu, sigma, a, b = (float(stats[k]) for k in ("mean", "std", "min", "max"))
    if not (a <= float(stats["p0_5"]) <= float(stats["p99_5"]) <= b):
        raise ValueError("percentiles must lie inside the training extrema")
    if not (sigma > 0 and b > a):
        raise ValueError("constant / invalid channel: needs an explicit special case")
    error_limit = (b - a) / sigma
    return (AsinhNorm(linear_width=sigma, vmin=a - mu, vmax=b - mu, clip=False),
            AsinhNorm(linear_width=1.0, vmin=0.0, vmax=error_limit, clip=False), error_limit)


def spaced_ticks(candidates, ends, position, min_gap=0.14) -> list[float]:
    """The ends plus the candidates whose colour position (0..1) keeps min_gap to every tick already chosen
    (0 first, then powers of ten, e.g. e = 1 sigma, then the largest)."""
    chosen = list(ends)
    power_of_ten = lambda v: v != 0 and np.isclose(np.log10(abs(v)) % 1, 0)
    for value in sorted(candidates, key=lambda v: (v != 0, not power_of_ten(v), -abs(v))):
        if all(abs(position(value) - position(other)) >= min_gap for other in chosen):
            chosen.append(value)
    return sorted(chosen)


def annotation_text(stats: dict, unit: str, time_label: str, frame: dict) -> str:
    L, U, a, b, mu, sigma = (float(stats[k]) for k in ("p0_5", "p99_5", "min", "max", "mean", "std"))
    rmse = "undefined (non-finite pixels)" if frame["rmse"] is None else f"{frame['rmse']:.4g} {unit}"
    return "\n".join([
        r"$D_c=\mathrm{asinh}((b_c-\mu_c)/\sigma_c)-\mathrm{asinh}((a_c-\mu_c)/\sigma_c)$",
        r"$\Phi_c(v)=[\mathrm{asinh}((v-\mu_c)/\sigma_c)-\mathrm{asinh}((a_c-\mu_c)/\sigma_c)]/D_c$",
        r"$I_{\mathrm{GT},c}=\Phi_c(u_c)=[\mathrm{asinh}((u_c-\mu_c)/\sigma_c)"
        r"-\mathrm{asinh}((a_c-\mu_c)/\sigma_c)]/D_c$",
        r"$I_{\mathrm{pred},c}=\Phi_c(\hat u_c)=[\mathrm{asinh}((\hat u_c-\mu_c)/\sigma_c)"
        r"-\mathrm{asinh}((a_c-\mu_c)/\sigma_c)]/D_c$",
        r"$\Delta_c=|\hat u_c-u_c|,\quad e_c=\Delta_c/\sigma_c=|\hat u_c-u_c|/\sigma_c,"
        r"\quad E_{\max,c}=(b_c-a_c)/\sigma_c$",
        r"$\Psi_c(e)=\mathrm{asinh}(e)/\mathrm{asinh}(E_{\max,c})$",
        r"$I_{\mathrm{err},c}=\Psi_c(e_c)=\mathrm{asinh}(|\hat u_c-u_c|/\sigma_c)"
        r"/\mathrm{asinh}((b_c-a_c)/\sigma_c)$   (PLOTTED ERROR INTENSITY)",
        "",
        r"$L_c$" + f" = {L:.6g} {unit}    " + r"$U_c$" + f" = {U:.6g} {unit}    "
        + r"$a_c$" + f" = {a:.6g} {unit}    " + r"$b_c$" + f" = {b:.6g} {unit}",
        r"$\mu_c$" + f" = {mu:.6g} {unit}    " + r"$\sigma_c$" + f" = {sigma:.6g} {unit}    "
        + r"$E_{\max,c}$" + f" = {(b - a) / sigma:.6g}    {time_label}",
        r"($L_c, U_c$ = 0.5th / 99.5th training percentiles, marked on the colour bars; not clipping limits)",
        "",
        f"frame RMSE = {rmse}    MAE = {frame['mae']:.4g} {unit}   (physical fields, un-clipped)",
        r"prediction outside $[a_c, b_c]$: " + f"{100 * frame['fraction_prediction_outside_field_range']:.3f} %    "
        + r"$e_c > E_{\max,c}$: " + f"{100 * frame['fraction_error_over_limit']:.3f} %    "
        + f"non-finite: {100 * frame['fraction_nonfinite']:.3f} %",
        r"markers: cyan $< a_c$, magenta $> b_c$ (fields) or $> E_{\max,c}$ (error), gray = non-finite",
    ])


def frame_numbers(gt: np.ndarray, pred: np.ndarray, sigma: float, a: float, b: float, error_limit: float) -> dict:
    """Per-frame numbers on the physical fields (never on plotted intensities)."""
    finite = np.isfinite(gt) & np.isfinite(pred)
    error = np.abs(pred - gt) / sigma
    return {
        "rmse": float(np.sqrt(np.mean((pred - gt) ** 2))) if finite.all() else None,
        "mae": float(np.mean(np.abs(pred[finite] - gt[finite]))) if finite.any() else float("nan"),
        "fraction_error_over_limit": float(np.mean(error[finite] > error_limit)) if finite.any() else float("nan"),
        "fraction_prediction_outside_field_range":
            float(np.mean((pred[finite] < a) | (pred[finite] > b))) if finite.any() else float("nan"),
        "fraction_nonfinite": float(1.0 - np.mean(finite)),
    }


def render_channel_video(gt: np.ndarray, pred: np.ndarray, stats: dict, channel: str, path: Path, *,
                         trajectory: str, model: str, frame_indices, given_frames: int = 1, fps: int = 8,
                         dataset: str = "", extra_metadata: dict | None = None) -> Path:
    """gt, pred: (T, H, W) in physical units, frame t shown as frame_indices[t] (the first given_frames are the
    given condition). Writes path.mp4 and path.json; returns the video path."""
    import imageio.v2 as imageio

    mu, sigma, a, b = (float(stats[k]) for k in ("mean", "std", "min", "max"))
    L, U = float(stats["p0_5"]), float(stats["p99_5"])
    unit = stats.get("unit", "dimensionless")
    field_norm, error_norm, error_limit = norms(stats)
    last = frame_indices[-1]

    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    assert tuple(round(v * fig.dpi) for v in fig.get_size_inches()) == FRAME_SIZE
    titles = ("ground truth " + r"$u_c$", "prediction " + r"$\hat u_c$",
              r"$e_c=|\hat u_c-u_c|/\sigma_c$" + "  (shown: " + r"$\Psi_c(e_c)$" + ")")
    images = []
    for k in range(3):
        ax = fig.add_axes(IMAGE_AXES[k])
        data, norm, cmap = ((gt[0] - mu, field_norm, FIELD_CMAP), (pred[0] - mu, field_norm, FIELD_CMAP),
                            (np.abs(pred[0] - gt[0]) / sigma, error_norm, ERROR_CMAP))[k]
        images.append(ax.imshow(data, norm=norm, cmap=cmap, interpolation="nearest"))
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(titles[k], fontsize=11)
        bar = fig.colorbar(images[-1], cax=fig.add_axes(BAR_AXES[k]), orientation="horizontal",
                           extend="max" if k == 2 else "both")
        bar.ax.tick_params(labelsize=8.5)
        if k < 2:  # ticks at physical values (colour bar coordinate z = v - mu)
            position = lambda v: float(field_norm(v - mu))
            nice = [s * n for n in NICE for s in (1, -1) if a < s * n < b] + ([0.0] if a < 0 < b else [])
            bar.ax.xaxis.set_major_locator(FixedLocator([v - mu for v in spaced_ticks(nice, (a, b), position)]))
            bar.ax.xaxis.set_major_formatter(FuncFormatter(lambda z, _: f"{z + mu:.3g}"))
            for value, name in ((L, r"$L_c$"), (U, r"$U_c$")):
                bar.ax.axvline(value - mu, color="white", lw=1.2)
                bar.ax.text(value - mu, 1.15, name, transform=bar.ax.get_xaxis_transform(), ha="center",
                            va="bottom", fontsize=8.5)
            bar.set_label(f"{channel} [{unit}]", fontsize=9, labelpad=2)
        else:
            position = lambda e: float(error_norm(e))
            nice = [n for n in NICE if n < error_limit]
            bar.ax.xaxis.set_major_locator(FixedLocator(spaced_ticks(nice, (0.0, error_limit), position)))
            bar.ax.xaxis.set_major_formatter(FuncFormatter(lambda e, _: f"{e:.3g}"))
            bar.set_label(r"$e_c$ [multiples of $\sigma_c$]", fontsize=9, labelpad=2)
    header = fig.text(0.5, 0.985, "", ha="center", va="top", fontsize=12)
    annotation = fig.text(0.03, ANNOTATION_TOP, "", ha="left", va="top", fontsize=10.5, linespacing=1.45)

    per_frame, frames = [], []
    for t in range(gt.shape[0]):
        numbers = frame_numbers(gt[t], pred[t], sigma, a, b, error_limit)
        per_frame.append({"frame": int(frame_indices[t]), "given": t < given_frames, **numbers})
        images[0].set_data(gt[t] - mu)
        images[1].set_data(pred[t] - mu)
        images[2].set_data(np.abs(pred[t] - gt[t]) / sigma)  # Psi_c is applied by error_norm, not here
        state = "given" if t < given_frames else "generated"
        time_label = f"frame {frame_indices[t]}/{last} ({state})"
        header.set_text(f"{channel}  |  {trajectory}  |  {model}\n{time_label}")
        annotation.set_text(annotation_text(stats, unit, f"t: {time_label}", numbers))
        fig.canvas.draw()
        rgb = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        assert rgb.shape[1::-1] == FRAME_SIZE, rgb.shape
        frames.append(rgb)
    plt.close(fig)

    video = path.with_name(path.name + ".mp4")  # appended: sim ids contain dots
    imageio.mimsave(video, frames, fps=fps, codec="libx264", quality=9, macro_block_size=16,
                    pixelformat="yuv420p")
    reader = imageio.get_reader(video)
    encoded = tuple(reader.get_meta_data()["size"])
    reader.close()
    if encoded != FRAME_SIZE:
        raise RuntimeError(f"encoded video is {encoded}, must be {FRAME_SIZE}")

    metadata = {
        "dataset": dataset, "trajectory": trajectory, "model": model, "channel": channel, "unit": unit,
        "statistics": {k: stats[k] for k in ("mean", "std", "min", "max", "p0_5", "p99_5")},
        "statistics_source": "training-set ground truth, physical units (rollout_video_stats.json)",
        "field_display": {"norm": "AsinhNorm on v - mean", "linear_width": sigma, "vmin": a - mu, "vmax": b - mu,
                          "colormap": "viridis", "under": "cyan", "over": "magenta", "bad": "gray"},
        "error_display": {"quantity": "|pred - gt| / std", "norm": "AsinhNorm", "linear_width": 1.0, "vmin": 0.0,
                          "vmax": error_limit, "colormap": "inferno", "over": "magenta", "bad": "gray"},
        "spatial": {"grid": list(gt.shape[1:]), "orientation": "imshow origin='upper': array row 0 at the top",
                    "crop": "none"},
        "frames": [int(i) for i in frame_indices], "given_frames": given_frames, "fps": fps,
        "figure": {"figsize_in": list(FIGSIZE), "dpi": DPI, "frame_px": list(FRAME_SIZE)},
        "encoded_px": list(encoded), "per_frame": per_frame, **(extra_metadata or {}),
    }
    path.with_name(path.name + ".json").write_text(json.dumps(metadata, indent=1))
    return video


def save_rollout_videos(prediction, truth, channel_names, all_stats: dict, output_dir: Path, *, trajectory: str,
                        model: str, name: str, fps: int = 8, dataset: str = "") -> list[Path]:
    """prediction, truth: (1, C, T, H, W) physical tensors, frames 0..T-1 with frame 0 given. One video per
    channel: <output_dir>/<name>_<channel>.mp4 (+ .json)."""
    gt = truth[0].float().cpu().numpy()
    pred = prediction[0].float().cpu().numpy()
    frame_indices = list(range(gt.shape[1]))
    return [render_channel_video(gt[c], pred[c], all_stats["channels"][channel], channel,
                                 output_dir / f"{name}_{channel}", trajectory=trajectory, model=model,
                                 frame_indices=frame_indices, given_frames=1, fps=fps, dataset=dataset)
            for c, channel in enumerate(channel_names)]


def acceptance_checks() -> None:
    """Section 7 of the spec on a synthetic channel (density-like stats)."""
    stats = {"mean": 0.9, "std": 0.66, "min": 0.023, "max": 30.3, "p0_5": 0.1, "p99_5": 3.5}
    field_norm, error_norm, error_limit = norms(stats)
    mu, sigma, a, b = 0.9, 0.66, 0.023, 30.3
    phi = lambda v: (np.arcsinh((v - mu) / sigma) - np.arcsinh((a - mu) / sigma)) / (
        np.arcsinh((b - mu) / sigma) - np.arcsinh((a - mu) / sigma))
    values = np.array([a, 0.5, mu, 3.0, 12.0, b])
    assert np.allclose(field_norm(values - mu), phi(values)), "AsinhNorm != Phi_c"
    assert np.isclose(field_norm(a - mu), 0) and np.isclose(field_norm(b - mu), 1)
    assert np.isclose(error_norm(0.0), 0) and np.isclose(error_norm(error_limit), 1)
    assert np.allclose(error_norm(np.array([0.3, 2.0])), np.arcsinh([0.3, 2.0]) / np.arcsinh(error_limit))
    assert phi(12.0) < 1 and phi(12.0) > phi(stats["p99_5"]), "a rare value inside [a, b] keeps its own colour"
    assert field_norm(b + 1 - mu) > 1 and field_norm(a - 0.01 - mu) < 0, "out of range must map outside [0, 1]"
    gt = np.full((8, 8), 1.0)
    pred = gt.copy(); pred[0, 0] = np.nan; pred[0, 1] = b + 5
    numbers = frame_numbers(gt, pred, sigma, a, b, error_limit)
    assert numbers["rmse"] is None and numbers["fraction_nonfinite"] == 1 / 64
    assert numbers["fraction_prediction_outside_field_range"] == 1 / 63
    same = frame_numbers(gt, gt, sigma, a, b, error_limit)
    assert same["rmse"] == 0 and same["mae"] == 0, "a perfect prediction gives zero error"
    print("acceptance checks passed")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true")
    if p.parse_args().check:
        acceptance_checks()
