"""Figures and videos written into a run folder: training curves, a reconstruction panel, strips and mp4s."""

from __future__ import annotations

from pathlib import Path

import torch

from windinet.field_adapter import group_tag


def to_uint8(video: torch.Tensor) -> torch.Tensor:
    """RGB [3, F, H, W] in (-1, 1) -> [F, H, W, 3] uint8."""
    return ((video.permute(1, 2, 3, 0) + 1) * 127.5).round().clamp(0, 255).to(torch.uint8).cpu()


def write_video(frames: list, path: Path, fps: int) -> None:
    """mp4 from uint8 [H, W, 3] frames. diffusers scales numpy frames by 255 (it expects floats in [0, 1]), so
    uint8 input would wrap around and invert every colour."""
    from diffusers.utils import export_to_video

    export_to_video([f / 255 for f in frames], str(path), fps=fps)


def save_video(video: torch.Tensor, path: Path, fps: int) -> None:
    """mp4 of an RGB [3, F, H, W] video in (-1, 1)."""
    write_video(list(to_uint8(video).numpy()), path, fps)


def save_strip(video: torch.Tensor, path: Path, reference: torch.Tensor | None = None, n: int = 5) -> None:
    """`n` frames spread over an RGB [3, F, H, W] video in (-1, 1), side by side; with `reference` (same shape) two
    more rows: the reference and |video - reference| on a fixed scale (0 to a quarter of the colour range), so strips of
    different runs compare directly."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [("generated", video)] if reference is None else [("generated", video), ("ground truth", reference)]
    idx = torch.linspace(0, video.shape[1] - 1, n).round().long()
    n_rows = len(rows) + (reference is not None)
    h, w = video.shape[-2:]
    fig, axes = plt.subplots(n_rows, n, figsize=(3 * n, 3 * n_rows * h / w), squeeze=False)
    for r, (label, v) in enumerate(rows):
        frames = to_uint8(v)
        for ax, i in zip(axes[r], idx, strict=True):
            ax.imshow(frames[i])
            ax.set_title(f"{label}, frame {i}", fontsize=9)
    if reference is not None:
        diff = (video - reference).abs().mean(0).cpu()  # [F, H, W], mean over RGB
        for ax, i in zip(axes[-1], idx, strict=True):
            ax.imshow(diff[i], cmap="magma", vmin=0, vmax=0.5)
            ax.set_title(f"|difference|, frame {i}", fontsize=9)
    for ax in axes.flat:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def save_examples(
    examples: dict,
    fields: list[str],
    stats_json: str,
    run_dir: Path,
    weights_dir: str | None,
    kind: str,
    frame_stride: int,
) -> None:
    """<kind>_gamma<g>.mp4 (windinet/video.py: truth, prediction, |error| per field on train-stat scales) and
    example_gamma<g>.pt of the examples {gamma: (sim id, truth, pred)}, [F, C, H, W] physical fields."""
    from windinet.eulermq.data import FRAME_DT, load_field_stats
    from windinet.video import write_video
    from windinet.wan.weights import big_file

    stats = load_field_stats(stats_json)
    shown = [{**stats[n], "signed": n.startswith("momentum")} for n in fields]
    for g, (sid, truth, pred) in examples.items():
        times = [FRAME_DT * frame_stride * i for i in range(len(truth))]
        title = f"sim {sid.split('_')[0]} (γ {g:g})"
        path = big_file(run_dir, f"{kind}_gamma{g:g}.mp4", weights_dir)
        write_video(truth.numpy(), pred.numpy(), times, fields, shown, path, title)
        torch.save({"id": sid, "truth": truth, "pred": pred}, big_file(run_dir, f"example_gamma{g:g}.pt", weights_dir))


def _curves(steps: list[dict], history: list[dict], fields: list[str]):
    """A 2 x 2 figure with the shared top row: train loss per update (10-update mean on top: shock frames make it
    heavy-tailed) and validation VRMSE per field over the evaluations (raw weights dashed when the EMA is evaluated,
    the train-sim mean dashed black). The bottom row is the stage's."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    loss = torch.tensor([s["loss"] for s in steps])
    ax[0, 0].semilogy([s["step"] for s in steps], loss, lw=0.5, alpha=0.5)
    if len(loss) >= 10:
        ax[0, 0].semilogy(range(10, len(loss) + 1), loss.unfold(0, 10, 1).mean(1), "k", label="10-update mean")
        ax[0, 0].legend(fontsize=8, loc="upper right")
    ax[0, 0].set(title="train loss per update", xlabel="step")
    x = [h["step"] for h in history]
    for name in fields:
        (line,) = ax[0, 1].plot(x, [h[f"vrmse_{name}"] for h in history], marker="o", label=name)
        if history and f"raw_vrmse_{name}" in history[-1]:
            ax[0, 1].plot(x, [h[f"raw_vrmse_{name}"] for h in history], "--", c=line.get_color(), label=f"{name} raw")
    if history and "train_vrmse_mean" in history[-1]:
        ax[0, 1].plot(x, [h["train_vrmse_mean"] for h in history], "k--", marker="o", label="train mean")
    ax[0, 1].set(title="validation VRMSE", xlabel="step", ylim=(0, None))
    ax[0, 1].legend(fontsize=8)
    return fig, ax


def _save(fig, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def _lr_and_gain(ax_loss, ax_val, steps: list[dict], history: list[dict], fields: list[str]) -> None:
    """The lr per update over the train loss, and over the validation curve its gain per unit lr: the drop of the
    field-mean VRMSE between two evaluations divided by the summed lr of the updates in between. A flat gain means the
    curve flattens only because the cosine lowers the lr; a gain falling to 0 means the model saturates."""
    lr = ax_loss.twinx()
    lr.plot([s["step"] for s in steps], [s["lr"] for s in steps], c="gray", lw=1)
    lr.set_ylabel("lr", color="gray")
    if len(history) < 2:
        return
    mean = [sum(h[f"vrmse_{n}"] for n in fields) / len(fields) for h in history]
    summed = torch.tensor([s["lr"] for s in steps]).cumsum(0)
    at = [summed[h["step"] - 1].item() for h in history]
    gain = [(mean[i - 1] - mean[i]) / (at[i] - at[i - 1]) for i in range(1, len(history))]
    twin = ax_val.twinx()
    twin.plot([h["step"] for h in history[1:]], gain, "r^:", label="gain per summed lr")
    twin.axhline(0, c="r", lw=0.5)
    twin.set_ylabel("val gain per summed lr", color="r")


def save_dit_curves(steps: list[dict], history: list[dict], final: dict, fields: list[str], path: Path) -> None:
    """The shared top row, then the VRMSE per frame (last validation, else the test) and per gamma (last validation
    and test)."""
    fig, ax = _curves(steps, history, fields)
    if steps and "lr" in steps[0]:
        _lr_and_gain(ax[0, 0], ax[0, 1], steps, history, fields)
    last = history[-1] if history else final  # an eval-only run has no validation
    what = f"validation, step {last['step']}" if history else "test"
    curve = last["vrmse_per_frame"]
    ax[1, 0].plot(range(1, len(curve)), curve[1:], "k")  # frame 0 is the given frame: left out
    ax[1, 0].set(title=f"VRMSE per frame ({what})", xlabel="frame", ylim=(0, None))
    gammas = [k[len("vrmse_mean_gamma") :] for k in final if k.startswith("vrmse_mean_gamma")]
    x = torch.arange(len(gammas))
    if history:
        ax[1, 1].bar(x - 0.2, [last.get(f"vrmse_mean_gamma{g}", 0) for g in gammas], 0.4, label="validation")
    ax[1, 1].bar(x + 0.2 if history else x, [final[f"vrmse_mean_gamma{g}"] for g in gammas], 0.4, label="test")
    ax[1, 1].set(title="VRMSE per gamma, last evaluation", xticks=x, xticklabels=gammas, ylim=(0, None))
    ax[1, 1].tick_params(axis="x", labelsize=8)
    ax[1, 1].legend(fontsize=8)
    _save(fig, path)


def save_vae_curves(
    steps: list[dict],
    history: list[dict],
    fields: list[str],
    path: Path,
    blocks: dict[str, float] | None = None,
    groups: list[str] = ("adapter", "vae"),
) -> None:
    """The shared top row, then the relative weight change per update and param group (the VAE drift from its start
    weights dashed) and the drift per VAE block."""
    fig, ax = _curves(steps, history, fields)
    measured = [s for s in steps if "update_ratio" in s]
    n_groups = len(measured[0]["update_ratio"]) if measured else 0
    for g, label in enumerate(groups[:n_groups]):
        ax[1, 0].semilogy(
            [s["step"] for s in measured],
            [s["update_ratio"][g] for s in measured],
            lw=0.8,
            label=f"{label}: step |dw|/|w|",
        )
    if history and "vae_drift" in history[-1]:
        ax[1, 0].semilogy(
            [h["step"] for h in history], [h["vae_drift"] for h in history], "k--", marker="o", label="vae: drift"
        )
    ax[1, 0].set(title="relative weight change", xlabel="step")
    if ax[1, 0].lines:  # empty for an eval-only run
        ax[1, 0].legend(fontsize=8)
    ax[1, 1].barh(list(blocks or {}), list((blocks or {}).values()))
    ax[1, 1].set(title="drift from start weights per block" if blocks else "VAE frozen")
    ax[1, 1].tick_params(labelsize=7)
    _save(fig, path)


def save_panel(example: tuple, fields: list[str], path: Path) -> None:
    """Per field: GT / reconstruction / residual; last column the RGB image(s) fed to the VAE. Small PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x, x_hat, rgbs = example
    fig, axes = plt.subplots(len(fields), 4, figsize=(12, 2.75 * len(fields)), squeeze=False)
    for i, name in enumerate(fields):
        lo, hi = x[i].min().item(), x[i].max().item()
        same = {"vmin": lo, "vmax": hi}
        panels = [
            (x[i], f"{name} GT", same),
            (x_hat[i], "recon", same),
            (x_hat[i] - x[i], "residual", {"cmap": "RdBu_r"}),
        ]
        for j, (img, title, kw) in enumerate(panels):
            im = axes[i, j].imshow(img, **kw)
            axes[i, j].set_title(title, fontsize=9)
            plt.colorbar(im, ax=axes[i, j], fraction=0.046)
    for g, rgb in enumerate(rgbs):
        axes[g, 3].imshow((rgb.permute(1, 2, 0) + 1) / 2)
        axes[g, 3].set_title(f"adapter RGB {group_tag(g)}", fontsize=9)
    for ax in axes.flat:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=70)
    plt.close(fig)
