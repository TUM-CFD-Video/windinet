"""Weight files of a run folder. Big files (vae.pt, dit.pt, cond.pt) go to `weights_dir` behind a symlink, so the
repo keeps only the small adapter.pt; `load_weights` is the inverse of `save_weights`."""

from __future__ import annotations

from pathlib import Path

import torch

from windinet.field_adapter import GroupedAdapter, group_tag


def big_file(run_dir: Path, name: str, weights_dir: str | None = None) -> Path:
    """Where a large output of the run goes: <run_dir>/<name>, or with `weights_dir` the same path under weights_dir
    (the results tree mirrored, so nested stages of different runs cannot collide), symlinked from the run."""
    rel = run_dir.relative_to("results") if run_dir.parts[:1] == ("results",) else Path(run_dir.name)
    path = Path(weights_dir) / rel / name if weights_dir else run_dir / name
    if weights_dir:
        path.parent.mkdir(parents=True, exist_ok=True)
        (run_dir / name).unlink(missing_ok=True)
        (run_dir / name).symlink_to(path)
    return path


def save_state(run_dir: Path, name: str, state: dict, weights_dir: str | None = None) -> None:
    torch.save(state, big_file(run_dir, name, weights_dir))


def vae_file(i: int, n_vaes: int) -> str:
    return f"vae_{group_tag(i)}.pt" if n_vaes > 1 else "vae.pt"


def save_weights(run_dir: Path, adapter: GroupedAdapter, vaes: list, weights_dir: str | None) -> None:
    """adapter.pt and, per fine-tuned VAE, its trainable tensors as vae.pt or vae_<group>.pt."""
    save_state(run_dir, "adapter.pt", adapter.state_dict())
    for i, vae in enumerate(vaes):
        if vae.trainable_params:
            state = {k: v.detach().cpu() for k, v in vae.vae.state_dict(keep_vars=True).items() if v.requires_grad}
            save_state(run_dir, vae_file(i, len(vaes)), state, weights_dir)


def load_weights(run: Path, adapter: GroupedAdapter | None, vaes: list, device: torch.device) -> None:
    """adapter.pt unless `adapter` is None; a VAE takes its own vae_<group>.pt, else the run's shared vae.pt, else
    stays pretrained."""
    if adapter is not None:
        adapter.load_state_dict(torch.load(run / "adapter.pt", map_location=device))
    for i, vae in enumerate(vaes):
        for name in (vae_file(i, len(vaes)), "vae.pt"):
            if (run / name).exists():
                vae.vae.load_state_dict(torch.load(run / name, map_location=device), strict=False)
                break
