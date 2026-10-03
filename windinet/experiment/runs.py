"""Per-run records: what produced a run (provenance) and the index of a results stage.

A run folder holds config.yaml (the resolved config: name, description, every setting; re-runnable as is),
metrics.json (the outcome, plus the provenance block) and the figures and weights. The index README.md
of a stage folder is generated from those files; edit a description in config.yaml, not in the index.
"""

from __future__ import annotations

import json
import platform
import shlex
import string
import subprocess
import sys
import time
from pathlib import Path

import yaml

CHANNELS = ("density", "momentum_x", "momentum_y", "pressure")
SYMBOLS = {"density": "ρ", "momentum_x": "m_x", "momentum_y": "m_y", "pressure": "p"}


def group_tag(i: int) -> str:
    """Image group i as a letter: A, B, C, ... (file names vae_A.pt, metric keys latentA_frechet, labels)."""
    return string.ascii_uppercase[i]


def provenance(started: float) -> dict:
    """Code version, command, library versions, GPU and wall time of the run that is finishing now."""
    import diffusers
    import torch

    from windinet.experiment.distributed import world

    def git(*args: str) -> str | None:
        try:
            return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    return {
        "commit": git("rev-parse", "--short", "HEAD"),
        "dirty": bool(
            git("status", "--porcelain", "--", "windinet", "scripts", "configs")
        ),  # code changed since that commit
        "command": "python " + shlex.join(sys.argv),  # quoted, so it pastes back into a shell
        "python": platform.python_version(),
        "torch": torch.__version__,
        "diffusers": diffusers.__version__,
        "gpu": f"{world} x " + (torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"),
        "started": time.strftime("%Y-%m-%d %H:%M", time.localtime(started)),
        "seconds": round(time.time() - started),
    }


def write_config(cfg: dict, path: Path, note: str | None = None) -> None:
    """The resolved config as YAML, with a header on how to re-run it."""
    head = f"# {cfg['name']}: {cfg.get('description', '')}\n"
    head += f"# Re-run: python {sys.argv[0]} {path} --name <new name>\n"
    if note:
        head += f"# {note}\n"
    path.write_text(head + yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True, width=200))


def settings(cfg: dict) -> str:
    """One phrase per run: what was trained and how."""
    train, data, loss = cfg.get("train", {}), cfg.get("data", {}), cfg.get("loss", {})
    steps, lr = train.get("steps", 0), train.get("lr", 0)
    load = cfg.get("load")
    adapter = f"weights from {Path(load).name}" if load else "adapter untrained"
    if steps and lr > 0:
        adapter += f", lr {lr:g}"
    groups = cfg.get("adapter", {}).get("groups")
    parts = [" + ".join("(" + ", ".join(SYMBOLS.get(f, f) for f in g) + ")" for g in groups)] if groups else []
    parts.append(adapter)
    if cfg.get("vae") == "none":
        parts.append("no VAE")
    elif train.get("vae_parts", "none") != "none":
        parts.append(f"vae {train['vae_parts']} lr {train['vae_lr']:g}")
    parts.append(f"{steps} steps" if steps else "eval only")
    if data.get("clip"):
        parts.append(f"{data['frames_per_sim']}-frame clips")
    if data.get("test_gamma") is not None:
        parts.append(f"gamma {data['test_gamma']:g} held out")
    if loss.get("h1", 1.0) != 1.0:
        parts.append(f"h1 {loss['h1']:g}")
    return ", ".join(parts)


def run_row(run_dir: Path) -> list[str] | None:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return None
    metrics = json.loads(metrics_path.read_text())
    cfg = (
        yaml.safe_load((run_dir / "config.yaml").read_text())
        if (run_dir / "config.yaml").exists()
        else metrics["config"]
    )
    final = metrics["final"]
    vrmse = [final.get(f"vrmse_{c}") for c in CHANNELS]  # None for a field the run's groups drop
    present = [v for v in vrmse if v is not None]
    frechet = " / ".join(f"{v:.2f}" for k, v in final.items() if k.endswith("_frechet")) or "–"
    prov = metrics.get("provenance", {})
    return [
        f"[{run_dir.name}]({run_dir.name}/)",
        cfg.get("description", ""),
        settings(cfg),
        *(f"{v:.3f}" if v is not None else "–" for v in vrmse),
        f"{final.get('vrmse_mean', sum(present) / len(present)):.3f}",
        f"{final['vrmse_clipped_mean']:.3f}" if "vrmse_clipped_mean" in final else "–",
        frechet,
        prov.get("commit", "") + (" (dirty)" if prov.get("dirty") else ""),
    ]


def write_index(stage_dir: Path) -> Path:
    """README.md of a stage: one row per run folder, from config.yaml + metrics.json."""
    rows = [r for d in sorted(stage_dir.iterdir()) if d.is_dir() and (r := run_row(d))]
    head = [
        "run",
        "description",
        "settings",
        "ρ",
        "m_x",
        "m_y",
        "p",
        "mean",
        "mean (5σ clipped)",
        "Fréchet A / B",
        "commit",
    ]
    lines = [
        f"# Runs in `{stage_dir}`",
        "",
        "Generated by `scripts/wan/index_runs.py` from each run's config.yaml and metrics.json; do not edit.",
        "VRMSE per field on the test simulations in physical units (0.05 = 5 % of the field's spread); the clipped",
        "mean is the LTX baseline's metric (values clipped at 5 sigma). Each folder holds config.yaml",
        "(re-runnable), metrics.json, panel.png, curves.png, adapter.pt and, for VAE fine-tunes, vae.pt (not in git).",
        "",
        "| " + " | ".join(head) + " |",
        "|" + "---|" * len(head),
    ]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    path = stage_dir / "README.md"
    path.write_text("\n".join(lines) + "\n")
    return path
