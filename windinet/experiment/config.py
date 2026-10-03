"""How every stage script takes its settings and starts: a YAML config plus `--set a.b=value` overrides, parsed into
the script's pydantic schema, then the run folder, the process group and the seeds."""

from __future__ import annotations

import random
from pathlib import Path

import torch
import typer
import yaml

from windinet.experiment.distributed import rank, setup, world
from windinet.experiment.runs import write_config


def load_config(path: Path, overrides: list[str]) -> dict:
    """The YAML as a dict with each dotted override applied; values are parsed as YAML (3e-3, true, [a, b])."""
    raw = yaml.safe_load(path.read_text())
    for item in overrides:
        key, _, value = item.partition("=")
        *parents, leaf = key.split(".")
        node = raw
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(value)
    return raw


def start_run(schema, config_path: Path, name: str | None, desc: str | None, overrides: list[str], overwrite: bool):
    """The config (`schema` over the YAML, --name/--desc and --set), its run folder with config.yaml written first
    (a crashed run still shows what it was), the process group and the seeds. Returns (cfg, run_dir, device)."""
    raw = load_config(config_path, overrides)
    raw |= {k: v for k, v in {"name": name, "description": desc}.items() if v}
    cfg = schema(**raw)
    run_dir = Path(cfg.results_dir) / cfg.name
    if (run_dir / "config.yaml").exists() and not overwrite:
        raise typer.BadParameter(f"{run_dir} exists; pick another --name or pass --overwrite")
    train = getattr(cfg, "train", None)
    if train and train.steps and train.batch_sims % world:
        raise typer.BadParameter(f"train.batch_sims={train.batch_sims} is not a multiple of {world} processes")
    setup()
    seed = (train.seed if train else cfg.seed) + rank
    torch.manual_seed(seed)
    random.seed(seed)  # frame sampling in the main process; loader workers derive theirs from torch
    if rank == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
        write_config(cfg.model_dump(), run_dir / "config.yaml")
        print((run_dir / "config.yaml").read_text())
    return cfg, run_dir, torch.device("cuda" if torch.cuda.is_available() else "cpu")
