# CLAUDE.md

Rules for working in this repository with Claude Code. Two people work here: Weihao (LTX-Video VAE and DiT
pipeline, branch `main`) and Lars (Wan 2.1 field adapter, branch `dev-lars/main`). This file is Lars's.

## Ownership and branches

- Lars's parts: `windinet/field_adapter`, `windinet/wan`, `scripts/field_adapter`, `configs/field_adapter`,
  `docs/field_adapter`, `results/field_adapter`, `tests`, `jobs/lumi`, `jobs/jupiter/*_lc.sbatch`, this file.
- Everything else is Weihao's. Do not edit, move or delete his files. If a change there is unavoidable, keep
  it minimal and say so; it will conflict with his next push.
- Never merge into `main`. Merge `origin/main` into `dev-lars/main` regularly; resolve his files toward
  `origin/main`, Lars's toward the branch.
- His output trees (`dit_outputs`, `dit_preprocessed`, `finetune_vae_outputs`, `logs/*` except
  `logs/lumi` and `logs/jupiter`, `figures`, `resource_profile`) stay in git and are hidden from the working
  tree with sparse-checkout (see Setup). Hide, do not delete.

## How to work

- Plan first: read, propose, and ask on every ambiguity with a recommendation and the reason. Do not start
  edits before the plan is agreed.
- No commits, pushes or merges unless asked for that specific one. Commit messages carry no co-author line.
- Never launch GPU jobs on your own. CPU smoke tests are fine.
- Report faithfully: what was verified, what was not.

## Code

- Python 3.11, `ruff` for lint and format (`pyproject.toml`, line length 120). Run `ruff check --fix` and
  `ruff format` on the files you touched, never on Weihao's.
- Small modules with one job each. Configs are pydantic models with `Field(description=...)`, CLIs use typer,
  paths use `pathlib`. Comments say why, not what; one-line docstrings.
- No new dependencies without asking. Tests: `pytest tests`.

## Experiments

- Every run needs `--name` (the results folder) and `--desc` (one sentence on what it tests). The script
  writes `config.yaml`, `metrics.json` with provenance, figures and `adapter.pt`; `vae.pt` is not in git.
- `results/field_adapter/<stage>/README.md` is generated (`scripts/field_adapter/index_runs.py`); do not edit.
- A run folder stays only if a number in `docs/field_adapter/ablations.md` comes from it; every table there
  has a `run` column. Unused runs are deleted.
- Documents are short: numbers in tables, one claim per sentence, no filler.

## Setup on a new machine

```bash
conda activate windinet                      # local: /home/schwollie/anaconda3/envs/windinet
pip install -e . && pip install ruff
git sparse-checkout init --cone            # then hide Weihao's output trees; `git sparse-checkout disable` undoes it
git sparse-checkout set .vscode configs docs euler_mq_dataset jobs pretrained results scripts tests \
    windinet logs/lumi logs/jupiter
```

Clusters: outputs go to `$WINDINET_WORK/finetune_vae_outputs` (`windinet/cluster_config.py`), LTX weights to
`$WINDINET_HF_CACHE`. Lars's launchers set both: `jobs/lumi/finetune_vae_debug.sbatch`,
`jobs/jupiter/finetune_vae_lc.sbatch`. Local GPU is a 2080 Ti (11 GB, fp32 only).
