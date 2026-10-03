# CLAUDE.md

Rules for working in this repository with Claude Code. Two people work here: Weihao (LTX-Video VAE and DiT
pipeline, branch `main`) and Lars (Wan 2.1 field adapter, branch `dev-lars/main`). This file is Lars's.

## Ownership and branches

- Lars's parts: `windinet/{field_adapter,wan,eulermq,experiment}`, `scripts/wan`, `configs/wan`, `docs/field_adapter`, `docs/wan`,
  `docs/references`, `results/{field_adapter,wan}`, `tests`, `jobs/lc`, `jobs/lumi/lc`, `jobs/jupiter/lc`, this file.
- Layout of Lars's code: `field_adapter` is the field-to-RGB map only; `wan` the model wrappers and the per-stage
  config schema (`vae_stage.py`); `eulermq` data access and splits; `experiment` run bookkeeping
  (config overrides, provenance, index, lr schedule, metrics). One script per stage in `scripts/wan`, one schema per script.
- Weihao's thesis (the LTX baseline we compare against) is `docs/references/weihao_li_master_thesis_2026.pdf`.
- Everything else is Weihao's. Do not edit, move or delete his files. If a change there is unavoidable, keep
  it minimal and say so; it will conflict with his next push.
- Never merge into `main`. Merge `origin/main` into `dev-lars/main` regularly; resolve his files toward
  `origin/main`, Lars's toward the branch.
- His output trees (`dit_outputs`, `dit_preprocessed`, `finetune_vae_outputs`, `logs/*` except
  `logs/lc`, `figures`, `resource_profile`) stay in git and are hidden from the working
  tree with sparse-checkout (see Setup). Hide, do not delete.

## How to work

- Plan first: read, propose, and ask on every ambiguity with a recommendation and the reason. Do not start
  edits before the plan is agreed.
- Never commit on your own. Every change is reviewed by Lars first, and a commit happens only when he says
  so, for that change. Same for pushes and merges. Commit messages carry no co-author line.
- Never launch GPU jobs on your own. CPU smoke tests are fine.
- Report faithfully: what was verified, what was not.

## Code

- Python 3.11, `ruff` for lint and format (`pyproject.toml`, line length 120). Run `ruff check --fix` and
  `ruff format` on the files you touched, never on Weihao's.
- As little code as does the job, fast, and built to extend without rewriting: one implementation per concern,
  one config schema per script, one place per fact. Before adding a file, config or function, look for the one
  that already does it and extend it; a second copy of anything is a bug. Delete what a change makes unused.
- Be a professional SWE: keep good coding practices and well structured code but while keeping code simple and readable with clear non ai like naming no long ai like comments only comments where really necessary short and precise
- Small modules with one job each. Configs are pydantic models with `Field(description=...)`, CLIs use typer,
  paths use `pathlib`. Comments say why, not what; one-line docstrings.
- No new dependencies without asking. Tests: `pytest tests`.

## Experiments

- Every run needs `--name` (the results folder) and `--desc` (one sentence on what it tests). The script
  writes `config.yaml`, `metrics.json` with provenance, figures and `adapter.pt`; `vae.pt` is not in git.
- The `README.md` index of a results folder is generated (`scripts/wan/index_runs.py`); do not edit.
- A run folder stays only if a number in a document under `docs/` comes from it; every table there has a `run`
  column. Unused runs are deleted. New VAE-stage runs go to `results/wan/vae`; `results/field_adapter` is the
  finished adapter study.
- Settings live in the pydantic schema with defaults; a YAML names only what differs, a run variant is a `--set`.
- Documents are short: numbers in tables, one claim per sentence, no filler.

## Setup on a new machine

```bash
conda activate windinet                      # local: /home/schwollie/anaconda3/envs/windinet
pip install -e . && pip install ruff
git sparse-checkout init --cone            # then hide Weihao's output trees; `git sparse-checkout disable` undoes it
# DANGER: sparse-checkout DELETES ignored files (datasets, checkpoints) in any directory that leaves the cone.
# Never change the cone on a checkout that holds local data; keep datasets outside the repo behind a symlink.
git sparse-checkout set .vscode configs docs euler_mq_dataset jobs pretrained results scripts tests \
    windinet logs/lc
```

Clusters: outputs go to `$WINDINET_WORK/finetune_vae_outputs` (`windinet/cluster_config.py`), LTX weights to
`$WINDINET_HF_CACHE`. Lars's launchers (`jobs/lumi/lc`, `jobs/jupiter/lc`) source `env.sh` there, which sets both, and `run.sbatch <command>`
runs any command on a full node (clusters bill node hours, so never request fewer GPUs). Slurm logs go to `logs/lc/`. Local GPU is a 2080 Ti (11 GB, fp32 only).
