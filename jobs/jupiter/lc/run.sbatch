#!/bin/bash
#SBATCH --job-name=lc_run
#SBATCH --account=e-dev-2026d09-262
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=logs/lc/%x_%j.log
# Any command on JUPITER, 30 min by default. Always a full node: JUPITER bills node hours, not GPU hours,
# so a single-GPU script costs the same as four. From the repo root, windinet env active:
#   sbatch jobs/jupiter/lc/run.sbatch python scripts/field_adapter/train_adapter.py configs/field_adapter/eulermq.yaml \
#       --name <run> --desc "<what it tests>"
#   sbatch --time=02:00:00 --job-name=lc_x jobs/jupiter/lc/run.sbatch <command>
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source jobs/jupiter/lc/env.sh
"${RUN[@]}" "$@"
