#!/bin/bash
#SBATCH --job-name=lc_run
#SBATCH --account=project_465003416
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=56
#SBATCH --time=00:30:00
#SBATCH --output=logs/lc/%x_%j.log
# Any command in the windinet container on LUMI, 30 min by default. Always a full node: LUMI bills node hours,
# not GPU hours, so a single-GPU script costs the same as eight. From the repo root:
#   sbatch jobs/lumi/lc/run.sbatch python scripts/field_adapter/train_adapter.py configs/field_adapter/eulermq.yaml \
#       --name <run> --desc "<what it tests>"
#   sbatch --time=02:00:00 --job-name=lc_x jobs/lumi/lc/run.sbatch <command>
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source jobs/lumi/lc/env.sh
srun "${RUN[@]}" "$@"
