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
#   sbatch jobs/lumi/lc/run.sbatch python scripts/wan/train_vae.py configs/wan/vae.yaml \
#       --name <run> --desc "<what it tests>"
#   sbatch --time=02:00:00 --job-name=lc_x jobs/lumi/lc/run.sbatch <command>
#   sbatch jobs/lumi/lc/run.sbatch torchrun --standalone --nproc_per_node=8 scripts/wan/train_vae.py ...   # 8 GPUs, one run
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source jobs/lumi/lc/env.sh
export OMP_NUM_THREADS=$(( SLURM_CPUS_PER_TASK / SLURM_GPUS_ON_NODE ))  # one GPU's share of the cores per process (torchrun would set 1)
# MIOpen's kernel-tuning cache lives in node-local /tmp; staged from scratch and back, so a new node does not retune (~9 min per run).
MIOPEN_SAVED=${WORK}/miopen_cache
mkdir -p "${MIOPEN_USER_DB_PATH}"
[ -d "${MIOPEN_SAVED}" ] && cp -r "${MIOPEN_SAVED}/." "${MIOPEN_USER_DB_PATH}/"
trap 'mkdir -p "${MIOPEN_SAVED}" && cp -r "${MIOPEN_USER_DB_PATH}/." "${MIOPEN_SAVED}/"' EXIT TERM
srun "${RUN[@]}" "$@"
