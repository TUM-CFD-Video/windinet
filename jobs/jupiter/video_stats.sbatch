#!/bin/bash
#SBATCH --job-name=wh_video_stats
#SBATCH --account=e-dev-2026d09-262
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00  # one streaming pass over 256x256_ds/train.h5; not timed on jupiter yet
#SBATCH --output=logs/jupiter/%x_%j.log
#SBATCH --error=logs/jupiter/%x_%j.err

# Shared channel statistics of the rollout videos (docs/video_spec.md section 2):
# mean, std, min, max, 0.5th / 99.5th percentile of the training ground truth in
# physical units, plus unit and signed (momentum). CPU only; booster is the only
# partition this project uses, so the job takes a node like every other stage.
# One pass: the histogram range for the percentiles comes from the existing
# euler_mq_256_only_train.yaml (same file's min/max; checked after the pass).
#
# Writes configs/video/euler_mq_256_video_stats.json in this checkout: commit it,
# it is the file every collaborator renders with.
#
#   sbatch jobs/jupiter/video_stats.sbatch [TRAIN_H5] [OUT_JSON] [RANGE_YAML]

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
if [[ -f sc_venv_template/activate.sh ]]; then
    set +eu  # modules.sh has a failing "module load mpi4py" and unset vars
    source sc_venv_template/activate.sh
    set -eu
fi
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE

JOB_START_EPOCH=$(date +%s)
trap 'echo "end_time=$(date -Is) elapsed=$(( $(date +%s) - JOB_START_EPOCH ))s"' EXIT
trap 'exit 143' TERM
echo "job=${SLURM_JOB_ID} node=$(hostname) start_time=$(date -Is)"

TRAIN_H5=${1:-/e/project1/e-dev-2026d09-262/datasets/euler_mq_dataset/256x256_ds/train.h5}
OUT_JSON=${2:-configs/video/euler_mq_256_video_stats.json}
RANGE_YAML=${3:-configs/finetune_vae/euler_mq_256_only_train.yaml}
echo "TRAIN_H5=${TRAIN_H5} OUT_JSON=${OUT_JSON} RANGE_YAML=${RANGE_YAML}"

python scripts/compute_channel_stats.py "${TRAIN_H5}" \
    --video-stats "${OUT_JSON}" \
    --hist-range-from "${RANGE_YAML}" \
    --compare-to "${RANGE_YAML}"
