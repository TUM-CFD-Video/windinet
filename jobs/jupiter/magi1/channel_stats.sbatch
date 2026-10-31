#!/bin/bash
#SBATCH --job-name=ig_magi1_channel_stats
#SBATCH --account=e-dev-2026d09-262
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=72
#SBATCH --time=02:00:00
#SBATCH --output=/e/scratch/e-dev-2026d09-262/ig_work/logs/%x_%j.log
#SBATCH --error=/e/scratch/e-dev-2026d09-262/ig_work/logs/%x_%j.err

# Normalization statistics of the train split only, incl. log_density / log_pressure (meeting item 9):
# magi_finetuning/utils/channel_stats.py -> configs/magi1/euler_mq_256_train_split_stats.yaml. CPU work (reads
# all of train.h5 with 32 processes); the GPU is only requested because booster nodes come with GPUs.
# Submit from the repo root:  sbatch jobs/jupiter/magi1/channel_stats.sbatch

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source jobs/jupiter/magi1/magi1_env.sh

echo "job=${SLURM_JOB_ID} node=$(hostname) start=$(date -Is)"
python -m magi_finetuning.utils.channel_stats --workers 32 "$@"
echo "end=$(date -Is)"
