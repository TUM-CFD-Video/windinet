#!/bin/bash
#SBATCH --job-name=ig_magi1_official_vae
#SBATCH --account=e-dev-2026d09-262
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --output=/e/scratch/e-dev-2026d09-262/ig_work/logs/%x_%j.log
#SBATCH --error=/e/scratch/e-dev-2026d09-262/ig_work/logs/%x_%j.err

# First official MAGI-1 run, stage 1 (Weihao's best LTX recipe): full VAE finetune, zero-init pressure
# channel, loss RMSE + 50*H1 + 0.01*SDS, 20 epochs, best checkpoint by val VRMSE.
# Stops at --max-minutes and writes resume.pt; resubmitting continues automatically (--resume). Extra
# resubmissions after the run finished just exit. Normally submitted by magi1_pipeline.sh.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source jobs/jupiter/magi1/magi1_env.sh

CKPT_DIR="${MAGI1_OFFICIAL}/vae"
mkdir -p "${CKPT_DIR}"


echo "job=${SLURM_JOB_ID} node=$(hostname) start=$(date -Is)"
launch_4gpu -m magi_finetuning.train.train_vae \
    --train full --init zeros --epochs 20 \
    --ckpt-dir "${CKPT_DIR}" --out "${MAGI1_OUTPUTS}/official/vae" \
    --max-minutes 675 --resume "$@"  # --resume: continue if resume.pt exists
touch_scratch
echo "end=$(date -Is)"
