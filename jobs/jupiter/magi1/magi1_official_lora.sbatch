#!/bin/bash
#SBATCH --job-name=ig_magi1_official_lora
#SBATCH --account=e-dev-2026d09-262
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --output=/e/scratch/e-dev-2026d09-262/ig_work/logs/%x_%j.log
#SBATCH --error=/e/scratch/e-dev-2026d09-262/ig_work/logs/%x_%j.err

# LoRA DiT run on the official stage-1 VAE (before the full finetune): rank-32 LoRA + gamma embedding,
# lr 1e-4 (50 warmup steps), effective batch 32, frame 0 given -> frames 1-100 generated (as Weihao), 1500 steps (~12 epochs).
# No resume in the LoRA trainer, so it must fit one 12 h job. LoRA checkpoints (~150 MB) every 250 steps
# go to SCRATCH ($MAGI1_OFFICIAL/lora); logs and args to $MAGI1_OUTPUTS/official/lora. Normally submitted
# by magi1_pipeline.sh.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source jobs/jupiter/magi1/magi1_env.sh

echo "job=${SLURM_JOB_ID} node=$(hostname) start=$(date -Is)"
launch_4gpu -m magi_finetuning.train.finetune \
    --frames 100 --steps 1500 --save-every 250 --log-every 10 \
    --vae-ckpt "${MAGI1_OFFICIAL}/vae/vae_best.safetensors" \
    --ckpt-dir "${MAGI1_OFFICIAL}/lora" --out "${MAGI1_OUTPUTS}/official/lora" "$@"
touch_scratch
echo "end=$(date -Is)"
