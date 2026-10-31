#!/bin/bash
#SBATCH --job-name=ig_magi1_official_dit
#SBATCH --account=e-dev-2026d09-262
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --output=/e/scratch/e-dev-2026d09-262/ig_work/logs/%x_%j.log
#SBATCH --error=/e/scratch/e-dev-2026d09-262/ig_work/logs/%x_%j.err

# First official MAGI-1 run, stage 2: full DiT finetune on the stage-1 VAE's latents, Weihao's DiT
# settings (lr 1e-5 cosine -> 1e-8, wd 0.01, batch 32, 8000 steps, seed 42; 8-bit AdamW), frame 0 -> frames 1-100.
# 8000 steps take several 12 h jobs: each stops at --max-minutes and writes resume.pt, the next one
# continues. bf16 weights every 1000 steps (full_stepXXXXXX.safetensors). Normally submitted by
# magi1_pipeline.sh.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source jobs/jupiter/magi1/magi1_env.sh

CKPT_DIR="${MAGI1_OFFICIAL}/dit"
mkdir -p "${CKPT_DIR}"


echo "job=${SLURM_JOB_ID} node=$(hostname) start=$(date -Is)"
launch_4gpu -m magi_finetuning.train.finetune \
    --full-finetune --frames 100 --steps 8000 --save-every 1000 \
    --vae-ckpt "${MAGI1_OFFICIAL}/vae/vae_best.safetensors" \
    --ckpt-dir "${CKPT_DIR}" --out "${MAGI1_OUTPUTS}/official/dit" \
    --max-minutes 675 --resume "$@"  # --resume: continue if resume.pt exists
touch_scratch
echo "end=$(date -Is)"
