#!/bin/bash
#SBATCH --job-name=ig_evaluate_magi1
#SBATCH --account=e-dev-2026d09-262
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --output=/e/scratch/e-dev-2026d09-262/ig_work/logs/%x_%j.log
#SBATCH --error=/e/scratch/e-dev-2026d09-262/ig_work/logs/%x_%j.err

# Evaluate a MAGI-1 VAE + DiT pair on test.h5 (magi_finetuning/eval/evaluate.py): one shard per GPU in
# parallel, then one pass merging everything into results.md, CSVs, panels, videos, error_over_time.png.
# Resumable: already-scored sims are skipped, so just resubmit if it runs out of time.
# Timing (measured): ~1 min per sim at 48 frames on 1 GPU; ~3-4 min at 96-100 frames -> 500 sims on 4 GPUs ~6-8 h.
# Submit from the repo root, e.g.
#   sbatch jobs/jupiter/magi1/evaluate_magi1.sbatch --dit-ckpt <ckpt> --vae-ckpt <vae> --frames 100 --out <dir>

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source jobs/jupiter/magi1/magi1_env.sh

NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "job=${SLURM_JOB_ID} node=$(hostname) gpus=${NUM_GPUS} start=$(date -Is)"
for shard in $(seq 0 $((NUM_GPUS - 1))); do
    CUDA_VISIBLE_DEVICES=${shard} python -m magi_finetuning.eval.evaluate "$@" \
        --shard "${shard}" --num-shards "${NUM_GPUS}" > "${MAGI1_LOGS}/eval_${SLURM_JOB_ID}_shard${shard}.log" 2>&1 &
done
wait
python -m magi_finetuning.eval.evaluate "$@" --report-only
echo "end=$(date -Is)"
