#!/bin/bash
#SBATCH --job-name=ig_symmetry_mean_500
#SBATCH --account=e-dev-2026d09-262
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=01:18:00
#SBATCH --output=/e/scratch/e-dev-2026d09-262/ig_work/logs/%x_%j.log
#SBATCH --error=/e/scratch/e-dev-2026d09-262/ig_work/logs/%x_%j.err

# The +10,000 model on the 500 test sims, 4 steps, shift 5: one sample and the mean over the 8 symmetries of frame 0
# (magi_finetuning/eval/sampling_sweep.py), one shard per GPU. Panels and videos of the low / mid / high gamma cases
# and of sim 0456 come first. Expected about 65 min (about 310 core-h); a run that is cut off continues on resubmission.
# Submit from the repo root: sbatch jobs/jupiter/magi1/symmetry_mean_500.sbatch

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source jobs/jupiter/magi1/magi1_env.sh

RUN="${MAGI1_OUTPUTS}/dit_experiments/full_14000_plus10000_lr5e-6_fast_on_F3"
OUT="${RUN}/eval_symmetry_mean_4steps"
NEW="${MAGI1_SCRATCH}/magi1_dit_full_14000_plus10000_lr5e-6_fast_on_F3/full_step010000.safetensors"
mkdir -p "${OUT}"
NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "job=${SLURM_JOB_ID} node=$(hostname) gpus=${NUM_GPUS} start=$(date -Is)"

for shard in $(seq 0 $((NUM_GPUS - 1))); do
    CUDA_VISIBLE_DEVICES=${shard} python -m magi_finetuning.eval.sampling_sweep --dit-ckpt "${NEW}" --tag new --roundtrip \
        --test-sims-from "${RUN}/eval" --test-variants d4 \
        --video-sims low_gamma=0000_gamma1.1299999952 mid_gamma=0250_gamma1.3999999762 high_gamma=0450_gamma1.7599999905 \
                     0456_gamma1.7599999905 \
        --shard "${shard}" --num-shards "${NUM_GPUS}" --max-minutes 72 --out "${OUT}" \
        > "${MAGI1_LOGS}/symmetry_mean_${SLURM_JOB_ID}_shard${shard}.log" 2>&1 &
done
wait
echo "scoring done=$(date -Is)"
python -m magi_finetuning.eval.sampling_sweep --report-only --out "${OUT}" --test-sims-from "${RUN}/eval"
echo "end=$(date -Is)"
