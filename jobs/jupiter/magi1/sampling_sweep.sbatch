#!/bin/bash
#SBATCH --job-name=ig_sampling_sweep
#SBATCH --account=e-dev-2026d09-262
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=00:24:00
#SBATCH --output=/e/scratch/e-dev-2026d09-262/ig_work/logs/%x_%j.log
#SBATCH --error=/e/scratch/e-dev-2026d09-262/ig_work/logs/%x_%j.err

# One packed job (magi_finetuning/eval/sampling_sweep.py), 4 steps, shift 5:
#   GPU 0     the 8000-step model on 60 train + 60 val sims (the reference for the change per sim)
#   GPU 1..3  the +10,000 model: the 31 test sims with the symmetry / seed / step variants, then 60 train + 60 val sims
# Measured (job 2889885): 16 min 50 s, 81 core-h; every process stops starting sims after 20 min so the report still runs.
# Submit from the repo root: sbatch jobs/jupiter/magi1/sampling_sweep.sbatch

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source jobs/jupiter/magi1/magi1_env.sh

OUT="${MAGI1_OUTPUTS}/diagnostics/sampling_sweep/run10k_step10000"
NEW="${MAGI1_SCRATCH}/magi1_dit_full_14000_plus10000_lr5e-6_fast_on_F3/full_step010000.safetensors"
OLD="${MAGI1_SCRATCH}/magi1_dit_full_8000steps_on_F3/full_step008000.safetensors"
REFERENCE="${MAGI1_OUTPUTS}/diagnostics/run10k_checkpoints/r10k_step10000"   # the 31 test sims, same sampler
OLD_TEST="${MAGI1_OUTPUTS}/diagnostics/steps_ablation/steps4_shift5"         # the 8000-step model on them
mkdir -p "${OUT}"
echo "job=${SLURM_JOB_ID} node=$(hostname) start=$(date -Is)"

CUDA_VISIBLE_DEVICES=0 python -m magi_finetuning.eval.sampling_sweep --dit-ckpt "${OLD}" --tag old --split-sims 60 \
    --max-minutes 20 --out "${OUT}" > "${MAGI1_LOGS}/sampling_sweep_${SLURM_JOB_ID}_old.log" 2>&1 &
for shard in 0 1 2; do
    CUDA_VISIBLE_DEVICES=$((shard + 1)) python -m magi_finetuning.eval.sampling_sweep --dit-ckpt "${NEW}" --tag new \
        --split-sims 60 --roundtrip --test-sims-from "${REFERENCE}" --test-variants d4,seeds,steps --save-sims 0456_gamma1.7599999905 \
        --shard "${shard}" --num-shards 3 --max-minutes 20 --out "${OUT}" \
        > "${MAGI1_LOGS}/sampling_sweep_${SLURM_JOB_ID}_new${shard}.log" 2>&1 &
done
wait
echo "scoring done=$(date -Is)"
python -m magi_finetuning.eval.sampling_sweep --report-only --out "${OUT}" --test-sims-from "${REFERENCE}" \
    --old-test-scores "${OLD_TEST}"
echo "end=$(date -Is)"
