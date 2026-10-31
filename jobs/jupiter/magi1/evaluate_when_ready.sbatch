#!/bin/bash
#SBATCH --job-name=ig_evaluate_when_ready
#SBATCH --account=e-dev-2026d09-262
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --output=/e/scratch/e-dev-2026d09-262/ig_work/logs/%x_%j.log
#SBATCH --error=/e/scratch/e-dev-2026d09-262/ig_work/logs/%x_%j.err

# Intermediate report: wait until a checkpoint of a running training exists, then evaluate it with
# evaluate_magi1.sbatch (same arguments). For a look at e.g. LoRA step 500 while training continues.
# Submit from the repo root, with a start time so the node isn't held while training is far away:
#   sbatch --begin=2026-10-22T05:30:00 jobs/jupiter/magi1/evaluate_when_ready.sbatch <checkpoint> <evaluate.py args>
# Waits at most 3 h for <checkpoint>, then gives up (exit 1).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
CHECKPOINT="$1"; shift

echo "job=${SLURM_JOB_ID} node=$(hostname) waiting for ${CHECKPOINT} since $(date -Is)"
for minute in $(seq 1 180); do
    [[ -f "${CHECKPOINT}" ]] && break
    sleep 60
done
[[ -f "${CHECKPOINT}" ]] || { echo "gave up: ${CHECKPOINT} did not appear within 3 h"; exit 1; }
sleep 60  # let the trainer finish writing the file
echo "found ${CHECKPOINT} at $(date -Is); evaluating"
bash jobs/jupiter/magi1/evaluate_magi1.sbatch --dit-ckpt "${CHECKPOINT}" "$@"
