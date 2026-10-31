#!/bin/bash
#SBATCH --job-name=ig_magi1_quick_test
#SBATCH --account=e-dev-2026d09-262
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:20:00
#SBATCH --output=/e/scratch/e-dev-2026d09-262/ig_work/logs/%x_%j.log
#SBATCH --error=/e/scratch/e-dev-2026d09-262/ig_work/logs/%x_%j.err

# Short GPU checks on a compute node -- never on the login node (shared, no compute).
# Runs `python <args>` with the MAGI-1 environment, on 1 GPU, for at most 20 minutes.
# Submit from the repo root, e.g.
#   sbatch jobs/jupiter/magi1/quick_test.sbatch -m magi_finetuning.tests.test_attention
#   sbatch jobs/jupiter/magi1/quick_test.sbatch -m magi_finetuning.eval.evaluate --dit-ckpt <ckpt> --metric-sims 1 --num-steps 4 --out <dir>

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source jobs/jupiter/magi1/magi1_env.sh

echo "job=${SLURM_JOB_ID} node=$(hostname) start=$(date -Is) cmd: python $*"
python "$@"
echo "end=$(date -Is)"
