#!/bin/bash
#SBATCH --job-name=wh_eval_vae_test
#SBATCH --account=e-dev-2026d09-262
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00  # 500 sims took ~8 min on one sng_pvc tile; generous here
#SBATCH --output=logs/jupiter/%x_%j.log
#SBATCH --error=logs/jupiter/%x_%j.err

# jupiter sibling of jobs/sng_pvc/eval_vae_test.sbatch: launcher for
# scripts/eval_vae_test.py (VAE-only reconstruction, pooled/chmean/per-channel
# VRMSE, per-sim values kept) on the standalone 256x256 test.h5 -- the common
# held-out set every Chapter 6 VAE arm is compared on. Forward-only, plain python.
# Allocates the full node (1 node, 4 GPUs) like the other jupiter jobs; the
# eval itself is plain python and runs on the first GPU.
#
# Only <config> is required. By default the checkpoint and output JSON are
# derived from the run that jobs/jupiter/finetune_vae.sbatch trained with that
# config (same patch_config_for_cluster call, so the paths always match):
#   checkpoint: <output_dir>/checkpoints/vae_shockwave_best.safetensors
#   output:     finetune_vae_outputs/jupiter/<run>/test_eval_256.json
# Pass "auto" to keep a default while overriding a later argument, or "none"
# as checkpoint for the freshly-inflated (unfinetuned) reference.
#
# Figure samples go to scratch (wh_work/figure_data/vae_test/<run>), not into
# the repo mirror, same as on sng_pvc.
#
# Submit from the repo root, windinet env active (sbatch inherits it):
#   sbatch jobs/jupiter/eval_vae_test.sbatch <config_yaml> [checkpoint|auto|none] [output_json|auto] [test_h5] [num_samples]
# e.g.
#   sbatch jobs/jupiter/eval_vae_test.sbatch configs/finetune_vae/finetune_vae_ch6_conv_in_only_256res.yaml
# or chained on a training job:
#   sbatch --dependency=afterok:<train_job> jobs/jupiter/eval_vae_test.sbatch <config_yaml>

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
if [[ -f sc_venv_template/activate.sh ]]; then
    set +eu  # modules.sh has a failing "module load mpi4py" and unset vars
    source sc_venv_template/activate.sh
    set -eu
fi
PYTHON=python

export SCRATCH="${SCRATCH_e_dev_2026d09_262}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WINDINET_HF_CACHE=/e/project1/e-dev-2026d09-262/wh_work/ltx_pretrained_wh
export HF_HUB_CACHE="${WINDINET_HF_CACHE}"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"  # windinet is not pip-installed in the venv
export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE

JOB_START_EPOCH=$(date +%s)
trap 'echo "end_time=$(date -Is) elapsed=$(( $(date +%s) - JOB_START_EPOCH ))s"' EXIT
trap 'exit 143' TERM
echo "job=${SLURM_JOB_ID} node=$(hostname) gpus=${CUDA_VISIBLE_DEVICES:-}"
echo "start_time=$(date -Is)"

CONFIG=${1:?config yaml required}
CHECKPOINT=${2:-auto}
OUTPUT_JSON=${3:-auto}
TEST_H5=${4:-/e/project1/e-dev-2026d09-262/datasets/euler_mq_dataset/256x256_ds/test.h5}
NUM_SAMPLES=${5:-0}

# The run's output_dir exactly as finetune_vae.sbatch resolves it.
OUT_DIR=$("${PYTHON}" - "${CONFIG}" <<'EOF'
import sys
import yaml
from windinet.cluster_config import patch_config_for_cluster
cfg = yaml.safe_load(open(sys.argv[1]))
patch_config_for_cluster(cfg, cluster="jupiter", num_processes=4, output_suffix="")
print(cfg["output_dir"])
EOF
)
RUN=$(basename "${OUT_DIR}")
[[ "${CHECKPOINT}" == auto ]] && CHECKPOINT="${OUT_DIR}/checkpoints/vae_shockwave_best.safetensors"
[[ "${OUTPUT_JSON}" == auto ]] && OUTPUT_JSON="finetune_vae_outputs/jupiter/${RUN}/test_eval_256.json"
SAMPLES_DIR="${SCRATCH}/wh_work/figure_data/vae_test/${RUN}"

CKPT_ARGS=()
if [[ "${CHECKPOINT}" != none ]]; then
    [[ -f "${CHECKPOINT}" ]] || { echo "checkpoint not found: ${CHECKPOINT}" >&2; exit 1; }
    CKPT_ARGS=(--checkpoint "${CHECKPOINT}")
fi
mkdir -p "$(dirname "${OUTPUT_JSON}")"

echo "config=${CONFIG} checkpoint=${CHECKPOINT} test_h5=${TEST_H5} num_samples=${NUM_SAMPLES} output=${OUTPUT_JSON} samples_dir=${SAMPLES_DIR}"
nvidia-smi --query-gpu=name,memory.total --format=csv || true
printf '%s\t%s\t%s\t%s\t%s\n' "$(date -Is)" "${SLURM_JOB_ID}" "${SLURM_JOB_NAME}" \
    "${CHECKPOINT}" "${OUTPUT_JSON}" >> logs/jupiter/INDEX.tsv

"${PYTHON}" scripts/eval_vae_test.py "${CONFIG}" \
    ${CKPT_ARGS[@]+"${CKPT_ARGS[@]}"} \
    --test-h5 "${TEST_H5}" \
    --num-samples "${NUM_SAMPLES}" \
    --samples-dir "${SAMPLES_DIR}" \
    --output "${OUTPUT_JSON}"
