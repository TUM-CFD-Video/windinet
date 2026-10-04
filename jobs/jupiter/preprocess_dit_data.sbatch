#!/bin/bash
#SBATCH --job-name=wh_preprocess_dit
#SBATCH --account=e-dev-2026d09-262
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=logs/jupiter/%x_%j.log
#SBATCH --error=logs/jupiter/%x_%j.err

# Encode step of the jupiter DiT pipeline: turns a finetuned VAE checkpoint
# into latents for train_dit.sbatch. scripts/preprocess_dataset.py is a
# single-process encode loop (runs on the first GPU; the job takes the full
# node, 1 node x 4 GPUs, like every other pipeline stage). --time is a guess (sng_pvc needed
# >4h on one XPU tile for 256res; not measured on jupiter yet).
#
# VAE_CHECKPOINT is REQUIRED (env var). $1 (optional) overrides OUTPUT_NAME,
# default = the VAE run directory name; latents land in
# /e/scratch/e-dev-2026d09-262/wh_work/dit_preprocessed/<OUTPUT_NAME>/.
# $2 (optional) overrides DATA_ROOT (default: the train.h5 the VAE run trained
# on, from its training_config.yaml -- 256x256_ds or 512x512_orig).
#
# EVAL_SIMS defaults to 500 = the Chapter 6 VAE held-out split (NOT sng_pvc's
# old 675 default), so DiT val is on the same sims as the VAE's val_vrmse.
# EVAL_SIMS=0 disables the split; MAX_SAMPLES=<n> gives a smoke-test encode.
#
# NUM_SIM_FRAMES: frames kept per sim (e.g. 97 = 8*12+1, no padding). Defaults
# to the VAE run's own data.num_sim_frames, read from the training_config.yaml
# next to its checkpoints/, so the latents always match what the VAE was
# trained on; set it explicitly only to override that.
#
# Submit from the repo root, windinet env active:
#   VAE_CHECKPOINT=/e/scratch/e-dev-2026d09-262/wh_work/finetune_vae_outputs/<run>/checkpoints/vae_shockwave_best.safetensors \
#     sbatch jobs/jupiter/preprocess_dit_data.sbatch [OUTPUT_NAME] [DATA_ROOT]

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
export WINDINET_HF_CACHE=/e/project1/e-dev-2026d09-262/wh_work/ltx_pretrained_wh
export HF_HUB_CACHE="${WINDINET_HF_CACHE}"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

JOB_START_EPOCH=$(date +%s)
trap 'echo "end_time=$(date -Is) elapsed=$(( $(date +%s) - JOB_START_EPOCH ))s"' EXIT
trap 'exit 143' TERM
echo "job=${SLURM_JOB_ID} node=$(hostname) gpus=${CUDA_VISIBLE_DEVICES:-}"
echo "start_time=$(date -Is)"

if [[ -z "${VAE_CHECKPOINT:-}" ]]; then
    echo "error: VAE_CHECKPOINT is required" >&2
    exit 1
fi

DEFAULT_OUTPUT_NAME=$(basename "$(dirname "$(dirname "${VAE_CHECKPOINT}")")")
OUTPUT_NAME=${1:-${DEFAULT_OUTPUT_NAME}}
MAX_SAMPLES=${MAX_SAMPLES:-0}
VAE_TRAINING_CONFIG="$(dirname "$(dirname "${VAE_CHECKPOINT}")")/training_config.yaml"
# Default DATA_ROOT: the train.h5 the VAE itself trained on (its resolved
# data.data_root), so the latents always come from the same resolution.
if [[ -z "${2:-}" ]]; then
    [[ -f "${VAE_TRAINING_CONFIG}" ]] || { echo "error: ${VAE_TRAINING_CONFIG} not found; pass DATA_ROOT as \$2" >&2; exit 1; }
    DATA_ROOT=$("${PYTHON}" -c "import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))['data']['data_root'])" "${VAE_TRAINING_CONFIG}")
else
    DATA_ROOT=$2
fi
if [[ -z "${NUM_SIM_FRAMES:-}" ]]; then
    [[ -f "${VAE_TRAINING_CONFIG}" ]] || { echo "error: ${VAE_TRAINING_CONFIG} not found; set NUM_SIM_FRAMES" >&2; exit 1; }
    NUM_SIM_FRAMES=$("${PYTHON}" -c "import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))['data'].get('num_sim_frames') or '')" "${VAE_TRAINING_CONFIG}")
fi
EVAL_SIMS=${EVAL_SIMS:-500}
SPLIT_SEED=${SPLIT_SEED:-42}
OUT_DIR="/e/scratch/e-dev-2026d09-262/wh_work/dit_preprocessed/${OUTPUT_NAME}"

echo "VAE_CHECKPOINT: ${VAE_CHECKPOINT}"
echo "DATA_ROOT: ${DATA_ROOT}"
echo "OUTPUT_NAME: ${OUTPUT_NAME} -> ${OUT_DIR}"
echo "EVAL_SIMS=${EVAL_SIMS} SPLIT_SEED=${SPLIT_SEED} MAX_SAMPLES=${MAX_SAMPLES} NUM_SIM_FRAMES=${NUM_SIM_FRAMES:-<all>}"

printf '%s\t%s\t%s\t%s\t%s\n' "$(date -Is)" "${SLURM_JOB_ID}" "${SLURM_JOB_NAME}" \
    "${VAE_CHECKPOINT}" "${OUT_DIR}" >> logs/jupiter/INDEX.tsv

EXTRA_ARGS=()
if [[ "${MAX_SAMPLES}" != "0" ]]; then
    EXTRA_ARGS+=(--max-samples "${MAX_SAMPLES}")
elif [[ "${EVAL_SIMS}" != "0" ]]; then
    EXTRA_ARGS+=(--eval-sims "${EVAL_SIMS}" --split-seed "${SPLIT_SEED}")
fi
[[ -n "${NUM_SIM_FRAMES}" ]] && EXTRA_ARGS+=(--num-sim-frames "${NUM_SIM_FRAMES}")

"${PYTHON}" scripts/preprocess_dataset.py \
    "${DATA_ROOT}" \
    --output-dir "${OUT_DIR}" \
    --inflate-checkpoint "${VAE_CHECKPOINT}" \
    "${EXTRA_ARGS[@]}"

# Mirror just the small manifests into the repo checkout (latents stay on scratch).
LOCAL_DIR="finetune_vae_outputs/jupiter/dit_preprocessed/${OUTPUT_NAME}"
mkdir -p "${LOCAL_DIR}"
cp "${OUT_DIR}/normalization.json" "${LOCAL_DIR}/"
[ -f "${OUT_DIR}/split_manifest.json" ] && cp "${OUT_DIR}/split_manifest.json" "${LOCAL_DIR}/"
echo "Mirrored manifests to ${LOCAL_DIR}"
