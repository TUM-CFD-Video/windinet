#!/bin/bash
#SBATCH --job-name=wh_eval_vae_test
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=general
#SBATCH --account=pn82ku
#SBATCH -o logs/sng_pvc/%j-%x.out
#SBATCH -e logs/sng_pvc/%j-%x.err
# Same broken-scratch-mount node jobs/sng_pvc/latent_stats.sbatch excludes.
#SBATCH --exclude=i20r02c04s06

# Single-tile launcher for scripts/eval_vae_test.py: VAE-only reconstruction
# (pooled VRMSE, chmean and per-channel VRMSE) on the standalone 256x256
# test.h5, the one common held-out set every Chapter 6 VAE arm is compared
# on. Forward-only, plain python, same reasoning as latent_stats.sbatch.
#
# --time=03:00:00 is a generous first guess. The VAE eval pass over 500 val
# sims takes ~30s on 8 tiles (545466), so ~4-5 min on one tile plus the model
# load. Tighten it once the first job's elapsed time is known.
#
# Usage (normally via `bash jobs/sng_pvc/ch6_submit.sh vaetest ...`):
#   sbatch jobs/sng_pvc/eval_vae_test.sbatch <config_yaml> <checkpoint|none> <output_json> [test_h5] [num_samples]

set -eo pipefail

module load micromamba/2.1.1
export PATH="$MICROMAMBA_BASE/bin:$PATH"
eval "$(micromamba shell hook -s bash)"
micromamba activate windinet

cd /dss/dsshome1/0D/go76fuz2/windinet

echo "START TIME: $(date)"
echo "job=${SLURM_JOB_ID} node=$(hostname)"
JOB_START_EPOCH=$(date +%s)
echo "start_time=$(date -Is)"
trap 'echo "end_time=$(date -Is) elapsed=$(( $(date +%s) - JOB_START_EPOCH ))s"' EXIT
trap 'exit 143' TERM
xpu-smi discovery || true

export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HDF5_USE_FILE_LOCKING=FALSE

CONFIG=${1:?config yaml required}
CHECKPOINT=${2:?checkpoint path (or 'none') required}
OUTPUT_JSON=${3:?output json required}
TEST_H5=${4:-"${SCRATCH}/windinet/euler_mq_dataset/256x256_ds/test.h5"}
NUM_SAMPLES=${5:-0}

CKPT_ARGS=()
[ "${CHECKPOINT}" != none ] && CKPT_ARGS=(--checkpoint "${CHECKPOINT}")

echo "config=${CONFIG} checkpoint=${CHECKPOINT} test_h5=${TEST_H5} num_samples=${NUM_SAMPLES} output=${OUTPUT_JSON}"
printf '%s\t%s\t%s\t%s\t%s\n' "$(date -Is)" "${SLURM_JOB_ID}" "${SLURM_JOB_NAME}" \
    "${CHECKPOINT}" "${OUTPUT_JSON}" >> logs/sng_pvc/INDEX.tsv

python scripts/eval_vae_test.py "${CONFIG}" \
    "${CKPT_ARGS[@]}" \
    --test-h5 "${TEST_H5}" \
    --num-samples "${NUM_SAMPLES}" \
    --output "${OUTPUT_JSON}"

echo "END TIME: $(date)"
