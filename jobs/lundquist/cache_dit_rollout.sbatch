#!/bin/bash
#SBATCH --job-name=cache_dit_rollout
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=12:00:00
#SBATCH --output=logs/lundquist/%x_%j.log
#SBATCH --error=logs/lundquist/%x_%j.log

# Phase-1-ONLY launcher for scripts/finetune_decoder_on_rollout.py --cache_only:
# rolls out the frozen DiT on NUM_ROLLOUT_SAMPLES train-split + NUM_EVAL_SAMPLES
# val-split sims and caches each (pred_latent, target) pair to disk, then exits
# -- no decoder loaded, no training. Split out from
# jobs/lundquist/finetune_decoder_on_rollout.sbatch so the expensive,
# DiT-forward-heavy rollout phase (full dataset: ~4500 samples x ~10.7s/sample
# ~= 13h, i.e. already past this cluster's 12h debug-partition hard cap on its
# own) can be its own job/job-chain, separate from the decoder-training phase
# (cheap per step, no DiT forward pass) -- the training job then just loads
# this finished cache and trains, with a much smaller/more predictable --time.
#
# RUN_NAME/OUT_DIR/CACHE_DIR conventions are IDENTICAL to
# jobs/lundquist/finetune_decoder_on_rollout.sbatch (same default derivation
# from VAE_CHECKPOINT's arm directory name) -- run this first with a given
# RUN_NAME, then run that OTHER script with the SAME RUN_NAME to train against
# the cache this one built. Resumable the same way: rerunning this exact
# command (or chaining with --dependency=afterany:<prev_jobid>) only rolls out
# ids not already cached.
#
# Usage:
#   sbatch jobs/lundquist/cache_dit_rollout.sbatch <PREPROCESSED_DATA_ROOT> <DIT_CHECKPOINT> <VAE_CHECKPOINT> [NUM_ROLLOUT_SAMPLES] [NUM_EVAL_SAMPLES] [RUN_NAME]
#
# Example -- chain 2 jobs of 12h each for the full dataset (3825 train + 675
# val ~= 13h estimated, so 1 job likely isn't quite enough):
#   RUN=sds_untrained_full
#   J1=$(sbatch --parsable jobs/lundquist/cache_dit_rollout.sbatch \
#     dit_preprocessed/finetune_vae_whole_structure_baseline_ep20_sds_untrained \
#     dit_outputs/lundquist/dit_sds_untrained/checkpoints/model_weights_step_08033.safetensors \
#     finetune_vae_outputs/lundquist/finetune_vae_whole_structure_baseline_ep20_sds_untrained/checkpoints/vae_shockwave_best.safetensors \
#     3825 675 "${RUN}")
#   sbatch --dependency=afterany:$J1 jobs/lundquist/cache_dit_rollout.sbatch \
#     dit_preprocessed/finetune_vae_whole_structure_baseline_ep20_sds_untrained \
#     dit_outputs/lundquist/dit_sds_untrained/checkpoints/model_weights_step_08033.safetensors \
#     finetune_vae_outputs/lundquist/finetune_vae_whole_structure_baseline_ep20_sds_untrained/checkpoints/vae_shockwave_best.safetensors \
#     3825 675 "${RUN}"
#   # then, once caching is confirmed done, train (separate script/job):
#   sbatch --time=04:00:00 jobs/lundquist/finetune_decoder_on_rollout.sbatch \
#     dit_preprocessed/finetune_vae_whole_structure_baseline_ep20_sds_untrained \
#     dit_outputs/lundquist/dit_sds_untrained/checkpoints/model_weights_step_08033.safetensors \
#     finetune_vae_outputs/lundquist/finetune_vae_whole_structure_baseline_ep20_sds_untrained/checkpoints/vae_shockwave_best.safetensors \
#     3825 15 675 "${RUN}"

set -euo pipefail

PYTHON=/local/disk/hramachandran/miniconda/envs/windinet/bin/python
cd /local/disk/hramachandran/work/wh_work/windinet

export HF_HUB_CACHE="${PWD}/pretrained/hub"

if [ -z "${1:-}" ] || [ -z "${2:-}" ] || [ -z "${3:-}" ]; then
    echo "error: PREPROCESSED_DATA_ROOT, DIT_CHECKPOINT, and VAE_CHECKPOINT are all required" >&2
    echo "  usage: sbatch jobs/lundquist/cache_dit_rollout.sbatch <PREPROCESSED_DATA_ROOT> <DIT_CHECKPOINT> <VAE_CHECKPOINT> [NUM_ROLLOUT_SAMPLES] [NUM_EVAL_SAMPLES] [RUN_NAME]" >&2
    exit 1
fi
PREPROCESSED_DATA_ROOT=$1
DIT_CHECKPOINT=$2
VAE_CHECKPOINT=$3
NUM_ROLLOUT_SAMPLES=${4:-300}
NUM_EVAL_SAMPLES=${5:-20}
# MUST match jobs/lundquist/finetune_decoder_on_rollout.sbatch's derivation
# exactly -- this is how the later training job finds this job's cache.
RUN_NAME=${6:-$(basename "$(dirname "$(dirname "${VAE_CHECKPOINT}")")")}

NORMALIZATION=$(dirname "$(dirname "${VAE_CHECKPOINT}")")/training_config.yaml
OUT_DIR="logs/lundquist/finetune_decoder_on_rollout/${RUN_NAME}"
CACHE_DIR="${OUT_DIR}/rollout_cache"
SCALAR_CHECKPOINT=${DIT_CHECKPOINT/model_weights_/scalar_embedding_}

echo "job=${SLURM_JOB_ID} node=$(hostname) gpus=${CUDA_VISIBLE_DEVICES}"
echo "PREPROCESSED_DATA_ROOT: ${PREPROCESSED_DATA_ROOT}"
echo "DIT_CHECKPOINT: ${DIT_CHECKPOINT}"
echo "SCALAR_CHECKPOINT: ${SCALAR_CHECKPOINT}"
echo "VAE_CHECKPOINT: ${VAE_CHECKPOINT}"
echo "NORMALIZATION: ${NORMALIZATION}"
echo "NUM_ROLLOUT_SAMPLES: ${NUM_ROLLOUT_SAMPLES}"
echo "NUM_EVAL_SAMPLES: ${NUM_EVAL_SAMPLES}"
echo "RUN_NAME: ${RUN_NAME}"
echo "OUT_DIR: ${OUT_DIR}"
echo "CACHE_DIR: ${CACHE_DIR}"

"${PYTHON}" scripts/finetune_decoder_on_rollout.py configs/dit/inference_dit.yaml \
    --preprocessed_data_root "${PREPROCESSED_DATA_ROOT}" \
    --checkpoint "${DIT_CHECKPOINT}" \
    --scalar_checkpoint "${SCALAR_CHECKPOINT}" \
    --vae_checkpoint "${VAE_CHECKPOINT}" \
    --cache_dir "${CACHE_DIR}" \
    --normalization "${NORMALIZATION}" \
    --num_rollout_samples "${NUM_ROLLOUT_SAMPLES}" \
    --num_eval_samples "${NUM_EVAL_SAMPLES}" \
    --out_dir "${OUT_DIR}" \
    --cache_only
