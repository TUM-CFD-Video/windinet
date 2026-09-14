#!/bin/bash
#SBATCH --job-name=finetune_decoder_on_rollout
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=12:00:00
#SBATCH --output=logs/lundquist/%x_%j.log
#SBATCH --error=logs/lundquist/%x_%j.log

# Single-GPU launcher for scripts/finetune_decoder_on_rollout.py -- fine-tunes
# ONLY the VAE decoder against the DiT's own real rollout latents (instead
# of the encoder's clean posterior-mean latents it has only ever seen
# before), to test whether this closes any of the gap between DiT latent
# accuracy and decoded pixel accuracy diagnosed by diagnose_frame_drift.py
# (see that script's docstring and EXPERIMENTS.md's "Why does a real ~10x
# latent-space accuracy win only produce a ~7% pixel-space win" open
# question). Output is a drop-in-compatible VAE checkpoint (same format,
# same encoder/adapter weights, same latent-space fingerprint) -- no
# re-encoding or DiT retraining needed to use it.
#
# --time=12:00:00: the original 04:00:00 guess (300 rollout samples at
# ~10.7s/sample per eval_dit_vrmse.py's sng_pvc rate, plus 20 eval rollouts
# plus 15 "cheap" decoder-only epochs) badly undersized a large
# NUM_ROLLOUT_SAMPLES run in practice -- job 22791 (3825 train + 675 val,
# reusing an already-complete phase-1 cache) spent the ENTIRE 4h just
# reloading the 3825 cached train pairs from disk (torch.load in a tight
# loop) and was SLURM-killed before starting the val load or a single
# training epoch. Root cause was likely an unconditional
# torch.cuda.empty_cache() call on every loop iteration in build_dataset()
# (fixed in scripts/finetune_decoder_on_rollout.py -- it now only runs
# after an actual GPU rollout, not on a cache hit), which forced a CUDA
# sync 4500 times for pure disk reads. 12h leaves real headroom for the
# reload plus the 15 training epochs; shrink back down for small
# NUM_ROLLOUT_SAMPLES runs if 12h is wasteful queue time.
#
# RESUMABLE (scripts/finetune_decoder_on_rollout.py): DiT rollouts are cached
# to disk per-sample-id, and the decoder-training loop checkpoints after
# every epoch, so a SLURM time-limit kill loses at most partial progress on
# one id or one epoch, never the whole run. OUT_DIR is therefore keyed by
# RUN_NAME (a stable name), NOT ${SLURM_JOB_ID} -- resuming means submitting
# this EXACT same command again (same positional args, so it computes the
# same RUN_NAME/OUT_DIR) after a kill, or chaining jobs upfront with
# `sbatch --dependency=afterany:<prev_jobid> ...` for a large run that's
# expected to need more than one job. If the run already finished (its
# OUT_DIR has both vae_shockwave_decoder_ft.safetensors and
# finetune_summary.json), the script exits immediately -- harmless to submit
# extra chained jobs "just in case".
#
# Usage:
#   sbatch jobs/lundquist/finetune_decoder_on_rollout.sbatch <PREPROCESSED_DATA_ROOT> <DIT_CHECKPOINT> <VAE_CHECKPOINT> [NUM_ROLLOUT_SAMPLES] [EPOCHS] [NUM_EVAL_SAMPLES] [RUN_NAME]
#
# Same PREPROCESSED_DATA_ROOT/DIT_CHECKPOINT/VAE_CHECKPOINT conventions as
# jobs/lundquist/eval_dit_vrmse.sbatch. NUM_ROLLOUT_SAMPLES defaults to 300,
# EPOCHS to 15, NUM_EVAL_SAMPLES to 20 (script's own default -- bump to 675
# for the full val split, same convention as eval_dit_vrmse.sbatch's own
# NUM_SAMPLES=675 usage; remember to also scale --time, since eval rollouts
# cost the same per-sample as training rollouts). RUN_NAME defaults to the
# VAE checkpoint's own arm directory name (e.g.
# "finetune_vae_whole_structure_baseline_ep20_sds_untrained") -- override it
# if running the same arm twice with different sample counts and you don't
# want them to collide/resume into each other's OUT_DIR.
#
# Example -- chaining 4 jobs of 12h each upfront for a large (all-data) run,
# each depending on the previous one finishing (successfully or by being
# killed -- afterany, not afterok):
#   RUN=sds_untrained_full
#   J1=$(sbatch --time=12:00:00 --parsable jobs/lundquist/finetune_decoder_on_rollout.sbatch \
#     dit_preprocessed/finetune_vae_whole_structure_baseline_ep20_sds_untrained \
#     dit_outputs/lundquist/dit_sds_untrained/checkpoints/model_weights_step_08033.safetensors \
#     finetune_vae_outputs/lundquist/finetune_vae_whole_structure_baseline_ep20_sds_untrained/checkpoints/vae_shockwave_best.safetensors \
#     3825 15 675 "${RUN}")
#   J2=$(sbatch --time=12:00:00 --parsable --dependency=afterany:$J1 jobs/lundquist/finetune_decoder_on_rollout.sbatch \
#     dit_preprocessed/finetune_vae_whole_structure_baseline_ep20_sds_untrained \
#     dit_outputs/lundquist/dit_sds_untrained/checkpoints/model_weights_step_08033.safetensors \
#     finetune_vae_outputs/lundquist/finetune_vae_whole_structure_baseline_ep20_sds_untrained/checkpoints/vae_shockwave_best.safetensors \
#     3825 15 675 "${RUN}")
#   # ... repeat --dependency=afterany:$J2 -> J3, afterany:$J3 -> J4, etc.
#
# Example (single job, small/default run):
#   sbatch jobs/lundquist/finetune_decoder_on_rollout.sbatch \
#     dit_preprocessed/finetune_vae_whole_structure_baseline_ep20_sds_untrained \
#     dit_outputs/lundquist/dit_sds_untrained/checkpoints/model_weights_step_08033.safetensors \
#     finetune_vae_outputs/lundquist/finetune_vae_whole_structure_baseline_ep20_sds_untrained/checkpoints/vae_shockwave_best.safetensors

set -euo pipefail

PYTHON=/local/disk/hramachandran/miniconda/envs/windinet/bin/python
cd /local/disk/hramachandran/work/wh_work/windinet

export HF_HUB_CACHE="${PWD}/pretrained/hub"

if [ -z "${1:-}" ] || [ -z "${2:-}" ] || [ -z "${3:-}" ]; then
    echo "error: PREPROCESSED_DATA_ROOT, DIT_CHECKPOINT, and VAE_CHECKPOINT are all required" >&2
    echo "  usage: sbatch jobs/lundquist/finetune_decoder_on_rollout.sbatch <PREPROCESSED_DATA_ROOT> <DIT_CHECKPOINT> <VAE_CHECKPOINT> [NUM_ROLLOUT_SAMPLES] [EPOCHS] [NUM_EVAL_SAMPLES] [RUN_NAME]" >&2
    exit 1
fi
PREPROCESSED_DATA_ROOT=$1
DIT_CHECKPOINT=$2
VAE_CHECKPOINT=$3
NUM_ROLLOUT_SAMPLES=${4:-300}
EPOCHS=${5:-15}
NUM_EVAL_SAMPLES=${6:-20}
# Stable across resubmissions (UNLIKE ${SLURM_JOB_ID}) -- this is what makes
# resuming/chaining work: every job in a chain must compute the SAME
# OUT_DIR/CACHE_DIR to find the previous job's cache and training_state.pt.
# Default derives from the VAE arm's own directory name; pass an explicit
# 7th arg to avoid a collision if running the same arm twice with different
# sample counts.
RUN_NAME=${7:-$(basename "$(dirname "$(dirname "${VAE_CHECKPOINT}")")")}

NORMALIZATION=$(dirname "$(dirname "${VAE_CHECKPOINT}")")/training_config.yaml
OUT_DIR="logs/lundquist/finetune_decoder_on_rollout/${RUN_NAME}"
CACHE_DIR="${OUT_DIR}/rollout_cache"
SCALAR_CHECKPOINT=${DIT_CHECKPOINT/model_weights_/scalar_embedding_}

echo "job=${SLURM_JOB_ID} node=$(hostname) gpus=${CUDA_VISIBLE_DEVICES}"
JOB_START_EPOCH=$(date +%s)
echo "start_time=$(date -Is)"
trap 'echo "end_time=$(date -Is) elapsed=$(( $(date +%s) - JOB_START_EPOCH ))s"' EXIT
trap 'exit 143' TERM
echo "PREPROCESSED_DATA_ROOT: ${PREPROCESSED_DATA_ROOT}"
echo "DIT_CHECKPOINT: ${DIT_CHECKPOINT}"
echo "SCALAR_CHECKPOINT: ${SCALAR_CHECKPOINT}"
echo "VAE_CHECKPOINT: ${VAE_CHECKPOINT}"
echo "NORMALIZATION: ${NORMALIZATION}"
echo "NUM_ROLLOUT_SAMPLES: ${NUM_ROLLOUT_SAMPLES}"
echo "EPOCHS: ${EPOCHS}"
echo "NUM_EVAL_SAMPLES: ${NUM_EVAL_SAMPLES}"
echo "RUN_NAME: ${RUN_NAME}"
echo "OUT_DIR: ${OUT_DIR}"

"${PYTHON}" scripts/finetune_decoder_on_rollout.py configs/dit/inference_dit.yaml \
    --preprocessed_data_root "${PREPROCESSED_DATA_ROOT}" \
    --checkpoint "${DIT_CHECKPOINT}" \
    --scalar_checkpoint "${SCALAR_CHECKPOINT}" \
    --vae_checkpoint "${VAE_CHECKPOINT}" \
    --cache_dir "${CACHE_DIR}" \
    --normalization "${NORMALIZATION}" \
    --num_rollout_samples "${NUM_ROLLOUT_SAMPLES}" \
    --epochs "${EPOCHS}" \
    --num_eval_samples "${NUM_EVAL_SAMPLES}" \
    --out_dir "${OUT_DIR}"
