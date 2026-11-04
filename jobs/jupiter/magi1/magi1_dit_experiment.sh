#!/bin/bash
# A full DiT finetune on any finished VAE, as one Slurm chain: N_DIT x 12 h resuming jobs (Weihao's DiT settings,
# 8000 steps, frames 1-100, 8-bit AdamW; magi1_official_dit.sbatch with overrides) -> evaluation on all 500 test
# sims (benchmark table + 3 gamma videos, sampled 4 chunks at once as MAGI-1 does). Run on the login node from the
# repo root; it only calls sbatch.
#
#   bash jobs/jupiter/magi1/magi1_dit_experiment.sh --name <run> --vae-ckpt <vae_best.safetensors> [--n-dit 5] \
#       [--clean-chunks] [--init-from <ckpt>] [--steps N] [--train-shift S] [--save-every N] [--lr LR] [--dry-run]
#     --clean-chunks  MAGI-1's video-continuation samples in training: some samples start with clean chunks
#                     (no loss on them), as the 4-chunk sampler meets them (dit.sample_num_clean_chunks)
#     --init-from     start from another run's weights (resume.pt or full_stepNNNNNN.safetensors) with a fresh
#                     optimizer and lr schedule, e.g. a continuation of full_8000steps_on_F3
#     --steps         training steps (default 8000); the evaluation uses the last checkpoint
#     --train-shift   sd3 shift of the training timesteps (default 3; 1 = uniform t, more near-clean samples)
#     --save-every    bf16 checkpoint interval (default 1000)
#     --lr            peak learning rate of the cosine schedule (default settings.FULL_FT: 1e-5)
#
# Checkpoints (bf16 every 1000 steps + resume state): $MAGI1_SCRATCH/magi1_dit_<run>/
# Args/logs: $MAGI1_OUTPUTS/dit_experiments/<run>/        results: .../<run>/eval/results.md
# A look at an intermediate checkpoint while training runs (20 sims, ~20 min):
#   sbatch jobs/jupiter/magi1/evaluate_magi1.sbatch --dit-ckpt $MAGI1_SCRATCH/magi1_dit_<run>/full_step001000.safetensors \
#       --vae-ckpt <vae> --frames 100 --metric-sims 20 --out $MAGI1_OUTPUTS/dit_experiments/<run>/eval_step1000 --no-archive
# Cancel: scancel --name=ig_magi1_dit_<run>

set -euo pipefail

NAME=""
VAE_CKPT=""
N_DIT=5
TRAIN_ARGS=()
DRY_RUN=0
STEPS=8000
while [[ $# -gt 0 ]]; do
    case "$1" in
        --name)     NAME="$2"; shift 2 ;;
        --vae-ckpt) VAE_CKPT="$2"; shift 2 ;;
        --n-dit)    N_DIT="$2"; shift 2 ;;
        --clean-chunks) TRAIN_ARGS+=(--clean-chunks); shift ;;
        --init-from) [[ -f "$2" ]] || { echo "--init-from: no file $2" >&2; exit 1; }; TRAIN_ARGS+=(--init-from "$2"); shift 2 ;;
        --steps)      STEPS="$2"; shift 2 ;;
        --train-shift) TRAIN_ARGS+=(--shift "$2"); shift 2 ;;
        --save-every) TRAIN_ARGS+=(--save-every "$2"); shift 2 ;;
        --lr)         TRAIN_ARGS+=(--lr "$2"); shift 2 ;;
        --dry-run)  DRY_RUN=1; shift ;;
        *) echo "unknown option: $1 (see the header of $0)" >&2; exit 1 ;;
    esac
done
[[ -n "${NAME}" && -f "${VAE_CKPT}" ]] || { echo "need --name and an existing --vae-ckpt (see the header of $0)" >&2; exit 1; }
[[ -f "$(dirname "${VAE_CKPT}")/finished.json" ]] || { echo "VAE run in $(dirname "${VAE_CKPT}") has not finished" >&2; exit 1; }

JOBS=jobs/jupiter/magi1
[[ -f "${JOBS}/magi1_env.sh" ]] || { echo "run this from the repo root" >&2; exit 1; }
source "${JOBS}/magi1_paths.sh"
CKPT_DIR="${MAGI1_SCRATCH}/magi1_dit_${NAME}"
OUT="${MAGI1_OUTPUTS}/dit_experiments/${NAME}"

submit() {  # submit <job id to wait for, or ""> <sbatch args...>; prints the new job id
    local dep="$1"; shift
    local cmd=(sbatch --parsable ${dep:+--dependency=afterok:${dep}} "$@")
    if (( DRY_RUN )); then echo "${cmd[*]}" >&2; echo "<job${RANDOM}>"; else "${cmd[@]}"; fi
}

prev=""
for i in $(seq 1 "${N_DIT}"); do
    prev=$(submit "${prev}" --job-name="ig_magi1_dit_${NAME}" "${JOBS}/magi1_official_dit.sbatch" \
        --vae-ckpt "${VAE_CKPT}" --ckpt-dir "${CKPT_DIR}" --out "${OUT}" --steps "${STEPS}" "${TRAIN_ARGS[@]}")
    echo "full DiT ${i}/${N_DIT}: ${prev}"
done
prev=$(submit "${prev}" "${JOBS}/evaluate_magi1.sbatch" --dit-ckpt "${CKPT_DIR}/full_step$(printf %06d "${STEPS}").safetensors" \
    --vae-ckpt "${VAE_CKPT}" --frames 100 --metric-sims 500 --out "${OUT}/eval" --archive-name "dit_${NAME}")
echo "eval        : ${prev}   -> ${OUT}/eval/results.md"
(( DRY_RUN )) && echo "(dry run: nothing submitted)"
exit 0
