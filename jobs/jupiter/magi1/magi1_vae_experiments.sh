#!/bin/bash
# VAE experiments against the first official VAE (full finetune, --init zeros, pooled RMSE + 50 H1 + 0.01 SDS,
# 20 epochs). Each changes ONE thing and is named after it. Run on the login node from the repo root; it only
# calls sbatch. Each experiment:
#   VAE x N_VAE (magi1_official_vae.sbatch with overrides, resuming)  ->  VAE-only evaluation on test.h5
#   checkpoints: $MAGI1_SCRATCH/magi1_vae_<name>/      args + results: $MAGI1_OUTPUTS/vae_experiments/<name>/
#   overview of all experiments: $MAGI1_OUTPUTS/vae_experiments/EXPERIMENTS.md
#
#   init_mean      pressure channel starts as the mean of the RGB filters              (done 2026-10-22)
#   log            log(density), log(pressure), train-split stats                        (done 2026-10-22)
#   adapter        frozen RGB encoder + windinet's 4->32->3 / 3->32->4 adapters, no SDS (not run yet)
#   A_io_lr20      20x learning rate for the two 4-channel layers (encoder input, decoder output)
#   B_vrmse_loss   loss = per-field VRMSE (the metric) instead of RMSE pooled over all fields
#   C_no_sds       no SDS term
#   D_init_random  new pressure filters random (RGB filters' spread) instead of zero
#   AB_vrmse_loss_io_lr20  B + A: per-field VRMSE loss and 20x learning rate for the two 4-channel layers
#   E_vrmse_loss_io_lr20_wide16_refine64  AB + a wider decoder output: each pixel from 4 + 16 values instead of 4,
#                  plus a residual 3-layer conv stack (width 64) after the unpatchify (vae.WideDecoder)
#   F1_vrmse_loss_io_lr20_wide16_refine64_sds0         E without SDS
#   F2_vrmse_loss_io_lr20_wide32_refine64_h1-200       E with 32 extra values per pixel and H1 weight 200
#   F3_vrmse_loss_io_lr20_wide16_refine64_40epochs     E trained for 40 epochs instead of 20 (submit with --n-vae 3)
#
# Usage:
#   bash jobs/jupiter/magi1/magi1_vae_experiments.sh --only NAME[,NAME] [--n-vae N] [--max-minutes M] [--no-eval] [--dry-run]
#     --max-minutes M  screening: stop each run after M minutes (same 20-epoch schedule, so the val curve is
#                      comparable step by step with the official run; continue a winner later, it resumes exactly)
#     --no-eval        skip the VAE-only test evaluation (for screening runs)
# e.g. the 2026-10-22 screening:
#   bash jobs/jupiter/magi1/magi1_vae_experiments.sh --only A_io_lr20,B_vrmse_loss,C_no_sds,D_init_random \
#       --n-vae 1 --max-minutes 150 --no-eval
# Cancel: scancel --name=ig_magi1_vae_<name>

set -euo pipefail

ONLY=""
N_VAE=2
MAX_MINUTES=""
EVAL=1
DRY_RUN=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --only)        ONLY="$2"; shift 2 ;;
        --n-vae)       N_VAE="$2"; shift 2 ;;
        --max-minutes) MAX_MINUTES="$2"; shift 2 ;;
        --no-eval)     EVAL=0; shift ;;
        --dry-run)     DRY_RUN=1; shift ;;
        *) echo "unknown option: $1 (see the header of $0)" >&2; exit 1 ;;
    esac
done
[[ -n "${ONLY}" ]] || { echo "say which experiments: --only NAME[,NAME] (see the header of $0)" >&2; exit 1; }

JOBS=jobs/jupiter/magi1
[[ -f "${JOBS}/magi1_env.sh" ]] || { echo "run this from the repo root" >&2; exit 1; }
source "${JOBS}/magi1_paths.sh"   # MAGI1_SCRATCH, MAGI1_OUTPUTS, ... (all on SCRATCH)
NEW_STATS="configs/magi1/euler_mq_256_train_split_stats.yaml"

# Every experiment states its SDS weight explicitly (the official run had 0.01).
declare -A EXTRA_ARGS=(
    [init_mean]="--init mean --sds-weight 0.01"
    [log]="--stats ${NEW_STATS} --log-fields density pressure --sds-weight 0.01"
    [adapter]="--train adapter --sds-weight 0"
    [A_io_lr20]="--io-lr-mult 20 --sds-weight 0.01"
    [B_vrmse_loss]="--recon-loss vrmse_per_channel --sds-weight 0.01"
    [C_no_sds]="--sds-weight 0"
    [D_init_random]="--init random --sds-weight 0.01"
    [AB_vrmse_loss_io_lr20]="--recon-loss vrmse_per_channel --io-lr-mult 20 --sds-weight 0.01"
    [F1_vrmse_loss_io_lr20_wide16_refine64_sds0]="--recon-loss vrmse_per_channel --io-lr-mult 20 --decoder-refine-channels 64 --decoder-extra-channels 16 --sds-weight 0"
    [F2_vrmse_loss_io_lr20_wide32_refine64_h1-200]="--recon-loss vrmse_per_channel --io-lr-mult 20 --decoder-refine-channels 64 --decoder-extra-channels 32 --h1-weight 200 --sds-weight 0.01"
    [F3_vrmse_loss_io_lr20_wide16_refine64_40epochs]="--recon-loss vrmse_per_channel --io-lr-mult 20 --decoder-refine-channels 64 --decoder-extra-channels 16 --sds-weight 0.01 --epochs 40"
    [E_vrmse_loss_io_lr20_wide16_refine64]="--recon-loss vrmse_per_channel --io-lr-mult 20 --decoder-extra-channels 16 --decoder-refine-channels 64 --sds-weight 0.01"
)

# A screening run asks Slurm for its minutes + 30 (setup, saving) instead of 12 h, so it starts sooner.
TIME_ARGS=()
MINUTE_ARGS=()
if [[ -n "${MAX_MINUTES}" ]]; then
    TIME_ARGS=(--time="$(( (MAX_MINUTES + 30) / 60 )):$(printf %02d $(( (MAX_MINUTES + 30) % 60 ))):00")
    MINUTE_ARGS=(--max-minutes "${MAX_MINUTES}")
fi

submit() {  # submit <job id to wait for, or ""> <sbatch args...>; prints the new job id
    local dep="$1"; shift
    local cmd=(sbatch --parsable ${dep:+--dependency=afterok:${dep}} "$@")
    if (( DRY_RUN )); then
        echo "${cmd[*]}" >&2
        echo "<job${RANDOM}>"
    else
        "${cmd[@]}"
    fi
}

stats_job=""
if [[ ",${ONLY}," == *",log,"* && ! -f "${NEW_STATS}" ]]; then
    stats_job=$(submit "" "${JOBS}/channel_stats.sbatch")
    echo "train-split stats : ${stats_job}   -> ${NEW_STATS}"
fi

for name in ${ONLY//,/ }; do
    [[ -n "${EXTRA_ARGS[$name]+set}" ]] || { echo "unknown experiment ${name}" >&2; exit 1; }
    ckpt_dir="${MAGI1_SCRATCH}/magi1_vae_${name}"
    out="${MAGI1_OUTPUTS}/vae_experiments/${name}"
    prev=""
    [[ "${name}" == log ]] && prev="${stats_job}"
    for i in $(seq 1 "${N_VAE}"); do
        # shellcheck disable=SC2086  # EXTRA_ARGS are deliberately split into words
        prev=$(submit "${prev}" --job-name="ig_magi1_vae_${name}" "${TIME_ARGS[@]}" "${JOBS}/magi1_official_vae.sbatch" \
            --ckpt-dir "${ckpt_dir}" --out "${out}" ${EXTRA_ARGS[$name]} "${MINUTE_ARGS[@]}")
        echo "${name} VAE ${i}/${N_VAE}: ${prev}"
    done
    if (( EVAL )); then
        prev=$(submit "${prev}" "${JOBS}/evaluate_magi1.sbatch" --vae-only \
            --vae-ckpt "${ckpt_dir}/vae_best.safetensors" --out "${out}/eval" --archive-name "vae_${name}")
        echo "${name} eval      : ${prev}   -> ${out}/eval/results.md"
    fi
done
(( DRY_RUN )) && echo "(dry run: nothing submitted)"
exit 0
