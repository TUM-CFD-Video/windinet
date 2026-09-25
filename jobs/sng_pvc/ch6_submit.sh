#!/bin/bash
# One entry point for every remaining Chapter 6 job on sng_pvc (2026-09-25
# plan: SSIM dropped from the baseline, which is now
# finetune_vae_ch6_loss_rmse_h1_256res; see that config and
# finetune_vae_ch6_loss_rmse_h1_ssimfix_256res.yaml's header).
#
# Subcommands (ARM = the part between "finetune_vae_ch6_" and "_256res" in
# the VAE config name; the named sets below expand to lists):
#
#   vae ARM...       VAE finetune only (Groups 1-3 + the fixed-SSIM arm).
#   dit ARM...       encode latents (EVAL_SIMS=500) -> train DiT, chained
#                    with afterok, for arms whose VAE checkpoint ALREADY
#                    exists on scratch (refuses otherwise).
#   pipeline ARM...  VAE finetune -> encode -> train DiT, all chained with
#                    afterok (Group 5: the VAE doesn't exist yet).
#   eval ARM...      eval_dit_vrmse on the arm's LATEST DiT checkpoint
#                    (run after its DiT job has finished).
#   vae128 ARM...    VAE finetune from finetune_vae_ch6_ARM_128res.yaml on
#                    the cluster-default 128x128_ds (resolution comparison
#                    against the 256res arm of the same name).
#
# Named sets:
#   vae_only  = decoder_only_nossim lr_1e4_nossim lr_1e5_nossim
#               color_adapter_nossim loss_rmse_h1_ssimfix
#   g4_ready  = loss_rmse_h1 unfinetuned   (VAE already trained)
#   g5        = kl_1e7 kl_1e6 kl_1e5 sds_stock_w1e4 sds_stock_w1e2 sds_stock_w1
#               (stock SDS teacher only, weights 1e-4/1e-2/1; sds_cfd dropped
#               and the single weight-0.1 sds_stock arm replaced, 2026-09-25)
#
# Typical order:
#   bash jobs/sng_pvc/ch6_submit.sh vae vae_only
#   bash jobs/sng_pvc/ch6_submit.sh dit g4_ready
#   bash jobs/sng_pvc/ch6_submit.sh pipeline g5
#   ...once each DiT job has finished:
#   bash jobs/sng_pvc/ch6_submit.sh eval g4_ready g5
#
# If an upstream job fails, its afterok dependents sit in PENDING with
# reason DependencyNeverSatisfied: scancel them and resubmit that arm.
#
# Why EVAL_SIMS=500: preprocess_dit_data.sbatch defaults to 675 (the
# pre-Chapter-6 split); Chapter 6 VAEs hold out 500, and the DiT val split
# must be the same sims. Why the explicit 256res DATA_ROOT on every VAE job:
# finetune_vae.sbatch falls back to the cluster's 128x128 default without
# it. Why --time=22:00:00 for the SDS arms: the pre-Chapter-6 SDS VAE runs
# took ~16.5h on 3825 train sims (539633-539635); Chapter 6 trains on 4000,
# which leaves no margin under the script's 18h default.
#
# Run from the repo root on sng_pvc (/dss/dsshome1/0D/go76fuz2/windinet).
set -euo pipefail
# Without this, a failing check inside $(submit_vae ...) does not abort the
# subshell and the job is submitted anyway.
shopt -s inherit_errexit

SCRATCH_ROOT="${SCRATCH}/windinet"
DATA_256="${SCRATCH_ROOT}/euler_mq_dataset/256x256_ds/train.h5"
EVAL_SIMS_CH6=500

expand_arms() {
    local a
    for a in "$@"; do
        case "$a" in
            vae_only) echo decoder_only_nossim lr_1e4_nossim lr_1e5_nossim color_adapter_nossim loss_rmse_h1_ssimfix ;;
            g4_ready) echo loss_rmse_h1 unfinetuned ;;
            g5)       echo kl_1e7 kl_1e6 kl_1e5 sds_stock_w1e4 sds_stock_w1e2 sds_stock_w1 ;;
            *)        echo "$a" ;;
        esac
    done
}

vae_config()  { echo "configs/finetune_vae/finetune_vae_ch6_$1_256res.yaml"; }
vae_run()     { echo "finetune_vae_ch6_$1_256res"; }
dit_config()  { echo "configs/dit/train_dit_sng_pvc_ch6_$1.yaml"; }
vae_ckpt() {
    # epochs: 0 (unfinetuned) never sets a best, only the last checkpoint.
    local f=vae_shockwave_best.safetensors
    [ "$1" = unfinetuned ] && f=vae_shockwave_last.safetensors
    echo "${SCRATCH_ROOT}/finetune_vae_outputs_sng_pvc/$(vae_run "$1")/checkpoints/${f}"
}

need_file() {
    [ -e "$1" ] || { echo "error: missing $1" >&2; return 1; }
}

submit_vae() {  # $1=arm, $2=optional dependency; prints job id
    local arm=$1 dep=${2:-} args=(--parsable)
    need_file "$(vae_config "$arm")"
    case "$arm" in sds_*) args+=(--time=22:00:00) ;; esac
    [ -n "$dep" ] && args+=(--dependency="afterok:${dep}")
    sbatch "${args[@]}" jobs/sng_pvc/finetune_vae.sbatch "$(vae_config "$arm")" "$DATA_256"
}

submit_encode_and_dit() {  # $1=arm, $2=optional dependency
    local arm=$1 dep=${2:-} pre_args=(--parsable) pre_id dit_id
    need_file "$(dit_config "$arm")"
    [ -n "$dep" ] && pre_args+=(--dependency="afterok:${dep}")
    pre_id=$(EVAL_SIMS=$EVAL_SIMS_CH6 VAE_CHECKPOINT="$(vae_ckpt "$arm")" \
        sbatch "${pre_args[@]}" jobs/sng_pvc/preprocess_dit_data.sbatch "$(vae_run "$arm")")
    dit_id=$(sbatch --parsable --dependency="afterok:${pre_id}" \
        jobs/sng_pvc/train_dit.sbatch "$(vae_run "$arm")" "$(dit_config "$arm")")
    echo "[$arm] encode=${pre_id} dit=${dit_id}"
}

cmd=${1:-}
[ -n "$cmd" ] && shift || true
arms=$(expand_arms "$@")
[ -n "$arms" ] || { sed -n '2,40p' "$0"; exit 1; }

case "$cmd" in
    vae)
        for arm in $arms; do
            vae_id=$(submit_vae "$arm")
            echo "[$arm] vae=${vae_id}"
        done
        ;;
    dit)
        for arm in $arms; do
            need_file "$(vae_ckpt "$arm")" || { echo "[$arm] skipped (VAE not trained yet? use pipeline)" >&2; continue; }
            submit_encode_and_dit "$arm"
        done
        ;;
    pipeline)
        for arm in $arms; do
            vae_id=$(submit_vae "$arm")
            # Never chain onto an empty id: that would submit the encode
            # with no dependency, i.e. before the VAE exists.
            [[ "$vae_id" =~ ^[0-9]+$ ]] || { echo "[$arm] bad VAE job id '${vae_id}', stopping" >&2; exit 1; }
            echo "[$arm] vae=${vae_id}"
            submit_encode_and_dit "$arm" "$vae_id"
        done
        ;;
    vae128)
        for arm in $arms; do
            cfg="configs/finetune_vae/finetune_vae_ch6_${arm}_128res.yaml"
            need_file "$cfg"
            # No data_root argument: finetune_vae.sbatch then uses
            # CLUSTER_DEFAULTS['sng_pvc']'s 128x128_ds.
            echo "[$arm 128res] vae=$(sbatch --parsable jobs/sng_pvc/finetune_vae.sbatch "$cfg")"
        done
        ;;
    eval)
        for arm in $arms; do
            ckdir="${SCRATCH_ROOT}/outputs/shockwave_dit_ch6_${arm}/checkpoints"
            dit=$(ls -1 "${ckdir}"/model_weights_step_*.safetensors 2>/dev/null | sort -V | tail -1 || true)
            if [ -z "$dit" ]; then
                echo "[$arm] skipped: no DiT checkpoint under ${ckdir}" >&2
                continue
            fi
            pre="${SCRATCH_ROOT}/dit_preprocessed/$(vae_run "$arm")"
            need_file "$pre" && need_file "$(vae_ckpt "$arm")" || { echo "[$arm] skipped" >&2; continue; }
            echo "[$arm] eval on $(basename "$dit")"
            sbatch jobs/sng_pvc/eval_dit_vrmse.sbatch "$pre" "$dit" "$(vae_ckpt "$arm")" "$EVAL_SIMS_CH6"
        done
        ;;
    *)
        sed -n '2,40p' "$0"
        exit 1
        ;;
esac
