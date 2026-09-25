#!/bin/bash
# Chapter 6 baseline on jupiter, chained with Slurm dependencies. Every job is
# capped at the booster partition's 12h limit, so the two long stages are
# split into a first job + resume jobs:
#
#   VAE    finetune_vae_ch6_loss_rmse_h1_256res   4 GPUs  1 + NUM_VAE_RESUMES jobs
#            each resume: afterany on the previous VAE job
#   encode EVAL_SIMS=500                          1 GPU   afterok on the LAST VAE job
#   DiT    train_dit_jupiter_ch6_loss_rmse_h1_30k 4 GPUs  1 + NUM_DIT_RESUMES jobs
#            first: afterok on encode; each resume: afterany on the previous DiT job
#
# Both launchers resume from the newest checkpoint on disk and exit 0 at once
# if the run is already complete, so surplus resume jobs cost ~a minute each.
# The last VAE job only exits 0 once the VAE has really finished (or was
# already finished), so encode never starts on a half-trained VAE.
#
# NUM_VAE_RESUMES (default 1 -> 24h total) / NUM_DIT_RESUMES (default 3 ->
# 48h total for 30k steps): neither throughput is measured on jupiter yet.
# If the chain runs out before a stage finishes, just resubmit that stage's
# sbatch with the same arguments (it continues), then chain the rest by hand.
# SKIP_VAE=1 skips the VAE stage (checkpoint already on scratch).
#
# If a job fails for real, its afterok dependents stay PENDING
# (DependencyNeverSatisfied): scancel them and resubmit from the failed stage.
#
# Run from the repo root on a jupiter login node, windinet env active:
#   bash jobs/jupiter/submit_baseline_pipeline.sh
set -euo pipefail

VAE_CONFIG=configs/finetune_vae/finetune_vae_ch6_loss_rmse_h1_256res.yaml
DIT_CONFIG=configs/dit/train_dit_jupiter_ch6_loss_rmse_h1_30k.yaml
VAE_RUN=finetune_vae_ch6_loss_rmse_h1_256res
VAE_DIR=/e/scratch/e-dev-2026d09-262/wh_work/finetune_vae_outputs/${VAE_RUN}
VAE_CKPT=${VAE_DIR}/checkpoints/vae_shockwave_best.safetensors
NUM_VAE_RESUMES=${NUM_VAE_RESUMES:-1}
NUM_DIT_RESUMES=${NUM_DIT_RESUMES:-3}

mkdir -p logs/jupiter
for f in "$VAE_CONFIG" "$DIT_CONFIG"; do
    [ -e "$f" ] || { echo "error: missing $f" >&2; exit 1; }
done

# submit_chain N DEP SCRIPT ARGS... -- first job afterok:DEP (if DEP set),
# then N resume jobs each afterany on the previous; prints the last job id.
submit_chain() {
    local n=$1 dep=$2 id i args
    shift 2
    args=(--parsable)
    [ -n "$dep" ] && args+=(--dependency="afterok:${dep}")
    id=$(sbatch "${args[@]}" "$@")
    echo "  ${id}" >&2
    for i in $(seq 1 "$n"); do
        id=$(sbatch --parsable --dependency="afterany:${id}" "$@")
        echo "  ${id} (resume ${i})" >&2
    done
    echo "$id"
}

vae_last=""
if [[ "${SKIP_VAE:-0}" == "1" ]]; then
    [ -e "${VAE_DIR}/.train_complete" ] || { echo "error: SKIP_VAE=1 but ${VAE_DIR}/.train_complete does not exist" >&2; exit 1; }
    echo "vae: skipped (${VAE_CKPT})"
else
    echo "vae:"
    vae_last=$(submit_chain "$NUM_VAE_RESUMES" "" jobs/jupiter/finetune_vae.sbatch "$VAE_CONFIG")
fi

pre_args=(--parsable)
[ -n "$vae_last" ] && pre_args+=(--dependency="afterok:${vae_last}")
pre_id=$(EVAL_SIMS=500 VAE_CHECKPOINT="$VAE_CKPT" \
    sbatch "${pre_args[@]}" jobs/jupiter/preprocess_dit_data.sbatch "$VAE_RUN")
echo "encode: ${pre_id}"

echo "dit:"
submit_chain "$NUM_DIT_RESUMES" "$pre_id" jobs/jupiter/train_dit.sbatch "$VAE_RUN" "$DIT_CONFIG" >/dev/null
