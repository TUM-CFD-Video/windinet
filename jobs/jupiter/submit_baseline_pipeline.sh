#!/bin/bash
# Jupiter baseline, end to end: one sbatch per stage, every stage on 1 node x
# 4 GPUs, chained with afterok dependencies. Thesis baseline recipe (VAE
# "Full FT" + 8k-step DiT, THESIS_RESULTS.md) on 97-frame sims, the project
# standard from 2026-10-04 (the thesis used 101, padded to 105), so its test
# numbers are the new reference rather than directly comparable to the thesis.
# The 97 comes from the VAE config's data.num_sim_frames; the encode stage
# reads it from the VAE run, and the DiT and its eval follow the latents.
#
#   VAE      finetune_vae.sbatch         finetune_vae_jupiter_baseline_256res
#   vaetest  eval_vae_test.sbatch        afterok VAE   (test.h5, 500 sims)
#   encode   preprocess_dit_data.sbatch  afterok VAE   (EVAL_SIMS=500)
#   DiT      train_dit.sbatch            afterok encode (train_dit_jupiter_baseline, 8k steps)
#   diteval  eval_dit_vrmse.sbatch       afterok DiT   (test.h5, 500 sims)
#
# Each training stage must finish inside its own job (VAE ~3.5h of 6h; DiT
# ~6.7h of the 12h booster limit at the ~0.33 steps/s measured by the 30k run,
# jobs 2028915-17, on 101-frame latents). If a training job hits its walltime, its dependents stay
# PENDING (DependencyNeverSatisfied): resubmit that stage's sbatch with the
# same arguments -- both launchers resume from the newest checkpoint -- then
# scancel the stale dependents and chain the rest by hand.
# SKIP_VAE=1 skips the VAE and vaetest stages (checkpoint already on scratch).
#
# Run from the repo root on a jupiter login node, windinet env active:
#   bash jobs/jupiter/submit_baseline_pipeline.sh
set -euo pipefail

VAE_CONFIG=configs/finetune_vae/finetune_vae_jupiter_baseline_256res.yaml
DIT_CONFIG=configs/dit/train_dit_jupiter_baseline.yaml
VAE_RUN=finetune_vae_jupiter_baseline_256res
VAE_DIR=/e/scratch/e-dev-2026d09-262/wh_work/finetune_vae_outputs/${VAE_RUN}
VAE_CKPT=${VAE_DIR}/checkpoints/vae_shockwave_best.safetensors
PRE_ROOT=/e/scratch/e-dev-2026d09-262/wh_work/dit_preprocessed/${VAE_RUN}
DIT_CKPT_DIR=/e/scratch/e-dev-2026d09-262/wh_work/dit_outputs/shockwave_dit_jupiter_baseline/checkpoints

mkdir -p logs/jupiter
for f in "$VAE_CONFIG" "$DIT_CONFIG"; do
    [ -e "$f" ] || { echo "error: missing $f" >&2; exit 1; }
done
grep -q "shockwave_dit_jupiter_baseline\"" "$DIT_CONFIG" || { echo "error: $DIT_CONFIG output_dir does not match DIT_CKPT_DIR" >&2; exit 1; }

pre_dep=()
if [[ "${SKIP_VAE:-0}" == "1" ]]; then
    [ -e "${VAE_DIR}/.train_complete" ] || { echo "error: SKIP_VAE=1 but ${VAE_DIR}/.train_complete does not exist" >&2; exit 1; }
    echo "vae:     skipped (${VAE_CKPT})"
else
    [ -e "${VAE_DIR}/.train_complete" ] && { echo "error: ${VAE_DIR} is already complete; use SKIP_VAE=1" >&2; exit 1; }
    vae_id=$(sbatch --parsable jobs/jupiter/finetune_vae.sbatch "$VAE_CONFIG")
    echo "vae:     ${vae_id}"
    echo "vaetest: $(sbatch --parsable --dependency="afterok:${vae_id}" jobs/jupiter/eval_vae_test.sbatch "$VAE_CONFIG")"
    pre_dep=(--dependency="afterok:${vae_id}")
fi

pre_id=$(EVAL_SIMS=500 VAE_CHECKPOINT="$VAE_CKPT" \
    sbatch --parsable ${pre_dep[@]+"${pre_dep[@]}"} jobs/jupiter/preprocess_dit_data.sbatch "$VAE_RUN")
echo "encode:  ${pre_id}"

dit_id=$(sbatch --parsable --dependency="afterok:${pre_id}" jobs/jupiter/train_dit.sbatch "$VAE_RUN" "$DIT_CONFIG")
echo "dit:     ${dit_id}"

echo "diteval: $(sbatch --parsable --dependency="afterok:${dit_id}" \
    jobs/jupiter/eval_dit_vrmse.sbatch "$PRE_ROOT" "$DIT_CKPT_DIR" "$VAE_CKPT")"
