#!/bin/bash
# Single-trajectory overfit (2026-10-17): how well can the VAE and the DiT fit ONE
# trajectory (2507_gamma1.3999999762) when generalization is out of the picture?
#   vae      finetune_vae_jupiter_overfit_1sim_256res: trains + evaluates on that sim only
#            (BATCH_SIZE=1 EFFECTIVE_BATCH=4, 3000 steps). Result: metrics/metrics.csv
#            val_vrmse_chmean. No test eval / encode -- this VAE is a diagnostic only.
#   dit      train_dit_jupiter_overfit_1sim on the baseline s44 latents of that sim
#            (4000 steps, LR 1e-4), then diteval on that sim (TEST_H5=none, SIM_IDS):
#            vae_dit_vrmse_chmean = DiT overfit pixel VRMSE, vae_only = the baseline
#            s44 VAE on the same sim (the floor the DiT can reach in pixel space).
#   ref      the regular baseline s44 LR 1e-4 8k DiT (2189166, 0.4333 on test) evaluated
#            on the same sim, for a trained-normally comparison.
# 4 jobs. ONLY=<names> (space-separated, e.g. ONLY="dit") submits a subset.
#
# Run from the repo root on a jupiter login node, windinet env active:
#   bash jobs/jupiter/submit_overfit_1sim.sh 2>&1 | tee logs/jupiter/submit_overfit_1sim_$(date +%Y%m%d_%H%M).txt
set -euo pipefail

ONLY=${ONLY:-"vae dit ref"}
want() { [[ " $ONLY " == *" $1 "* ]]; }
SIM=2507_gamma1.3999999762

W=/e/scratch/e-dev-2026d09-262/wh_work
VAE_RUN=finetune_vae_jupiter_baseline_s44_256res

want vae && echo "vae: $(BATCH_SIZE=1 EFFECTIVE_BATCH=4 sbatch --parsable --time=02:00:00 \
    jobs/jupiter/finetune_vae.sbatch configs/finetune_vae/finetune_vae_jupiter_overfit_1sim_256res.yaml)"
want dit && SIM_IDS=$SIM TEST_H5=none SKIP_ENCODE=1 DIT_TIME=03:00:00 EVAL_TIME=00:30:00 \
    bash jobs/jupiter/submit_pipeline.sh configs/finetune_vae/${VAE_RUN}.yaml configs/dit/train_dit_jupiter_overfit_1sim.yaml
want ref && echo "ref: $(SIM_IDS=$SIM TEST_H5=none sbatch --parsable --time=00:30:00 jobs/jupiter/eval_dit_vrmse.sbatch \
    ${W}/dit_preprocessed/${VAE_RUN} \
    ${W}/dit_outputs/shockwave_dit_jupiter_baseline_s44_lr1e4/checkpoints/model_weights_step_08000.safetensors \
    ${W}/finetune_vae_outputs/${VAE_RUN}/checkpoints/vae_shockwave_best.safetensors)"
true
