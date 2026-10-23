#!/bin/bash
# Low sampler-step sweep (2026-10-17): existing 8k DiTs re-evaluated with 1, 2,
# 5 and 10 flow-matching integration steps (configs/dit/inference_dit.yaml
# default was 20; 5 since 2026-10-22). Inference only, no dependencies -- the checkpoints exist.
#   lr1e5  shockwave_dit_jupiter_baseline_s44        (same DiT as the 40/80-step
#          evals 2189167/8; 20 steps = 2189145, ch-mean 0.489)
#   lr1e4  shockwave_dit_jupiter_baseline_s44_lr1e4  (reference; 20 steps = 2189166, 0.4333)
# 8 jobs. Eval time scales with the steps; 20 steps fit the 4h default, so <=10 do too.
# ONLY=<names> (space-separated, e.g. ONLY="lr1e5") submits a subset; STEPS overrides the list.
#
# Run from the repo root on a jupiter login node, windinet env active:
#   bash jobs/jupiter/submit_sampler_steps_low.sh 2>&1 | tee logs/jupiter/submit_sampler_steps_low_$(date +%Y%m%d_%H%M).txt
set -euo pipefail

ONLY=${ONLY:-"lr1e5 lr1e4"}
STEPS=${STEPS:-"1 2 5 10"}
want() { [[ " $ONLY " == *" $1 "* ]]; }

W=/e/scratch/e-dev-2026d09-262/wh_work
VAE_RUN=finetune_vae_jupiter_baseline_s44_256res
declare -A DIT=([lr1e5]=shockwave_dit_jupiter_baseline_s44 [lr1e4]=shockwave_dit_jupiter_baseline_s44_lr1e4)

for arm in lr1e5 lr1e4; do
    want "$arm" || continue
    for n in $STEPS; do
        echo "diteval_${arm}_steps${n}: $(NUM_INFERENCE_STEPS=$n sbatch --parsable jobs/jupiter/eval_dit_vrmse.sbatch \
            ${W}/dit_preprocessed/${VAE_RUN} \
            ${W}/dit_outputs/${DIT[$arm]}/checkpoints/model_weights_step_08000.safetensors \
            ${W}/finetune_vae_outputs/${VAE_RUN}/checkpoints/vae_shockwave_best.safetensors)"
    done
done
