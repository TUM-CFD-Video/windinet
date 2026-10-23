#!/bin/bash
# Sampler-step sweep on the best SWD arm (2026-10-24): the 8k DiT on the SWD 1e-1
# VAE (shockwave_dit_jupiter_swd_1e1_s44; 20 steps = 2252533, all-frames ch-mean
# 0.4188, the best 8k result) re-evaluated with 1, 2, 5, 10 and 20 flow-matching
# integration steps. The baseline (shockwave_dit_jupiter_baseline_s44_lr1e4, 8k)
# is re-evaluated with the same steps alongside it, because every eval now uses
# the 2026-10-24 standard (benchmark block: frames 40, 50, ..., 90, prediction and
# target clamped), which no earlier eval of either model was run under.
# Inference only, no dependencies -- the checkpoints exist. 10 jobs; 20 steps fit
# the 4h default, so fewer steps do too.
# ONLY=<names> (space-separated, e.g. ONLY="swd1e1") submits a subset; STEPS overrides the list.
#
# Run from the repo root on a jupiter login node, windinet env active:
#   bash jobs/jupiter/submit_swd_sampler_steps.sh 2>&1 | tee logs/jupiter/submit_swd_sampler_steps_$(date +%Y%m%d_%H%M).txt
set -euo pipefail

ONLY=${ONLY:-"swd1e1 baseline"}
STEPS=${STEPS:-"1 2 5 10 20"}
want() { [[ " $ONLY " == *" $1 "* ]]; }

W=/e/scratch/e-dev-2026d09-262/wh_work
declare -A VAE=([swd1e1]=finetune_vae_jupiter_swd_1e1_s44_256res [baseline]=finetune_vae_jupiter_baseline_s44_256res)
declare -A DIT=([swd1e1]=shockwave_dit_jupiter_swd_1e1_s44 [baseline]=shockwave_dit_jupiter_baseline_s44_lr1e4)

for arm in swd1e1 baseline; do
    want "$arm" || continue
    for n in $STEPS; do
        echo "diteval_${arm}_steps${n}: $(NUM_INFERENCE_STEPS=$n sbatch --parsable jobs/jupiter/eval_dit_vrmse.sbatch \
            ${W}/dit_preprocessed/${VAE[$arm]} \
            ${W}/dit_outputs/${DIT[$arm]}/checkpoints/model_weights_step_08000.safetensors \
            ${W}/finetune_vae_outputs/${VAE[$arm]}/checkpoints/vae_shockwave_best.safetensors)"
    done
done
