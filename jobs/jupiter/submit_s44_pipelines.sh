#!/bin/bash
# Seed-44 end-to-end comparison on jupiter (2026-10-05), each arm through
# jobs/jupiter/submit_pipeline.sh:
#
#   baseline   finetune_vae_jupiter_baseline_s44_256res (already trained by
#              submit_stability_s44.sh) + train_dit_jupiter_baseline_s44:
#              SKIP_VAE=1 -> encode + DiT + diteval (3 jobs)
#   KL         finetune_vae_jupiter_kl_1e{7,5,3}_s44_256res + train_dit_jupiter_kl_1e{7,5,3}_s44:
#              full chains (5 jobs each)
#   DiT LR     train_dit_jupiter_baseline_s44_lr{3e6,3e5,1e4} on the baseline's
#              latents (1e-5 is the baseline DiT itself): DiT + diteval only,
#              waiting on the baseline's encode job (2 jobs each)
#   sampler    the baseline s44 DiT re-evaluated with 40 and 80 flow-matching
#              integration steps instead of configs/dit/inference_dit.yaml's 20
#              (inference only, afterok on the baseline's DiT job; 2 jobs)
#
# 26 jobs. Round 1 (seed 42) lost the baseline and the KL arms to a mid-training
# VAE collapse; seed 44 trained the baseline cleanly, so this reruns the KL sweep
# on seed 44 and gives it a baseline DiT on the same seed.
# ONLY=<names> (space-separated, e.g. ONLY="kl_1e5") submits a subset;
# "lr3e6"/"lr3e5"/"lr1e4"/"steps" need the baseline in the same call (they chain
# on its encode / DiT job).
#
# Run from the repo root on a jupiter login node, windinet env active:
#   bash jobs/jupiter/submit_s44_pipelines.sh 2>&1 | tee logs/jupiter/submit_s44_pipelines_$(date +%Y%m%d_%H%M).txt
set -euo pipefail

ONLY=${ONLY:-"baseline kl_1e7 kl_1e5 kl_1e3 lr3e6 lr3e5 lr1e4 steps"}
want() { [[ " $ONLY " == *" $1 "* ]]; }
V=configs/finetune_vae
D=configs/dit

W=/e/scratch/e-dev-2026d09-262/wh_work
pre_baseline=""
dit_baseline=""
if want baseline; then
    out=$(SKIP_VAE=1 bash jobs/jupiter/submit_pipeline.sh \
        $V/finetune_vae_jupiter_baseline_s44_256res.yaml $D/train_dit_jupiter_baseline_s44.yaml)
    echo "$out"
    pre_baseline=$(awk '/^encode: /{print $2}' <<<"$out")
    dit_baseline=$(awk '/^dit: /{print $2}' <<<"$out")
fi
for arm in kl_1e7 kl_1e5 kl_1e3; do
    want "$arm" && bash jobs/jupiter/submit_pipeline.sh \
        $V/finetune_vae_jupiter_${arm}_s44_256res.yaml $D/train_dit_jupiter_${arm}_s44.yaml
done
for lr in lr3e6 lr3e5 lr1e4; do
    want "$lr" || continue
    if [ -z "$pre_baseline" ]; then
        echo "error: $lr chains on the baseline's encode job; include baseline in ONLY" >&2; exit 1
    fi
    SKIP_ENCODE=1 AFTER="$pre_baseline" bash jobs/jupiter/submit_pipeline.sh \
        $V/finetune_vae_jupiter_baseline_s44_256res.yaml $D/train_dit_jupiter_baseline_s44_${lr}.yaml
done
if want steps; then
    if [ -z "$dit_baseline" ]; then
        echo "error: steps chains on the baseline's DiT job; include baseline in ONLY" >&2; exit 1
    fi
    # eval time scales with the sampler steps (20 steps fit the 4h default; 12h is the booster limit)
    for n in 40 80; do
        echo "diteval_steps${n}: $(NUM_INFERENCE_STEPS=$n sbatch --parsable --dependency="afterok:${dit_baseline}" \
            --time=$([ "$n" = 40 ] && echo 08:00:00 || echo 12:00:00) jobs/jupiter/eval_dit_vrmse.sbatch \
            ${W}/dit_preprocessed/finetune_vae_jupiter_baseline_s44_256res \
            ${W}/dit_outputs/shockwave_dit_jupiter_baseline_s44/checkpoints \
            ${W}/finetune_vae_outputs/finetune_vae_jupiter_baseline_s44_256res/checkpoints/vae_shockwave_best.safetensors)"
    done
fi
