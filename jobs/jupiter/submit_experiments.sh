#!/bin/bash
# Submit the first round of post-thesis experiments on jupiter (2026-10-04),
# each a full jobs/jupiter/submit_pipeline.sh chain (1 node x 4 GPUs per job):
#
#   1  baseline           finetune_vae_jupiter_baseline_256res   + train_dit_jupiter_baseline
#   2  KL 1e-7/1e-5/1e-3  finetune_vae_jupiter_kl_1e{7,5,3}_256res + train_dit_jupiter_kl_1e{7,5,3}
#      (training now decodes reparameterized posterior samples when kl > 0)
#   3  3 channels         finetune_vae_jupiter_3ch_decoder_only_256res + train_dit_jupiter_3ch_decoder_only
#      (density/momentum_x/momentum_y, stock encoder frozen, decoder only)
#   4  DiT LR 3e-6/3e-5   train_dit_jupiter_baseline_lr3e{6,5} on the baseline's
#      latents: DiT + diteval only, waiting on the baseline's encode job
#
# 5 full chains (5 jobs each) + 2 DiT-only chains (2 each) = 29 jobs. Roughly 3.5h VAE + encode + 6.7h DiT +
# ~1.5h eval per chain; total ~100 node-hours if they all run in parallel.
# ONLY=<names> (space-separated, e.g. ONLY="baseline kl_1e5") submits a subset;
# "lr3e6"/"lr3e5" need the baseline in the same call (they chain on its encode).
#
# Run from the repo root on a jupiter login node, windinet env active:
#   bash jobs/jupiter/submit_experiments.sh 2>&1 | tee logs/jupiter/submit_experiments_$(date +%Y%m%d_%H%M).txt
set -euo pipefail

ONLY=${ONLY:-"baseline kl_1e7 kl_1e5 kl_1e3 3ch_decoder_only lr3e6 lr3e5"}
want() { [[ " $ONLY " == *" $1 "* ]]; }
V=configs/finetune_vae
D=configs/dit

pre_baseline=""
if want baseline; then
    out=$(bash jobs/jupiter/submit_pipeline.sh $V/finetune_vae_jupiter_baseline_256res.yaml $D/train_dit_jupiter_baseline.yaml)
    echo "$out"
    pre_baseline=$(awk '/^encode: /{print $2}' <<<"$out")
fi
for arm in kl_1e7 kl_1e5 kl_1e3 3ch_decoder_only; do
    want "$arm" && bash jobs/jupiter/submit_pipeline.sh $V/finetune_vae_jupiter_${arm}_256res.yaml $D/train_dit_jupiter_${arm}.yaml
done
for lr in lr3e6 lr3e5; do
    want "$lr" || continue
    if [ -z "$pre_baseline" ]; then
        echo "error: $lr chains on the baseline's encode job; include baseline in ONLY" >&2; exit 1
    fi
    SKIP_ENCODE=1 AFTER="$pre_baseline" bash jobs/jupiter/submit_pipeline.sh \
        $V/finetune_vae_jupiter_baseline_256res.yaml $D/train_dit_jupiter_baseline_${lr}.yaml
done
