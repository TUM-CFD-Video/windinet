#!/bin/bash
# Baseline s44 DiT at learning rates 1e-3 and 1e-2 (2026-10-17), extending the
# 8k LR sweep (3e-6 / 1e-5 / 3e-5 / 1e-4 = 0.4333) upward; pairs with the 3ch
# decoder-only 1e-3/1e-2 arms in submit_3ch_lr.sh. The latents already exist
# (dit_preprocessed/finetune_vae_jupiter_baseline_s44_256res), so each arm is
# DiT + diteval only (2 jobs each, 4 total).
# ONLY=<names> (space-separated, e.g. ONLY="lr1e3") submits a subset.
#
# Run from the repo root on a jupiter login node, windinet env active:
#   bash jobs/jupiter/submit_lr_high.sh 2>&1 | tee logs/jupiter/submit_lr_high_$(date +%Y%m%d_%H%M).txt
set -euo pipefail

ONLY=${ONLY:-"lr1e3 lr1e2"}
want() { [[ " $ONLY " == *" $1 "* ]]; }

for lr in lr1e3 lr1e2; do
    want "$lr" || continue
    SKIP_ENCODE=1 bash jobs/jupiter/submit_pipeline.sh \
        configs/finetune_vae/finetune_vae_jupiter_baseline_s44_256res.yaml \
        configs/dit/train_dit_jupiter_baseline_s44_${lr}.yaml
done
