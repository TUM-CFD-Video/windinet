#!/bin/bash
# 3ch decoder-only DiT at learning rates 1e-6, 1e-7 and 1e-4 (2026-10-06): one and two
# orders of magnitude below, and one above, the 1e-5 it ran at (ch-mean 0.616, vs 0.503 for the
# baseline s44's 3 channels). Tests whether that LR was too large or too small for the stock
# encoder's latents. The latents already exist
# (dit_preprocessed/finetune_vae_jupiter_3ch_decoder_only_256res), so each arm
# is DiT + diteval only (2 jobs each, 6 total).
# ONLY=<names> (space-separated, e.g. ONLY="lr1e6") submits a subset.
#
# Run from the repo root on a jupiter login node, windinet env active:
#   bash jobs/jupiter/submit_3ch_lr.sh 2>&1 | tee logs/jupiter/submit_3ch_lr_$(date +%Y%m%d_%H%M).txt
set -euo pipefail

ONLY=${ONLY:-"lr1e6 lr1e7 lr1e4"}
want() { [[ " $ONLY " == *" $1 "* ]]; }

for lr in lr1e6 lr1e7 lr1e4; do
    want "$lr" || continue
    SKIP_ENCODE=1 bash jobs/jupiter/submit_pipeline.sh \
        configs/finetune_vae/finetune_vae_jupiter_3ch_decoder_only_256res.yaml \
        configs/dit/train_dit_jupiter_3ch_decoder_only_${lr}.yaml
done
