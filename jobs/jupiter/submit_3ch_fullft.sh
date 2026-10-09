#!/bin/bash
# 3-channel full-finetune control (2026-10-09): full chain (VAE -> vaetest | encode ->
# DiT at LR 1e-4, 8k steps -> diteval), 5 jobs. Splits the 3ch decoder-only arm's DiT
# gap to the baseline (0.478 vs 0.449 over the same 3 channels) into "pressure
# missing" vs "encoder frozen"; see the VAE config's header.
#
# Run from the repo root on a jupiter login node, windinet env active:
#   bash jobs/jupiter/submit_3ch_fullft.sh 2>&1 | tee logs/jupiter/submit_3ch_fullft_$(date +%Y%m%d_%H%M).txt
set -euo pipefail

bash jobs/jupiter/submit_pipeline.sh \
    configs/finetune_vae/finetune_vae_jupiter_3ch_fullft_256res.yaml \
    configs/dit/train_dit_jupiter_3ch_fullft_lr1e4.yaml
