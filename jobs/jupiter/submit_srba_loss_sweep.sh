#!/bin/bash
# 2D-SRBA-OOOO VAE loss-weight sweep (2026-11-01) around the SRBA baseline VAE (rmse 1.0 + h1 50),
# one change per arm, full chain per arm (VAE -> vaetest | encode -> DiT LR 5e-5, 8k -> diteval),
# 5 jobs each, 45 total:
#   h1_0 h1_10 h1_100 h1_200   h1 weight (baseline 50)
#   h2_1 h2_10 h2_50           + h2 (second spatial differences)
#   ssim0p2                    + ssim 0.2
#   mlw0p05                    + mlw 0.05
# Reference: baseline VAE (test ch-mean 0.0657) + DiT LR 5e-5 8k, e2e ch-mean 0.1320
# (shockwave_dit_jupiter_srba_dit_lr5e5; seed replicate 0.1323). Pair by test sim.
# ONLY=<names> (space-separated, e.g. ONLY="h1_0 h2_10") submits a subset.
#
# Run from the repo root on a jupiter login node, windinet env active:
#   bash jobs/jupiter/submit_srba_loss_sweep.sh 2>&1 | tee logs/jupiter/submit_srba_loss_sweep_$(date +%Y%m%d_%H%M).txt
set -euo pipefail

ARMS="h1_0 h1_10 h1_100 h1_200 h2_1 h2_10 h2_50 ssim0p2 mlw0p05"
ONLY=${ONLY:-$ARMS}
want() { [[ " $ONLY " == *" $1 "* ]]; }

for arm in $ARMS; do
    want "$arm" || continue
    bash jobs/jupiter/submit_pipeline.sh \
        configs/finetune_vae/finetune_vae_jupiter_srba_loss_${arm}_256res.yaml \
        configs/dit/train_dit_jupiter_srba_loss_${arm}.yaml
done
