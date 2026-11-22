#!/bin/bash
# SRBA "1-1-2" setup (PhD advisor, 2026-11-22): 1 conditioning frame (frame 0) + temporal stride 2
# (stored frames 0..96, 49 frames, 7 latents), x-axis flip augmentation, VRMSE / VRMSE+H1 loss.
# Full chain per arm (VAE -> vaetest | encode -> DiT LR 5e-5 8k -> diteval), 5 jobs each, 55 total:
#   ref                      no flip, rmse + 50*h1 (SRBA baseline loss)   reference for this setup
#   flip                     flip,    rmse + 50*h1                        flip effect = flip - ref
#   flip_vrmse               flip,    vrms 1
#   flip_vrmse_h1_<w>        flip,    vrms 1 + w*h1, w = 10 30 100 300 1000
#   flip_sds_<w>             flip,    rmse + 50*h1 + w*SDS (stock LTX teacher), w = 3e-3 1e-2 3e-2
#                            (thesis Euler_MQ: stock SDS 1e-2 best pipeline, -8.1%); compare vs flip
# Flip: VAE mirrors training sims with p=0.5; the encode job (reading the VAE's data.flip_augment)
# also writes flipy_<id> latents for every DiT training sim. Val/test are never mirrored.
# Not comparable to the stride-1 SRBA numbers (0.132): different frames generated. Compare arms
# with each other, paired by test sim.
# ONLY=<names> (space-separated) submits a subset; SKIP_CHECK=1 skips the flip-axis check.
#
# Run from the repo root on a jupiter login node, windinet env active:
#   bash jobs/jupiter/submit_srba_s2.sh 2>&1 | tee logs/jupiter/submit_srba_s2_$(date +%Y%m%d_%H%M).txt
set -euo pipefail

ARMS="ref flip flip_vrmse flip_vrmse_h1_10 flip_vrmse_h1_30 flip_vrmse_h1_100 flip_vrmse_h1_300 flip_vrmse_h1_1000 flip_sds_3e3 flip_sds_1e2 flip_sds_3e2"
ONLY=${ONLY:-$ARMS}
want() { [[ " $ONLY " == *" $1 "* ]]; }

if [[ "${SKIP_CHECK:-0}" != "1" ]]; then
    # Gate: flip_y must mirror the axis velocityY points along (scripts/check_flip_axis.py).
    python scripts/check_flip_axis.py /e/project1/e-dev-2026d09-262/neptuna_datasets/2D_SRBA_OOOO/256x256/train.h5
fi

for arm in $ARMS; do
    want "$arm" || continue
    bash jobs/jupiter/submit_pipeline.sh \
        configs/finetune_vae/finetune_vae_jupiter_srba_s2_${arm}_256res.yaml \
        configs/dit/train_dit_jupiter_srba_s2_${arm}.yaml
done
