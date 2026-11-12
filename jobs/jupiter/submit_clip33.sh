#!/bin/bash
# 33-frame VAE training clips (2026-11-12, Harish 11-10: "please try this 30 frames for LTX").
# Two arms on the seed-44 baseline recipe, full chain each (VAE -> vaetest | encode -> DiT cond9
# at LR 1e-4, 8k steps -> diteval), 5 jobs per arm, 10 total:
#   aligned  clip starts 0, 8, ..., 64 (clip latents line up with the full 97-frame encoding)
#   any      clip starts anywhere in 0..64
# Only VAE training is clipped; val, test, encoding and the DiT use all 97 frames.
# References: VAE test ch-mean 0.0805 (baseline s44); DiT cond9 ch-mean 0.3516, benchmark 40-90
# VRMSE 0.576 (job 2634285). Chain noise ~ +-0.005.
# ONLY=<names> (space-separated, e.g. ONLY="aligned") submits a subset.
#
# Run from the repo root on a jupiter login node, windinet env active:
#   bash jobs/jupiter/submit_clip33.sh 2>&1 | tee logs/jupiter/submit_clip33_$(date +%Y%m%d_%H%M).txt
set -euo pipefail

ONLY=${ONLY:-"aligned any"}
for m in ${ONLY}; do
    bash jobs/jupiter/submit_pipeline.sh \
        configs/finetune_vae/finetune_vae_jupiter_clip33_${m}_s44_256res.yaml \
        configs/dit/train_dit_jupiter_clip33_${m}_s44_cond9.yaml
done
