#!/bin/bash
# Latent noise sweep (2026-11-01, Harish): latent_noise_std 1e-5 / 1e-4 / 1e-3 / 1e-2 / 1e-1 on the
# seed-44 baseline recipe, full chain per arm (VAE -> vaetest | encode -> DiT at
# LR 1e-4, 8k steps -> diteval), 5 jobs per arm, 25 total. References: baseline s44
# VAE test ch-mean 0.0799, DiT (LR 1e-4, 8k, 5 sampler steps) 0.4285; klmean 1e-5 (learned
# sigma) 0.4255.
# ONLY=<names> (space-separated, e.g. ONLY="1e4") submits a subset.
#
# Run from the repo root on a jupiter login node, windinet env active:
#   bash jobs/jupiter/submit_latnoise.sh 2>&1 | tee logs/jupiter/submit_latnoise_$(date +%Y%m%d_%H%M).txt
set -euo pipefail

ONLY=${ONLY:-"1e5 1e4 1e3 1e2 1e1"}
want() { [[ " $ONLY " == *" $1 "* ]]; }

for w in 1e5 1e4 1e3 1e2 1e1; do
    want "$w" || continue
    bash jobs/jupiter/submit_pipeline.sh \
        configs/finetune_vae/finetune_vae_jupiter_latnoise_${w}_s44_256res.yaml \
        configs/dit/train_dit_jupiter_latnoise_${w}_s44.yaml
done
