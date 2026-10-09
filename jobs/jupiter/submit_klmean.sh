#!/bin/bash
# KL sweep with the per-element KL (2026-10-09): weights 1e-5 / 1e-4 / 1e-3 on the
# seed-44 baseline recipe, full chain per arm (VAE -> vaetest | encode -> DiT at
# LR 1e-4, 8k steps -> diteval), 5 jobs per arm, 15 total. References: baseline s44
# VAE test ch-mean 0.0799, DiT (LR 1e-4, 8k) 0.433 (job 2189166).
# ONLY=<names> (space-separated, e.g. ONLY="1e4") submits a subset.
#
# Run from the repo root on a jupiter login node, windinet env active:
#   bash jobs/jupiter/submit_klmean.sh 2>&1 | tee logs/jupiter/submit_klmean_$(date +%Y%m%d_%H%M).txt
set -euo pipefail

ONLY=${ONLY:-"1e5 1e4 1e3"}
want() { [[ " $ONLY " == *" $1 "* ]]; }

for w in 1e5 1e4 1e3; do
    want "$w" || continue
    bash jobs/jupiter/submit_pipeline.sh \
        configs/finetune_vae/finetune_vae_jupiter_klmean_${w}_s44_256res.yaml \
        configs/dit/train_dit_jupiter_klmean_${w}_s44.yaml
done
