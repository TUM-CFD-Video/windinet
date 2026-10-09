#!/bin/bash
# Sliced-Wasserstein (WAE) prior sweep (2026-10-09): weights 1e-2 / 1e-1 / 1 on the
# seed-44 baseline recipe, full chain per arm (VAE -> vaetest | encode -> DiT at
# LR 1e-4, 8k steps -> diteval), 5 jobs per arm, 15 total. References: baseline s44
# VAE test ch-mean 0.0799, DiT (LR 1e-4, 8k) 0.433 (job 2189166); KL counterpart:
# submit_klmean.sh.
# ONLY=<names> (space-separated, e.g. ONLY="1e1") submits a subset.
#
# Run from the repo root on a jupiter login node, windinet env active:
#   bash jobs/jupiter/submit_swd.sh 2>&1 | tee logs/jupiter/submit_swd_$(date +%Y%m%d_%H%M).txt
set -euo pipefail

ONLY=${ONLY:-"1e2 1e1 1e0"}
want() { [[ " $ONLY " == *" $1 "* ]]; }

for w in 1e2 1e1 1e0; do
    want "$w" || continue
    bash jobs/jupiter/submit_pipeline.sh \
        configs/finetune_vae/finetune_vae_jupiter_swd_${w}_s44_256res.yaml \
        configs/dit/train_dit_jupiter_swd_${w}_s44.yaml
done
