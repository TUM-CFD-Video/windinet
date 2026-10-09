#!/bin/bash
# Longer DiT runs at LR 1e-4 (2026-10-09), DiT + diteval only on existing latents:
#   512_10k  train_dit_jupiter_baseline_512res_lr1e4_10k  (~9.3h at ~1070 steps/h -> 12h, no resume)
#   512_20k  train_dit_jupiter_baseline_512res_lr1e4_20k  (~18.7h -> 12h + 1 resume)
#   256_20k  train_dit_jupiter_baseline_s44_lr1e4_20k     (~17h  -> 12h + 1 resume)
# References: 512 LR 1e-5 8k ch-mean 0.441 (2189231); 256 s44 LR 1e-4 8k 0.433 (2189166).
# Step rates and eval times from sacct of the first 512 chain (submit_baseline_512.sh).
# ONLY=<names> (space-separated, e.g. ONLY="512_10k") submits a subset.
#
# Run from the repo root on a jupiter login node, windinet env active:
#   bash jobs/jupiter/submit_lr1e4_long.sh 2>&1 | tee logs/jupiter/submit_lr1e4_long_$(date +%Y%m%d_%H%M).txt
set -euo pipefail

ONLY=${ONLY:-"512_10k 512_20k 256_20k"}
want() { [[ " $ONLY " == *" $1 "* ]]; }

VAE512=configs/finetune_vae/finetune_vae_jupiter_baseline_512res.yaml
VAE256=configs/finetune_vae/finetune_vae_jupiter_baseline_s44_256res.yaml

want 512_10k && SKIP_ENCODE=1 DIT_TIME=12:00:00 DIT_RESUMES=0 EVAL_TIME=01:30:00 \
    bash jobs/jupiter/submit_pipeline.sh "$VAE512" configs/dit/train_dit_jupiter_baseline_512res_lr1e4_10k.yaml
want 512_20k && SKIP_ENCODE=1 DIT_TIME=12:00:00 DIT_RESUMES=1 EVAL_TIME=01:30:00 \
    bash jobs/jupiter/submit_pipeline.sh "$VAE512" configs/dit/train_dit_jupiter_baseline_512res_lr1e4_20k.yaml
want 256_20k && SKIP_ENCODE=1 DIT_TIME=12:00:00 DIT_RESUMES=1 \
    bash jobs/jupiter/submit_pipeline.sh "$VAE256" configs/dit/train_dit_jupiter_baseline_s44_lr1e4_20k.yaml
true
