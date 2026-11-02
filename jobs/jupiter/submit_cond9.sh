#!/bin/bash
# Multi-frame conditioning, option A (2026-11-01): DiT + diteval on the
# existing baseline s44 latents, conditioned on frames 0-8 (2 latent frames)
# instead of frame 0. Compare against the baseline VAE s44 + DiT lr1e4 8k
# (ch-mean 0.4285, job 2362471). SHOCK_ANALYSIS=1: the diteval also splits the
# error by region and latent token (baseline counterpart: submit_shock_analysis.sh).
# See submit_pipeline.sh for options.
#
# Run from the repo root on a jupiter login node, windinet env active:
#   bash jobs/jupiter/submit_cond9.sh
set -euo pipefail
SHOCK_ANALYSIS=1 SKIP_ENCODE=1 exec bash jobs/jupiter/submit_pipeline.sh \
    configs/finetune_vae/finetune_vae_jupiter_baseline_s44_256res.yaml \
    configs/dit/train_dit_jupiter_baseline_s44_lr1e4_cond9.yaml
