#!/bin/bash
# Stride-2 baseline, end to end (2026-10-24), trained and tested at stride 2:
# VAE finetune_vae_jupiter_baseline_s44_stride2_256res (every 2nd stored frame,
# 49 frames) -> vaetest | encode -> DiT train_dit_jupiter_baseline_s44_lr1e4_stride2
# -> diteval. Compare against the stride-1 baseline (VAE s44, DiT lr1e4 8k).
# See submit_pipeline.sh for options.
#
# Run from the repo root on a jupiter login node, windinet env active:
#   bash jobs/jupiter/submit_stride2_pipeline.sh
set -euo pipefail
exec bash jobs/jupiter/submit_pipeline.sh \
    configs/finetune_vae/finetune_vae_jupiter_baseline_s44_stride2_256res.yaml \
    configs/dit/train_dit_jupiter_baseline_s44_lr1e4_stride2.yaml
