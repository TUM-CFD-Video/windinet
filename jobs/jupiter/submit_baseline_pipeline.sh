#!/bin/bash
# Jupiter baseline, end to end (VAE -> vaetest | encode -> DiT -> diteval).
# Thesis baseline recipe (VAE "Full FT" + 8k-step DiT, THESIS_RESULTS.md) on
# 97-frame sims, the project standard from 2026-10-04 (the thesis used 101,
# padded to 105), so its test numbers are the new reference rather than
# directly comparable to the thesis. See submit_pipeline.sh for the stages and
# options (SKIP_VAE / SKIP_ENCODE / AFTER); submit_experiments.sh submits this
# together with the other arms.
#
# Run from the repo root on a jupiter login node, windinet env active:
#   bash jobs/jupiter/submit_baseline_pipeline.sh
set -euo pipefail
exec bash jobs/jupiter/submit_pipeline.sh \
    configs/finetune_vae/finetune_vae_jupiter_baseline_256res.yaml \
    configs/dit/train_dit_jupiter_baseline.yaml
