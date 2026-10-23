#!/bin/bash
# 2D-SRBA-OOOO baseline, end to end (VAE -> vaetest | encode -> DiT -> diteval),
# on the PhD advisor's new dataset (2026-10-24). The configs select the dataset
# (data.dataset: srba); see their headers for the recipe. vaetest/diteval score
# the 60 sims of the test.h5 next to the train.h5; encode holds out the VAE
# config's eval_sims (24) as DiT val. See submit_pipeline.sh for options.
#
# Run from the repo root on a jupiter login node, windinet env active:
#   bash jobs/jupiter/submit_srba_baseline.sh
set -euo pipefail
exec bash jobs/jupiter/submit_pipeline.sh \
    configs/finetune_vae/finetune_vae_jupiter_srba_baseline_256res.yaml \
    configs/dit/train_dit_jupiter_srba_baseline.yaml
