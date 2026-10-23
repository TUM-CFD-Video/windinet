#!/bin/bash
# SRBA smoke test (2026-10-24): the whole pipeline (VAE 2 epochs -> vaetest |
# encode -> DiT 40 steps -> diteval on the 60 test sims) with 1h walltimes,
# only to check the srba code path runs on jupiter. Numbers are meaningless.
# Rerun after deleting the outputs it wrote, e.g.:
#   rm -rf /e/scratch/e-dev-2026d09-262/wh_work/{finetune_vae_outputs/finetune_vae_jupiter_srba_smoketest_256res,dit_preprocessed/finetune_vae_jupiter_srba_smoketest_256res,dit_outputs/shockwave_dit_jupiter_srba_smoketest}
#
# Run from the repo root on a jupiter login node, windinet env active:
#   bash jobs/jupiter/submit_srba_smoketest.sh
set -euo pipefail
VAE_TIME=01:00:00 VAETEST_TIME=00:30:00 ENCODE_TIME=01:00:00 DIT_TIME=01:00:00 EVAL_TIME=01:00:00 \
exec bash jobs/jupiter/submit_pipeline.sh \
    configs/finetune_vae/finetune_vae_jupiter_srba_smoketest_256res.yaml \
    configs/dit/train_dit_jupiter_srba_smoketest.yaml
