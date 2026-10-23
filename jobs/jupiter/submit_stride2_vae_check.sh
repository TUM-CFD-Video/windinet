#!/bin/bash
# Stride-2 check, step 1 (2026-10-24), no training: the baseline VAE
# (finetune_vae_jupiter_baseline_s44_256res, test all-frames ch-mean 0.0799 at
# stride 1) evaluated on test.h5 subsampled to every 2nd frame (0, 2, ..., 96),
# through finetune_vae_jupiter_baseline_s44_stride2_256res.yaml. Compare
# vrmse_chmean in the output JSON against 0.0799; see that config's header.
# One job, VAE inference only.
#
# Run from the repo root on a jupiter login node, windinet env active:
#   bash jobs/jupiter/submit_stride2_vae_check.sh
set -euo pipefail
W=/e/scratch/e-dev-2026d09-262/wh_work
echo "vaetest_stride2: $(sbatch --parsable jobs/jupiter/eval_vae_test.sbatch \
    configs/finetune_vae/finetune_vae_jupiter_baseline_s44_stride2_256res.yaml \
    ${W}/finetune_vae_outputs/finetune_vae_jupiter_baseline_s44_256res/checkpoints/vae_shockwave_best.safetensors \
    finetune_vae_outputs/jupiter/finetune_vae_jupiter_baseline_s44_stride2_256res/test_eval_256_baseline_vae.json)"
