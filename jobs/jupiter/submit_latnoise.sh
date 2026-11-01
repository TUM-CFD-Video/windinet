#!/bin/bash
# Latent noise sweep (2026-11-01, Harish): latent_noise_std 1e-5 / 1e-4 / 1e-3 / 1e-2 / 1e-1 on the
# seed-44 baseline recipe, full chain per arm (VAE -> vaetest | encode -> DiT at
# LR 1e-4, 8k steps -> diteval), 5 jobs per arm, 25 total. References: baseline s44
# VAE test ch-mean 0.0799, DiT (LR 1e-4, 8k, 5 sampler steps) 0.4285; klmean 1e-5 (learned
# sigma) 0.4255.
# baseline: eval only, re-runs the baseline s44 + DiT LR 1e-4 8k eval with the same settings as
# the arms' diteval, so its benchmark block has the H1/H2 seminorms (5a0d4a6) the old eval lacks.
# ONLY=<names> (space-separated, e.g. ONLY="1e4 baseline") submits a subset.
#
# Run from the repo root on a jupiter login node, windinet env active:
#   bash jobs/jupiter/submit_latnoise.sh 2>&1 | tee logs/jupiter/submit_latnoise_$(date +%Y%m%d_%H%M).txt
set -euo pipefail

ONLY=${ONLY:-"1e5 1e4 1e3 1e2 1e1 baseline"}
want() { [[ " $ONLY " == *" $1 "* ]]; }

for w in 1e5 1e4 1e3 1e2 1e1; do
    want "$w" || continue
    bash jobs/jupiter/submit_pipeline.sh \
        configs/finetune_vae/finetune_vae_jupiter_latnoise_${w}_s44_256res.yaml \
        configs/dit/train_dit_jupiter_latnoise_${w}_s44.yaml
done
if want baseline; then
    W=/e/scratch/e-dev-2026d09-262/wh_work
    VAE_RUN=finetune_vae_jupiter_baseline_s44_256res
    echo "baseline: $(TABLE_NAME="256: baseline s44 + DiT 8k" sbatch --parsable jobs/jupiter/eval_dit_vrmse.sbatch \
        ${W}/dit_preprocessed/${VAE_RUN} \
        ${W}/dit_outputs/shockwave_dit_jupiter_baseline_s44_lr1e4/checkpoints/model_weights_step_08000.safetensors \
        ${W}/finetune_vae_outputs/${VAE_RUN}/checkpoints/vae_shockwave_best.safetensors)"
fi
true
