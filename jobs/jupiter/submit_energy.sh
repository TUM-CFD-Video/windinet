#!/bin/bash
# Total energy E = p/(gamma-1) + (m_x^2 + m_y^2)/(2 rho) (2026-11-12, Harish 11-07), windinet.losses.energy.
#   baseline  C: eval only. The cond9 baseline re-evaluated on test.h5 (500 sims) for the new
#             energy_vrmse block of vrmse_summary.json (gt_clipped floor / vae_only / vae_dit).
#   1e2 3e2 1e1  A: VAE energy-loss weight 0.01 / 0.03 / 0.1 on the seed-44 baseline recipe, full chain
#             each (VAE -> vaetest | encode -> DiT cond9 LR 1e-4 8k -> diteval), 5 jobs per arm.
# References: VAE test ch-mean 0.0805; DiT cond9 ch-mean 0.3516, benchmark 40-90 VRMSE 0.576.
# ONLY=<names> (space-separated, e.g. ONLY="baseline") submits a subset.
#
# Run from the repo root on a jupiter login node, windinet env active:
#   bash jobs/jupiter/submit_energy.sh 2>&1 | tee logs/jupiter/submit_energy_$(date +%Y%m%d_%H%M).txt
set -euo pipefail

ONLY=${ONLY:-"baseline 1e2 3e2 1e1"}
want() { [[ " $ONLY " == *" $1 "* ]]; }

if want baseline; then
    W=/e/scratch/e-dev-2026d09-262/wh_work
    VAE_RUN=finetune_vae_jupiter_baseline_s44_256res
    echo "baseline (cond9 energy eval): $(TABLE_NAME="cond9 energy eval" sbatch --parsable jobs/jupiter/eval_dit_vrmse.sbatch \
        ${W}/dit_preprocessed/${VAE_RUN} \
        ${W}/dit_outputs/shockwave_dit_jupiter_baseline_s44_lr1e4_cond9/checkpoints/model_weights_step_08000.safetensors \
        ${W}/finetune_vae_outputs/${VAE_RUN}/checkpoints/vae_shockwave_best.safetensors)"
fi
for w in 1e2 3e2 1e1; do
    want "$w" || continue
    bash jobs/jupiter/submit_pipeline.sh \
        configs/finetune_vae/finetune_vae_jupiter_energy_${w}_s44_256res.yaml \
        configs/dit/train_dit_jupiter_energy_${w}_s44_cond9.yaml
done
true
