#!/bin/bash
# Stability check (2026-10-05): the jupiter baseline VAE on seed 44, on 97
# frames (A) and on 101 frames (B), VAE + test eval only (no encode/DiT).
# Round 1's seed-42 baseline collapsed at epoch 5; see the two configs' headers.
# Both runs also write metrics/grad_norms.csv (pre-clip gradient norm per step).
#
# Run from the repo root on a jupiter login node, windinet env active:
#   bash jobs/jupiter/submit_stability_s44.sh 2>&1 | tee logs/jupiter/submit_stability_s44_$(date +%Y%m%d_%H%M).txt
set -euo pipefail

mkdir -p logs/jupiter
for cfg in configs/finetune_vae/finetune_vae_jupiter_baseline_s44_256res.yaml \
           configs/finetune_vae/finetune_vae_jupiter_baseline_101f_s44_256res.yaml; do
    echo "# ${cfg}"
    vae_id=$(sbatch --parsable jobs/jupiter/finetune_vae.sbatch "$cfg")
    echo "vae: ${vae_id}"
    echo "vaetest: $(sbatch --parsable --dependency="afterok:${vae_id}" jobs/jupiter/eval_vae_test.sbatch "$cfg")"
done
