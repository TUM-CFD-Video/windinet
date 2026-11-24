#!/bin/bash
# Harish's last parallel set (2026-11-24): sampler steps on the best SRBA 1-1-2 model
# (flip + VRMSE: VAE finetune_vae_jupiter_srba_s2_flip_vrmse_256res + DiT LR 5e-5 8k,
# frame-0 conditioning, stride 2). Eval only, no training. Reference: 5 steps = job 3054445
# (ch-mean 0.1188). Steps here: 10 / 20 (thesis protocol) / 50 / 100.
# The sampler noise is seeded per sim (seed + i), so the 5-step reference is the same
# noise and the arms are paired by test sim.
#
# Result: logs/jupiter/eval_dit_vrmse/<job>/{vrmse_summary.json, benchmark_table.md}
# (panels in visualizations/ are gitignored); table rows are named srba_s2_flip_vrmse_<n>steps.
#
# Run from the repo root on a jupiter login node, windinet env active:
#   bash jobs/jupiter/submit_srba_steps.sh
# STEPS="10 20" to change the step counts.
set -euo pipefail

W=/e/scratch/e-dev-2026d09-262/wh_work
VAE_RUN=finetune_vae_jupiter_srba_s2_flip_vrmse_256res
DIT_RUN=shockwave_dit_jupiter_srba_s2_flip_vrmse
STEPS=${STEPS:-"10 20 50 100"}

ARGS=(
    ${W}/dit_preprocessed/${VAE_RUN}
    ${W}/dit_outputs/${DIT_RUN}/checkpoints/model_weights_step_08296.safetensors
    ${W}/finetune_vae_outputs/${VAE_RUN}/checkpoints/vae_shockwave_best.safetensors
)
for f in "${ARGS[@]}"; do
    [[ -e "${f}" ]] || { echo "error: missing ${f}" >&2; exit 1; }
done

LOG=logs/jupiter/submit_srba_steps_$(date +%Y%m%d_%H%M).txt
for s in ${STEPS}; do
    name=srba_s2_flip_vrmse_${s}steps
    job=$(NUM_INFERENCE_STEPS="${s}" TABLE_NAME="${name}" \
        sbatch --parsable --time=01:00:00 \
        jobs/jupiter/eval_dit_vrmse.sbatch "${ARGS[@]}")
    echo "${name}: ${job}" | tee -a "${LOG}"
done
echo "job ids saved to ${LOG}"
