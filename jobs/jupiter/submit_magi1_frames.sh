#!/bin/bash
# MAGI-1 comparison (2026-10-29, Harish): DiTs conditioned on frame 0 only (not on
# input frames 0/10/20/30 like cfd_bench / MAGI-1), scored from frame 10 of
# 256x256_ds/test.h5. Prediction and target clamped to [-1, 1].
#   baseline   baseline s44 + DiT LR 1e-4 8k, 5 steps (cfd_bench 40..90: 2481831). 97-frame
#              sims end at frame 96, so frames 10..90 only -- say so when comparing.
#   thesis30k  thesis-VAE ch6_loss_rmse_h1 + DiT 30k, 20 steps as in its cfd_bench eval
#              (2413328). 101-frame sims, so the full 10..100.
# Result: the "benchmark" block of logs/jupiter/eval_dit_vrmse/<job>/vrmse_summary.json --
# mean over the scored frames (rmse/vrmse/ssim/mlw) and benchmark.per_frame (rmse/vrmse/ssim
# per frame); scripts/benchmark_table.py prints both as tables.
# 2 jobs. ONLY=<names> (space-separated) submits a subset.
#
# Run from the repo root on a jupiter login node, windinet env active:
#   bash jobs/jupiter/submit_magi1_frames.sh 2>&1 | tee logs/jupiter/submit_magi1_frames_$(date +%Y%m%d_%H%M).txt
set -euo pipefail

ONLY=${ONLY:-"baseline thesis30k"}
want() { [[ " $ONLY " == *" $1 "* ]]; }

W=/e/scratch/e-dev-2026d09-262/wh_work
export CLAMP_PREDICTION=1

if want baseline; then
    VAE_RUN=finetune_vae_jupiter_baseline_s44_256res
    echo "baseline: $(SCORE_FRAMES="10 20 30 40 50 60 70 80 90" sbatch --parsable jobs/jupiter/eval_dit_vrmse.sbatch \
        ${W}/dit_preprocessed/${VAE_RUN} \
        ${W}/dit_outputs/shockwave_dit_jupiter_baseline_s44_lr1e4/checkpoints/model_weights_step_08000.safetensors \
        ${W}/finetune_vae_outputs/${VAE_RUN}/checkpoints/vae_shockwave_best.safetensors)"
fi
if want thesis30k; then
    VAE_RUN=finetune_vae_ch6_loss_rmse_h1_256res
    echo "thesis30k: $(SCORE_FRAMES="10 20 30 40 50 60 70 80 90 100" NUM_INFERENCE_STEPS=20 sbatch --parsable \
        jobs/jupiter/eval_dit_vrmse.sbatch \
        ${W}/dit_preprocessed/${VAE_RUN} \
        ${W}/dit_outputs/shockwave_dit_jupiter_ch6_loss_rmse_h1_30k/checkpoints/model_weights_step_30000.safetensors \
        ${W}/finetune_vae_outputs/${VAE_RUN}/checkpoints/vae_shockwave_best.safetensors)"
fi
true
