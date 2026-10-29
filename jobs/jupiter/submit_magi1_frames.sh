#!/bin/bash
# MAGI-1 comparison (2026-10-29, Harish): the baseline DiT, which is conditioned on
# frame 0 only (not on input frames 0/10/20/30 like cfd_bench / MAGI-1), scored on
# frames 10, 20, ..., 90 of 256x256_ds/test.h5. 97-frame sims end at frame 96, so
# no frame 100. Prediction and target clamped to [-1, 1]; 5 sampler steps (config default).
# Result: the "benchmark" block of logs/jupiter/eval_dit_vrmse/<job>/vrmse_summary.json --
# mean over frames 10..90 (rmse/vrmse/ssim/mlw) and benchmark.per_frame (rmse/vrmse/ssim per frame).
#
# Run from the repo root on a jupiter login node, windinet env active:
#   bash jobs/jupiter/submit_magi1_frames.sh 2>&1 | tee logs/jupiter/submit_magi1_frames_$(date +%Y%m%d_%H%M).txt
set -euo pipefail

W=/e/scratch/e-dev-2026d09-262/wh_work
VAE_RUN=finetune_vae_jupiter_baseline_s44_256res
export CLAMP_PREDICTION=1

echo "baseline: $(SCORE_FRAMES="10 20 30 40 50 60 70 80 90" sbatch --parsable jobs/jupiter/eval_dit_vrmse.sbatch \
    ${W}/dit_preprocessed/${VAE_RUN} \
    ${W}/dit_outputs/shockwave_dit_jupiter_baseline_s44_lr1e4/checkpoints/model_weights_step_08000.safetensors \
    ${W}/finetune_vae_outputs/${VAE_RUN}/checkpoints/vae_shockwave_best.safetensors)"
