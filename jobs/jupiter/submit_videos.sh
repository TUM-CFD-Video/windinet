#!/bin/bash
# GT/Prediction/Residual videos (2026-10-19, Harish) of two test.h5 trajectories,
# 0000_gamma1.1299999952 and 0456_gamma1.7599999905, from the cfd_bench model:
# baseline s44 LR 1e-4 20k (2250009, 0.4215; cfd_bench job 2413329). Also scored
# like the cfd_bench run (frames 40..90, clamp), so the per-sim benchmark block
# in vrmse_summary.json goes with the videos.
# Result: logs/jupiter/eval_dit_vrmse/<job>/visualizations/{vae_only,vae_dit}/<sim>/video.mp4
# (vae_dit = the DiT rollout; vae_only = the VAE reconstruction, for reference).
#
# Run from the repo root on a jupiter login node, windinet env active:
#   bash jobs/jupiter/submit_videos.sh 2>&1 | tee logs/jupiter/submit_videos_$(date +%Y%m%d_%H%M).txt
set -euo pipefail

W=/e/scratch/e-dev-2026d09-262/wh_work
VAE_RUN=finetune_vae_jupiter_baseline_s44_256res

echo "videos: $(SIM_IDS="0000_gamma1.1299999952 0456_gamma1.7599999905" SAVE_VIDEO=1 \
    SCORE_FRAMES="40 50 60 70 80 90" CLAMP_PREDICTION=1 \
    sbatch --parsable --time=00:30:00 jobs/jupiter/eval_dit_vrmse.sbatch \
    ${W}/dit_preprocessed/${VAE_RUN} \
    ${W}/dit_outputs/shockwave_dit_jupiter_baseline_s44_lr1e4_20k/checkpoints/model_weights_step_20000.safetensors \
    ${W}/finetune_vae_outputs/${VAE_RUN}/checkpoints/vae_shockwave_best.safetensors)"
