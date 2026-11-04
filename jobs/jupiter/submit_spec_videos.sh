#!/bin/bash
# Standardized rollout videos (docs/video_spec.md, 2026-11-04) of the baseline
# cond9 (VAE s44 + DiT LR 1e-4 8k, conditioned on frames 0-8) on Harish's two
# test.h5 trajectories, 0000_gamma1.1299999952 and 0456_gamma1.7599999905:
#   1. video_stats.sbatch: shared training statistics
#      -> configs/video/euler_mq_256_video_stats.json (commit it; skipped if present)
#   2. eval_dit_vrmse.sbatch with SAVE_ROLLOUTS + VIDEO_STATS, after 1:
#      physical-unit rollouts on scratch, then the videos
# Result: logs/jupiter/eval_dit_vrmse/<job>/spec_videos/
#   cond9_<sim>.mp4 (+ .json per-frame RMSE / fractions, _t<i>.png),
#   cond9-vae_only_<sim>.mp4 (VAE reconstruction, for reference)
#
# Run from the repo root on a jupiter login node, windinet env active:
#   bash jobs/jupiter/submit_spec_videos.sh
set -euo pipefail

W=/e/scratch/e-dev-2026d09-262/wh_work
VAE_RUN=finetune_vae_jupiter_baseline_s44_256res
STATS=configs/video/euler_mq_256_video_stats.json

DEP=()
if [[ ! -f "${STATS}" ]]; then
    STATS_JOB=$(sbatch --parsable jobs/jupiter/video_stats.sbatch)
    echo "video_stats: ${STATS_JOB}"
    DEP=(--dependency=afterok:${STATS_JOB})
fi

echo "cond9 videos: $(SIM_IDS="0000_gamma1.1299999952 0456_gamma1.7599999905" \
    SAVE_ROLLOUTS=2 VIDEO_STATS="${STATS}" VIDEO_MODEL=cond9 SAVE_NPZ_SAMPLES=0 SAVE_VIS_SAMPLES=0 \
    sbatch --parsable --time=00:45:00 ${DEP[@]+"${DEP[@]}"} jobs/jupiter/eval_dit_vrmse.sbatch \
    ${W}/dit_preprocessed/${VAE_RUN} \
    ${W}/dit_outputs/shockwave_dit_jupiter_baseline_s44_lr1e4_cond9/checkpoints/model_weights_step_08000.safetensors \
    ${W}/finetune_vae_outputs/${VAE_RUN}/checkpoints/vae_shockwave_best.safetensors)"
