#!/bin/bash
# Fixed-noise test, step 0 (2026-11-12, rbischof 11-09 / Harish: "fixing the noise seed seems like a
# good experiment"): the trained cond9 DiT (VAE s44 + DiT LR 1e-4 8k, frames 0-8 conditioning) sampled
# from all-zero initial latents instead of N(0, I) -- inference only, no retraining. The rollout is then
# deterministic. Caveat: the DiT was trained on random noise, so a zero start is off-distribution
# (|eps| ~ sqrt(d), not 0); if this helps, the trained version (noise_mode zeros) is the next step.
#   test5 / test20  test.h5, 500 sims, 5 / 20 sampler steps. Reference: random start, 5 steps =
#                   job 2634285 (ch-mean 0.3516, benchmark 40-90 VRMSE 0.576); compare H1/H2 too
#                   (sharpness).
#   vid5 / vid50    trajectory 0456 alone, 5 / 50 steps, spec videos -- side by side with the
#                   random-start videos of submit_spec_videos.sh. Need the video stats json.
# ONLY=<names> (space-separated) submits a subset.
#
# Run from the repo root on a jupiter login node, windinet env active:
#   bash jobs/jupiter/submit_init_noise_zeros.sh
set -euo pipefail

ONLY=${ONLY:-"test5 test20 vid5 vid50"}
want() { [[ " $ONLY " == *" $1 "* ]]; }

W=/e/scratch/e-dev-2026d09-262/wh_work
VAE_RUN=finetune_vae_jupiter_baseline_s44_256res
STATS=configs/video/euler_mq_256_video_stats.json
ARGS=(
    ${W}/dit_preprocessed/${VAE_RUN}
    ${W}/dit_outputs/shockwave_dit_jupiter_baseline_s44_lr1e4_cond9/checkpoints/model_weights_step_08000.safetensors
    ${W}/finetune_vae_outputs/${VAE_RUN}/checkpoints/vae_shockwave_best.safetensors
)

for s in 5 20; do
    want "test${s}" || continue
    echo "test${s}: $(INIT_NOISE=zeros NUM_INFERENCE_STEPS=${s} TABLE_NAME="cond9 zero-noise ${s} steps" \
        sbatch --parsable --job-name=wh_zeros_test${s} jobs/jupiter/eval_dit_vrmse.sbatch "${ARGS[@]}")"
done
for s in 5 50; do
    want "vid${s}" || continue
    if [[ ! -f "${STATS}" ]]; then
        echo "vid${s}: skipped, ${STATS} missing (run submit_spec_videos.sh first, commit its stats)" >&2
        continue
    fi
    echo "vid${s}: $(INIT_NOISE=zeros NUM_INFERENCE_STEPS=${s} SIM_IDS=0456_gamma1.7599999905 \
        SAVE_ROLLOUTS=1 VIDEO_STATS="${STATS}" VIDEO_MODEL="cond9_zeros_${s}steps" \
        TABLE_NAME="cond9_zeros_${s}steps_0456" SAVE_NPZ_SAMPLES=0 SAVE_VIS_SAMPLES=0 \
        sbatch --parsable --time=01:00:00 --job-name=wh_zeros_vid${s} jobs/jupiter/eval_dit_vrmse.sbatch "${ARGS[@]}")"
done
true
