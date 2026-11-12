#!/bin/bash
# Harish's requests (WhatsApp, 2026-11-04..11) on the baseline cond9 (VAE s44 + DiT LR 1e-4 8k,
# conditioned on frames 0-8), videos per docs/video_spec.md (Lars's video-spec branch: |error| in
# magma, hi_c = min(mu + 2 sigma, max)):
#   1. standardized videos of test.h5 trajectories 0000_gamma1.1299999952 and 0456_gamma1.7599999905
#   2. sampler steps 5 / 20 / 50 on 0456: are more denoising steps sharper but "wronger"?
#      (videos side by side + VRMSE and relative H1/H2 per step count)
#   3. the full metric suite on the single trajectory 0456 (best model = cond9 at 5 steps)
# Each 0456 job evaluates only that sim (SIM_IDS), so its vrmse_summary.json / benchmark_table.md
# are single-trajectory numbers. 0000 gets its own job so it never mixes into the 0456 metrics.
#
#   0. video_stats.sbatch: shared training statistics
#      -> configs/video/euler_mq_256_video_stats.json (commit it; skipped if present)
#   then 4 eval jobs (after 0): 0456 at 5, 20, 50 steps; 0000 at 5 steps
# Result: logs/jupiter/eval_dit_vrmse/<job>/{vrmse_summary.json, benchmark_table.md, spec_videos/}
#   spec_videos/cond9_<n>steps_<sim>.mp4 (+ .json per-frame RMSE, _t<i>.png),
#   cond9_<n>steps-vae_only_<sim>.mp4 (VAE reconstruction, for reference)
#
# Run from the repo root on a jupiter login node, windinet env active:
#   bash jobs/jupiter/submit_spec_videos.sh
# STEPS="5 50" to change the step counts; VIDEO_SIGNED_ERROR=1 for signed error panels.
set -euo pipefail

W=/e/scratch/e-dev-2026d09-262/wh_work
VAE_RUN=finetune_vae_jupiter_baseline_s44_256res
DIT_RUN=shockwave_dit_jupiter_baseline_s44_lr1e4_cond9
STATS=configs/video/euler_mq_256_video_stats.json
SIM_0456=0456_gamma1.7599999905
SIM_0000=0000_gamma1.1299999952
STEPS=${STEPS:-"5 20 50"}

DEP=()
if [[ ! -f "${STATS}" ]]; then
    STATS_JOB=$(sbatch --parsable jobs/jupiter/video_stats.sbatch)
    echo "video_stats: ${STATS_JOB}"
    DEP=(--dependency=afterok:${STATS_JOB})
fi

ARGS=(
    ${W}/dit_preprocessed/${VAE_RUN}
    ${W}/dit_outputs/${DIT_RUN}/checkpoints/model_weights_step_08000.safetensors
    ${W}/finetune_vae_outputs/${VAE_RUN}/checkpoints/vae_shockwave_best.safetensors
)

submit() {  # <sim> <steps>
    local name=cond9_${2}steps
    echo "${name} ${1}: $(SIM_IDS="${1}" NUM_INFERENCE_STEPS="${2}" \
        SAVE_ROLLOUTS=1 VIDEO_STATS="${STATS}" VIDEO_MODEL="${name}" TABLE_NAME="${name}_${1}" \
        SAVE_NPZ_SAMPLES=0 SAVE_VIS_SAMPLES=0 \
        sbatch --parsable --time=01:00:00 --job-name="vid_${name}_${1:0:4}" ${DEP[@]+"${DEP[@]}"} \
        jobs/jupiter/eval_dit_vrmse.sbatch "${ARGS[@]}")"
}

for s in ${STEPS}; do
    submit "${SIM_0456}" "${s}"
done
submit "${SIM_0000}" 5
