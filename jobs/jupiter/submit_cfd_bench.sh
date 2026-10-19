#!/bin/bash
# cfd_bench comparison (2026-10-19, Harish): score the DiT on the frames his
# autoregressive baseline predicts -- 4 input frames at stride 10, so frames
# 40, 50, ..., 100 of 256x256_ds/test.h5 -- with RMSE / VRMSE (per field, then
# mean) / 1-SSIM / MLW and the prediction clamped to [-1, 1] like the target.
# Result: the "benchmark" block of logs/jupiter/eval_dit_vrmse/<job>/vrmse_summary.json.
#   thesis30k  thesis-VAE ch6_loss_rmse_h1 + DiT 30k (2185278, ch-mean 0.4298): 101-frame
#              sims, so frame 100 exists -- the exact cfd_bench frames.
#   base20k    baseline s44 LR 1e-4 20k (2250009, 0.4215): 97-frame sims end at frame 96,
#              so it is scored on 40..90 only -- not the same frame set, say so when comparing.
# 2 jobs. ONLY=<names> (space-separated, e.g. ONLY="thesis30k") submits a subset.
#
# Run from the repo root on a jupiter login node, windinet env active:
#   bash jobs/jupiter/submit_cfd_bench.sh 2>&1 | tee logs/jupiter/submit_cfd_bench_$(date +%Y%m%d_%H%M).txt
set -euo pipefail

ONLY=${ONLY:-"thesis30k base20k"}
want() { [[ " $ONLY " == *" $1 "* ]]; }

W=/e/scratch/e-dev-2026d09-262/wh_work
export CLAMP_PREDICTION=1

if want thesis30k; then
    VAE_RUN=finetune_vae_ch6_loss_rmse_h1_256res
    echo "thesis30k: $(SCORE_FRAMES="40 50 60 70 80 90 100" sbatch --parsable jobs/jupiter/eval_dit_vrmse.sbatch \
        ${W}/dit_preprocessed/${VAE_RUN} \
        ${W}/dit_outputs/shockwave_dit_jupiter_ch6_loss_rmse_h1_30k/checkpoints/model_weights_step_30000.safetensors \
        ${W}/finetune_vae_outputs/${VAE_RUN}/checkpoints/vae_shockwave_best.safetensors)"
fi
if want base20k; then
    VAE_RUN=finetune_vae_jupiter_baseline_s44_256res
    echo "base20k: $(SCORE_FRAMES="40 50 60 70 80 90" sbatch --parsable jobs/jupiter/eval_dit_vrmse.sbatch \
        ${W}/dit_preprocessed/${VAE_RUN} \
        ${W}/dit_outputs/shockwave_dit_jupiter_baseline_s44_lr1e4_20k/checkpoints/model_weights_step_20000.safetensors \
        ${W}/finetune_vae_outputs/${VAE_RUN}/checkpoints/vae_shockwave_best.safetensors)"
fi
true
