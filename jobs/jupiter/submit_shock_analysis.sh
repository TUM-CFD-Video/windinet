#!/bin/bash
# Eval-only re-runs of the new baseline cond9 (VAE s44 + DiT LR 1e-4 8k,
# conditioned on frames 0-8; baseline from 2026-11-02) and of the old
# frame-0 baseline it replaces, with everything eval_dit_vrmse.py gained
# since their first evals (test.h5, 500 sims, 5 steps):
#   - error by region (shock / shock band / contact / smooth) and per latent
#     token (2026-11-01, --shock_analysis, SHOCK_ANALYSIS=1)
#   - benchmark_10 block, every 10th frame from 10 (Harish, 2026-11-02)
#   - relative H1/H2 seminorm errors (Lars, dev-lars/bench-seminorm)
# Results: logs/jupiter/eval_dit_vrmse/<job>/vrmse_summary.json ("benchmark",
# "benchmark_10", "shock_analysis"), .../shock_analysis/token_heatmaps.png;
# per-sim tokens on scratch (figure_data/dit/<run>/shock_tokens_<job>.npz).
# ONLY=<names> (space-separated, e.g. ONLY="cond9") submits a subset.
#
# Run from the repo root on a jupiter login node, windinet env active:
#   bash jobs/jupiter/submit_shock_analysis.sh
set -euo pipefail

ONLY=${ONLY:-"cond9 baseline_f1"}
want() { [[ " $ONLY " == *" $1 "* ]]; }

W=/e/scratch/e-dev-2026d09-262/wh_work
VAE_RUN=finetune_vae_jupiter_baseline_s44_256res
submit() {  # <name> <DiT run> <table name>
    echo "$1: $(SHOCK_ANALYSIS=1 TABLE_NAME="$3" \
        sbatch --parsable jobs/jupiter/eval_dit_vrmse.sbatch \
        ${W}/dit_preprocessed/${VAE_RUN} \
        ${W}/dit_outputs/$2/checkpoints/model_weights_step_08000.safetensors \
        ${W}/finetune_vae_outputs/${VAE_RUN}/checkpoints/vae_shockwave_best.safetensors)"
}
want cond9 && submit cond9 shockwave_dit_jupiter_baseline_s44_lr1e4_cond9 "256: baseline cond9 (frames 0-8) + DiT 8k"
want baseline_f1 && submit baseline_f1 shockwave_dit_jupiter_baseline_s44_lr1e4 "256: old baseline (frame 0) + DiT 8k"
