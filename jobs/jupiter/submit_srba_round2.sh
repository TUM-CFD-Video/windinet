#!/bin/bash
# 2D-SRBA-OOOO DiT round 2 (2026-10-25), all on the SRBA baseline VAE's latents
# (dit_preprocessed/finetune_vae_jupiter_srba_baseline_256res, already encoded),
# DiT + diteval only. Round 1 (submit_srba_sweep.sh): DiT LR 5e-5 best
# (0.1320 vs baseline 0.1348), 4k steps much worse (0.1738), VAE arms no e2e gain.
#   dit_lr5e5_16k  LR 5e-5, 16000 steps        (main arm)
#   dit_16k        LR 1e-4, 16000 steps        (step effect alone)
#   dit_lr5e5_s43  LR 5e-5, 8000 steps, seed 43 (seed replicate of dit_lr5e5)
# 8k steps took ~6.5h on one booster node, so the 16k arms get one resume job
# (12h limit); a queued resume job is harmless when the first job finished in time.
# ONLY=<names> (space-separated, e.g. ONLY="dit_16k") submits a subset.
#
# Run from the repo root on a jupiter login node, windinet env active:
#   bash jobs/jupiter/submit_srba_round2.sh 2>&1 | tee logs/jupiter/submit_srba_round2_$(date +%Y%m%d_%H%M).txt
set -euo pipefail

ONLY=${ONLY:-"dit_lr5e5_16k dit_16k dit_lr5e5_s43"}
want() { [[ " $ONLY " == *" $1 "* ]]; }

for arm in dit_lr5e5_16k dit_16k dit_lr5e5_s43; do
    want "$arm" || continue
    resumes=0
    [[ "$arm" == *_16k ]] && resumes=1
    SKIP_ENCODE=1 DIT_RESUMES=$resumes bash jobs/jupiter/submit_pipeline.sh \
        configs/finetune_vae/finetune_vae_jupiter_srba_baseline_256res.yaml \
        configs/dit/train_dit_jupiter_srba_${arm}.yaml
done
