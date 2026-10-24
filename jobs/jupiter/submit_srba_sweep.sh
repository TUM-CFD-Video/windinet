#!/bin/bash
# 2D-SRBA-OOOO hyperparameter sweep (2026-10-24) around the SRBA baseline
# (submit_srba_baseline.sh), one change per arm:
#   VAE arms, full chain (VAE -> vaetest | encode -> DiT baseline recipe -> diteval),
#   5 jobs each: lr1e4, lr2e5 (baseline 5e-5), ep150, ep600 (baseline 300 epochs;
#   ep600 gets VAE_TIME=12:00:00).
#   DiT arms on the baseline VAE's latents (DiT + diteval), 2 jobs each:
#   dit_lr5e5, dit_lr2e4 (baseline 1e-4), dit_4k (baseline 8000 steps).
# The baseline latents (dit_preprocessed/finetune_vae_jupiter_srba_baseline_256res)
# do not exist until the baseline's encode job finishes: pass its job id as
# AFTER=<jobid> (squeue --me) to chain the DiT arms on it; without AFTER they
# need the latents to exist already.
# ONLY=<names> (space-separated, e.g. ONLY="lr1e4 dit_4k") submits a subset.
#
# Run from the repo root on a jupiter login node, windinet env active:
#   AFTER=<baseline encode jobid> bash jobs/jupiter/submit_srba_sweep.sh 2>&1 | tee logs/jupiter/submit_srba_sweep_$(date +%Y%m%d_%H%M).txt
set -euo pipefail

ONLY=${ONLY:-"lr1e4 lr2e5 ep150 ep600 dit_lr5e5 dit_lr2e4 dit_4k"}
want() { [[ " $ONLY " == *" $1 "* ]]; }

for arm in lr1e4 lr2e5 ep150 ep600; do
    want "$arm" || continue
    time=""
    [[ "$arm" == "ep600" ]] && time="12:00:00"
    AFTER="" VAE_TIME="$time" bash jobs/jupiter/submit_pipeline.sh \
        configs/finetune_vae/finetune_vae_jupiter_srba_${arm}_256res.yaml \
        configs/dit/train_dit_jupiter_srba_${arm}.yaml
done

for arm in dit_lr5e5 dit_lr2e4 dit_4k; do
    want "$arm" || continue
    SKIP_ENCODE=1 bash jobs/jupiter/submit_pipeline.sh \
        configs/finetune_vae/finetune_vae_jupiter_srba_baseline_256res.yaml \
        configs/dit/train_dit_jupiter_srba_${arm}.yaml
done
