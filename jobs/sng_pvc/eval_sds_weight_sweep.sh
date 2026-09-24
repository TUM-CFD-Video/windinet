#!/bin/bash
# eval_dit_vrmse submissions for the 256res SDS-weight sweep's three DiT arms
# (w=1e-4 / 1e-2 / 1), which all finished their full 8000-step runs on
# 2026-09-24 (jobs 541735/541736/541737, ~19.5h each, clean exit -- see
# logs/sng_pvc/54173{5,6,7}-shockwave_dit.out). Same pattern as
# jobs/sng_pvc/eval_sds_dit_arm.sh, one sbatch per arm.
#
# Paths come from logs/sng_pvc/INDEX.tsv (VAE finetune 539633-539635,
# preprocess 541001-541003, DiT train 541735-541737) plus the last "saved"
# line in each DiT log (model_weights_step_08033.safetensors;
# checkpoints.keep_last_n=2, so only 07920 and 08033 survive). The existence
# check below refuses to submit an arm whose checkpoint/VAE/preprocessed dir
# is missing, instead of letting the job die on the compute node.
#
# NUM_SAMPLES=675 (all held-out sims), matching the other full eval_dit_vrmse
# runs so the numbers are directly comparable to them.
#
# Run from the repo root on sng_pvc (/dss/dsshome1/0D/go76fuz2/windinet).
set -euo pipefail

SCRATCH_ROOT=/hppfs/scratch/0D/go76fuz2/windinet
STEP=08033
NUM_SAMPLES=675

for W in w0p0001 w0p01 w1; do
    NAME=finetune_vae_whole_structure_baseline_ep20_256res_sds_${W}
    PRE="${SCRATCH_ROOT}/dit_preprocessed/${NAME}"
    DIT="${SCRATCH_ROOT}/outputs/shockwave_dit_sds_${W}/checkpoints/model_weights_step_${STEP}.safetensors"
    VAE="${SCRATCH_ROOT}/finetune_vae_outputs_sng_pvc/${NAME}/checkpoints/vae_shockwave_best.safetensors"

    missing=0
    for p in "$PRE" "$DIT" "$VAE"; do
        [ -e "$p" ] || { echo "[${W}] MISSING: $p" >&2; missing=1; }
    done
    if [ "$missing" -ne 0 ]; then
        echo "[${W}] skipped" >&2
        continue
    fi

    echo "[${W}] submitting"
    sbatch jobs/sng_pvc/eval_dit_vrmse.sbatch "$PRE" "$DIT" "$VAE" "$NUM_SAMPLES"
done
