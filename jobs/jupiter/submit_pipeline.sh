#!/bin/bash
# One VAE + DiT experiment on jupiter, end to end: one sbatch per stage, every
# stage on 1 node x 4 GPUs, chained with afterok dependencies.
#
#   VAE      finetune_vae.sbatch         <VAE_CONFIG>
#   vaetest  eval_vae_test.sbatch        afterok VAE    (test.h5, 500 sims)
#   encode   preprocess_dit_data.sbatch  afterok VAE    (EVAL_SIMS=500; frames and
#                                                        channels follow the VAE run)
#   DiT      train_dit.sbatch            afterok encode (<DIT_CONFIG>)
#   diteval  eval_dit_vrmse.sbatch       afterok DiT    (test.h5, 500 sims)
#
# Run names and paths come from the configs' own output_dir, so any pair of
# configs works. Each training stage must finish inside its own job (VAE
# ~3.5h of 6h; DiT ~6.7h of 12h at the ~0.33 steps/s measured on jupiter).
# If a training job hits its walltime, its dependents stay PENDING
# (DependencyNeverSatisfied): resubmit that stage's sbatch with the same
# arguments -- both launchers resume from the newest checkpoint -- then
# scancel the stale dependents and chain the rest by hand.
#
# Env options:
#   SKIP_VAE=1     the VAE is already trained: skip VAE and vaetest
#   SKIP_ENCODE=1  its latents already exist (implies SKIP_VAE): DiT + diteval only,
#                  e.g. a second DiT config on the same VAE
#   AFTER=<jobid>  make the first submitted job wait (afterok) on that job, e.g.
#                  the encode job of a still-running VAE; skips the "already
#                  exists" checks, since the files only appear once it finishes
#
# Prints one "<stage>: <jobid>" line per submitted job.
#
# Run from the repo root on a jupiter login node, windinet env active:
#   bash jobs/jupiter/submit_pipeline.sh <VAE_CONFIG> <DIT_CONFIG>
set -euo pipefail

VAE_CONFIG=${1:?usage: bash jobs/jupiter/submit_pipeline.sh <VAE_CONFIG> <DIT_CONFIG>}
DIT_CONFIG=${2:?usage: bash jobs/jupiter/submit_pipeline.sh <VAE_CONFIG> <DIT_CONFIG>}
SKIP_ENCODE=${SKIP_ENCODE:-0}
SKIP_VAE=${SKIP_VAE:-${SKIP_ENCODE}}
AFTER=${AFTER:-}
W=/e/scratch/e-dev-2026d09-262/wh_work

for f in "$VAE_CONFIG" "$DIT_CONFIG"; do
    [ -e "$f" ] || { echo "error: missing $f" >&2; exit 1; }
done
output_dir() { python -c "import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))['output_dir'])" "$1"; }
VAE_RUN=$(basename "$(output_dir "$VAE_CONFIG")")
VAE_DIR=${W}/finetune_vae_outputs/${VAE_RUN}
VAE_CKPT=${VAE_DIR}/checkpoints/vae_shockwave_best.safetensors
PRE_ROOT=${W}/dit_preprocessed/${VAE_RUN}
DIT_CKPT_DIR=$(output_dir "$DIT_CONFIG")/checkpoints
echo "# ${VAE_RUN} -> $(basename "$(dirname "$DIT_CKPT_DIR")")"

mkdir -p logs/jupiter
dep=()
[ -n "$AFTER" ] && dep=(--dependency="afterok:${AFTER}")

if [[ "$SKIP_VAE" == "1" ]]; then
    if [ -z "$AFTER" ] && [ ! -e "${VAE_DIR}/.train_complete" ]; then
        echo "error: SKIP_VAE=1 but ${VAE_DIR}/.train_complete does not exist" >&2; exit 1
    fi
else
    if [ -e "${VAE_DIR}/.train_complete" ]; then
        echo "error: ${VAE_DIR} is already complete; use SKIP_VAE=1" >&2; exit 1
    fi
    vae_id=$(sbatch --parsable ${dep[@]+"${dep[@]}"} jobs/jupiter/finetune_vae.sbatch "$VAE_CONFIG")
    echo "vae: ${vae_id}"
    echo "vaetest: $(sbatch --parsable --dependency="afterok:${vae_id}" jobs/jupiter/eval_vae_test.sbatch "$VAE_CONFIG")"
    dep=(--dependency="afterok:${vae_id}")
fi

if [[ "$SKIP_ENCODE" == "1" ]]; then
    if [ -z "$AFTER" ] && [ ! -e "${PRE_ROOT}/normalization.json" ]; then
        echo "error: SKIP_ENCODE=1 but ${PRE_ROOT}/normalization.json does not exist" >&2; exit 1
    fi
else
    pre_id=$(EVAL_SIMS=500 VAE_CHECKPOINT="$VAE_CKPT" \
        sbatch --parsable ${dep[@]+"${dep[@]}"} jobs/jupiter/preprocess_dit_data.sbatch "$VAE_RUN")
    echo "encode: ${pre_id}"
    dep=(--dependency="afterok:${pre_id}")
fi

dit_id=$(sbatch --parsable ${dep[@]+"${dep[@]}"} jobs/jupiter/train_dit.sbatch "$VAE_RUN" "$DIT_CONFIG")
echo "dit: ${dit_id}"

echo "diteval: $(sbatch --parsable --dependency="afterok:${dit_id}" \
    jobs/jupiter/eval_dit_vrmse.sbatch "$PRE_ROOT" "$DIT_CKPT_DIR" "$VAE_CKPT")"
