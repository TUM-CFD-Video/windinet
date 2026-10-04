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
#   VAE_RESUMES / ENCODE_RESUMES / DIT_RESUMES=<n>  (default 0) queue n resume
#                  jobs after that stage's first job, each afterany on the previous
#                  one. The trainers resume from the newest checkpoint and the
#                  encoder skips latents already written; all exit 0 at once
#                  when the stage is already complete, so surplus resumes cost
#                  ~a minute; downstream stages wait (afterok) on the LAST job,
#                  which only exits 0 once the stage has really finished.
#   VAE_TIME / VAETEST_TIME / ENCODE_TIME / DIT_TIME / EVAL_TIME=<HH:MM:SS>
#                  override that stage's sbatch --time (defaults: the scripts'
#                  own #SBATCH lines, sized for 256x256)
#   BATCH_SIZE=<n> per-GPU VAE batch (finetune_vae.sbatch's own default: 4);
#                  gradient accumulation is derived to keep effective batch 32
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
VAE_RESUMES=${VAE_RESUMES:-0}
ENCODE_RESUMES=${ENCODE_RESUMES:-0}
DIT_RESUMES=${DIT_RESUMES:-0}
t() { if [ -n "${1:-}" ]; then echo "--time=$1"; fi; }   # optional --time override

# chain N TIME SCRIPT ARGS... -- first job with ${dep[@]}, then N resume jobs,
# each afterany on the previous; prints every id to stderr, the last to stdout.
chain() {
    local n=$1 time=$2 id i
    shift 2
    id=$(sbatch --parsable ${dep[@]+"${dep[@]}"} $(t "$time") "$@")
    echo "  ${id}" >&2
    for i in $(seq 1 "$n"); do
        id=$(sbatch --parsable --dependency="afterany:${id}" $(t "$time") "$@")
        echo "  ${id} (resume ${i})" >&2
    done
    echo "$id"
}

if [[ "$SKIP_VAE" == "1" ]]; then
    if [ -z "$AFTER" ] && [ ! -e "${VAE_DIR}/.train_complete" ]; then
        echo "error: SKIP_VAE=1 but ${VAE_DIR}/.train_complete does not exist" >&2; exit 1
    fi
else
    if [ -e "${VAE_DIR}/.train_complete" ]; then
        echo "error: ${VAE_DIR} is already complete; use SKIP_VAE=1" >&2; exit 1
    fi
    echo "vae (+${VAE_RESUMES} resumes):"
    vae_id=$(chain "$VAE_RESUMES" "${VAE_TIME:-}" jobs/jupiter/finetune_vae.sbatch "$VAE_CONFIG")
    echo "vae: ${vae_id}"
    echo "vaetest: $(sbatch --parsable --dependency="afterok:${vae_id}" $(t "${VAETEST_TIME:-}") jobs/jupiter/eval_vae_test.sbatch "$VAE_CONFIG")"
    dep=(--dependency="afterok:${vae_id}")
fi

if [[ "$SKIP_ENCODE" == "1" ]]; then
    if [ -z "$AFTER" ] && [ ! -e "${PRE_ROOT}/normalization.json" ]; then
        echo "error: SKIP_ENCODE=1 but ${PRE_ROOT}/normalization.json does not exist" >&2; exit 1
    fi
else
    echo "encode (+${ENCODE_RESUMES} resumes):"
    pre_id=$(EVAL_SIMS=500 VAE_CHECKPOINT="$VAE_CKPT" \
        chain "$ENCODE_RESUMES" "${ENCODE_TIME:-}" jobs/jupiter/preprocess_dit_data.sbatch "$VAE_RUN")
    echo "encode: ${pre_id}"
    dep=(--dependency="afterok:${pre_id}")
fi

echo "dit (+${DIT_RESUMES} resumes):"
dit_id=$(chain "$DIT_RESUMES" "${DIT_TIME:-}" jobs/jupiter/train_dit.sbatch "$VAE_RUN" "$DIT_CONFIG")
echo "dit: ${dit_id}"

echo "diteval: $(sbatch --parsable --dependency="afterok:${dit_id}" $(t "${EVAL_TIME:-}") \
    jobs/jupiter/eval_dit_vrmse.sbatch "$PRE_ROOT" "$DIT_CKPT_DIR" "$VAE_CKPT")"
