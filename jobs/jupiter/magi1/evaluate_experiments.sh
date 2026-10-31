#!/bin/bash
# Evaluate a whole set of experiments in one go (meeting item 7b). Run on the login node from the repo
# root; it only calls sbatch (one evaluate_magi1.sbatch job per experiment, all in parallel).
#
# Meta folder layout (default: /e/scratch/e-dev-2026d09-262/ig_work/magi1_final = settings.FINAL_MODELS,
# which every completed evaluation fills automatically, see evaluate.py archive_run):
#   <meta>/<name>/vae/<checkpoint>.safetensors    required (vae_best.safetensors if there are several)
#   <meta>/<name>/dit/<checkpoint>.safetensors    optional (the last one by name = highest step);
#                                                 without it the VAE alone is evaluated (--vae-only)
# Results: $MAGI1_OUTPUTS/experiments/<name>/eval/ (and copied back to <meta>/<name>/eval).
# Experiments that already have results.md there are skipped unless --force.
#
# Usage:
#   bash jobs/jupiter/magi1/evaluate_experiments.sh [META_DIR] [--force] [--dry-run] [-- <evaluate.py args>]
# e.g.
#   bash jobs/jupiter/magi1/evaluate_experiments.sh --dry-run
#   bash jobs/jupiter/magi1/evaluate_experiments.sh -- --metric-sims 100 --num-steps 32

set -euo pipefail

META_DIR="/e/scratch/e-dev-2026d09-262/ig_work/magi1_final"
FORCE=0
DRY_RUN=0
EVAL_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --force)   FORCE=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        --)        shift; EVAL_ARGS=("$@"); break ;;
        -*)        echo "unknown option: $1 (see the header of $0)" >&2; exit 1 ;;
        *)         META_DIR="$1"; shift ;;
    esac
done

[[ -f jobs/jupiter/magi1/magi1_env.sh ]] || { echo "run this from the repo root" >&2; exit 1; }
source jobs/jupiter/magi1/magi1_paths.sh   # MAGI1_OUTPUTS (on SCRATCH)
[[ -d "${META_DIR}" ]] || { echo "no meta folder ${META_DIR}" >&2; exit 1; }

pick_checkpoint() {  # pick_checkpoint <folder> -> vae_best.safetensors if present, else the last by name
    local folder="$1"
    [[ -d "${folder}" ]] || return 0
    if [[ -f "${folder}/vae_best.safetensors" ]]; then
        echo "${folder}/vae_best.safetensors"
    else
        find "${folder}" -maxdepth 1 -name "*.safetensors" | sort | tail -n 1
    fi
}

for experiment in "${META_DIR}"/*/; do
    name=$(basename "${experiment}")
    vae=$(pick_checkpoint "${experiment}vae")
    dit=$(pick_checkpoint "${experiment}dit")
    out="${MAGI1_OUTPUTS}/experiments/${name}/eval"
    if [[ -z "${vae}" ]]; then
        echo "skip ${name}: no VAE checkpoint in ${experiment}vae"; continue
    fi
    if [[ -f "${out}/results.md" && ${FORCE} -eq 0 ]]; then
        echo "skip ${name}: ${out}/results.md exists (--force to redo)"; continue
    fi

    model_args=(--vae-ckpt "${vae}")
    if [[ -n "${dit}" ]]; then model_args+=(--dit-ckpt "${dit}"); else model_args+=(--vae-only); fi
    cmd=(sbatch --parsable --job-name="ig_eval_${name}" jobs/jupiter/magi1/evaluate_magi1.sbatch
         "${model_args[@]}" --out "${out}" --archive-name "${name}" "${EVAL_ARGS[@]}")
    if (( DRY_RUN )); then
        echo "${cmd[*]}"
    else
        echo "${name}: job $("${cmd[@]}") -> ${out}/results.md"
    fi
done
(( DRY_RUN )) && echo "(dry run: nothing submitted)"
exit 0
