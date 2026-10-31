#!/bin/bash
# Randomized VAE hyperparameter sweep (magi_finetuning/utils/vae_sweep.py): N random configurations + 2 anchors
# (official recipe, experiment E), each a short complete training (5 epochs) on 1 node x 4 GH200, all in
# parallel; when all are done, a small job writes the results matrix. Run on the login node from the repo root.
#
#   bash jobs/jupiter/magi1/magi1_vae_sweep.sh --name <sweep> [--n 10] [--seed 0] [--minutes 210] [--dry-run]
#
# Outputs: $MAGI1_OUTPUTS/vae_sweep/<sweep>/{configs.json, <run>/, results.md}
# Checkpoints: $MAGI1_SCRATCH/magi1_vae_sweep/<sweep>/<run>/
# Matrix at any time (also while running): python3 magi_finetuning/utils/vae_sweep.py collect --name <sweep>
# Cancel: scancel --name=ig_vsw_<sweep>

set -euo pipefail

NAME=""
N=10
SEED=0
MINUTES=210
DRY_RUN=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --name)    NAME="$2"; shift 2 ;;
        --n)       N="$2"; shift 2 ;;
        --seed)    SEED="$2"; shift 2 ;;
        --minutes) MINUTES="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        *) echo "unknown option: $1 (see the header of $0)" >&2; exit 1 ;;
    esac
done
[[ -n "${NAME}" ]] || { echo "give the sweep a --name" >&2; exit 1; }

JOBS=jobs/jupiter/magi1
[[ -f "${JOBS}/magi1_env.sh" ]] || { echo "run this from the repo root" >&2; exit 1; }
source "${JOBS}/magi1_paths.sh"
TIME="$(( (MINUTES + 30) / 60 )):$(printf %02d $(( (MINUTES + 30) % 60 ))):00"

ids=()
while IFS=$'\t' read -r run args; do
    # shellcheck disable=SC2086  # args are deliberately split into words
    cmd=(sbatch --parsable --job-name="ig_vsw_${NAME}" --time="${TIME}" "${JOBS}/magi1_official_vae.sbatch"
         --ckpt-dir "${MAGI1_SCRATCH}/magi1_vae_sweep/${NAME}/${run}" --out "${MAGI1_OUTPUTS}/vae_sweep/${NAME}/${run}"
         ${args} --max-minutes "${MINUTES}")
    if (( DRY_RUN )); then
        echo "${cmd[*]}"
    else
        id=$("${cmd[@]}"); ids+=("${id}"); echo "${run}: ${id}"
    fi
done < <(python3 magi_finetuning/utils/vae_sweep.py sample --name "${NAME}" --n "${N}" --seed "${SEED}")

if (( ! DRY_RUN )); then
    deps=$(IFS=:; echo "${ids[*]}")
    collect=$(sbatch --parsable --dependency=afterany:"${deps}" --time=00:10:00 --job-name="ig_vsw_${NAME}_collect" \
        "${JOBS}/quick_test.sbatch" magi_finetuning/utils/vae_sweep.py collect --name "${NAME}")
    echo "collect: ${collect}   -> ${MAGI1_OUTPUTS}/vae_sweep/${NAME}/results.md"
else
    echo "(dry run: nothing submitted; configs.json was written)"
fi
