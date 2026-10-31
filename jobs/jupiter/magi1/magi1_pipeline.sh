#!/bin/bash
# Submit the MAGI-1 stages as one Slurm dependency chain. Run on the login node from the repo root
# (it only calls sbatch; all work happens in the jobs).
#
#                     +-> LoRA DiT (1 job) ----------> evaluation -> ${MAGI1_OUTPUTS}/official/lora/eval/
#   VAE x N_VAE  -----+
#                     +-> full DiT x N_DIT  --------> evaluation -> ${MAGI1_OUTPUTS}/official/full/eval/
#
# Every arrow is `sbatch --dependency=afterok:<job id>`: a job starts only after the previous one
# finished successfully. If one fails, everything after it stays pending (DependencyNeverSatisfied)
# and has to be removed with scancel. Long stages are several 12 h jobs: each one stops in time,
# writes resume.pt and the next continues; spare jobs after a finished stage exit within a minute.
#
# Usage:
#   bash jobs/jupiter/magi1/magi1_pipeline.sh [options]
#     --stages LIST   comma-separated subset of vae,lora,full        (default: vae,lora,full)
#     --n-vae N       VAE jobs, ~11 h each; 20 epochs need 2          (default: 2)
#     --n-dit N       full-DiT jobs; 8000 steps at 18 s/step need 4  (default: 5, one spare)
#     --after ID      job id(s), ':'-separated, the first stage waits for, e.g. an already-queued VAE
#                     when running only --stages lora or --stages full
#     --dry-run       print the sbatch commands, submit nothing
#
# Examples:
#   bash jobs/jupiter/magi1/magi1_pipeline.sh --dry-run                       # check what would be submitted
#   bash jobs/jupiter/magi1/magi1_pipeline.sh                                 # everything
#   bash jobs/jupiter/magi1/magi1_pipeline.sh --stages full --after 2430898   # full DiT on a queued VAE
#
# Cancel everything:
#   scancel --name=ig_magi1_official_vae,ig_magi1_official_lora,ig_magi1_official_dit,ig_evaluate_magi1

set -euo pipefail

STAGES="vae,lora,full"
N_VAE=2
N_DIT=5
AFTER=""
DRY_RUN=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --stages)  STAGES="$2"; shift 2 ;;
        --n-vae)   N_VAE="$2"; shift 2 ;;
        --n-dit)   N_DIT="$2"; shift 2 ;;
        --after)   AFTER="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        *) echo "unknown option: $1 (see the header of $0)" >&2; exit 1 ;;
    esac
done

JOBS=jobs/jupiter/magi1
source "${JOBS}/magi1_paths.sh"   # MAGI1_OUTPUTS, MAGI1_OFFICIAL, ... (all on SCRATCH)
VAE_CKPT="${MAGI1_OFFICIAL}/vae/vae_best.safetensors"
LORA_CKPT="${MAGI1_OFFICIAL}/lora/lora_step001500.safetensors"  # --steps 1500 in the LoRA sbatch
FULL_CKPT="${MAGI1_OFFICIAL}/dit/full_step008000.safetensors"                    # --steps 8000 in the DiT sbatch

[[ -f "${JOBS}/magi1_env.sh" ]] || { echo "run this from the repo root" >&2; exit 1; }

has_stage() { [[ ",${STAGES}," == *",$1,"* ]]; }

submit() {  # submit <job id(s) to wait for, or ""> <sbatch args...>; prints the new job id
    local dep="$1"; shift
    local cmd=(sbatch --parsable ${dep:+--dependency=afterok:${dep}} "$@")
    if (( DRY_RUN )); then
        echo "${cmd[*]}" >&2
        echo "<job${RANDOM}>"   # placeholder id so the printed chain shows the dependencies
    else
        "${cmd[@]}"
    fi
}

# Stage 1: VAE (all later stages need its finished checkpoint).
vae_done="${AFTER}"
if has_stage vae; then
    for i in $(seq 1 "${N_VAE}"); do
        vae_done=$(submit "${vae_done}" "${JOBS}/magi1_official_vae.sbatch")
        echo "VAE       ${i}/${N_VAE}: ${vae_done}"
    done
fi

# Stage 2a: LoRA DiT -> evaluation.
if has_stage lora; then
    lora=$(submit "${vae_done}" "${JOBS}/magi1_official_lora.sbatch")
    echo "LoRA DiT       : ${lora}"
    lora_eval=$(submit "${lora}" "${JOBS}/evaluate_magi1.sbatch" \
        --dit-ckpt "${LORA_CKPT}" --vae-ckpt "${VAE_CKPT}" \
        --frames 100 --metric-sims 500 --out "${MAGI1_OUTPUTS}/official/lora/eval")
    echo "LoRA eval      : ${lora_eval}   -> ${MAGI1_OUTPUTS}/official/lora/eval/results.md"
fi

# Stage 2b: full DiT (N_DIT resuming jobs) -> evaluation. Runs alongside 2a; both only need the VAE.
if has_stage full; then
    dit="${vae_done}"
    for i in $(seq 1 "${N_DIT}"); do
        dit=$(submit "${dit}" "${JOBS}/magi1_official_dit.sbatch")
        echo "full DiT  ${i}/${N_DIT}: ${dit}"
    done
    full_eval=$(submit "${dit}" "${JOBS}/evaluate_magi1.sbatch" \
        --dit-ckpt "${FULL_CKPT}" --vae-ckpt "${VAE_CKPT}" \
        --frames 100 --metric-sims 500 --out "${MAGI1_OUTPUTS}/official/full/eval")
    echo "full eval      : ${full_eval}   -> ${MAGI1_OUTPUTS}/official/full/eval/results.md"
fi

(( DRY_RUN )) && echo "(dry run: nothing submitted)"
echo "check with: squeue -u \$USER"
