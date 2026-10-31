#!/bin/bash
#SBATCH --job-name=ig_magi1_test_changes
#SBATCH --account=e-dev-2026d09-262
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=01:30:00
#SBATCH --output=/e/scratch/e-dev-2026d09-262/ig_work/logs/%x_%j.log
#SBATCH --error=/e/scratch/e-dev-2026d09-262/ig_work/logs/%x_%j.err

# Integration test of exact resuming, adapter VAE, LoRA on SCRATCH, VAE-only evaluation and archiving,
# with a few steps each on tiny data. Everything goes to $MAGI1_SCRATCH/test_changes_<mode> and
# $MAGI1_OUTPUTS/tests/<mode>, deleted again at the end except the logs.
#   sbatch jobs/jupiter/magi1/test_changes.sbatch vae    # VAE trainer (old-format resume, exact resume, adapter) + VAE-only eval
#   sbatch jobs/jupiter/magi1/test_changes.sbatch dit    # full DiT exact resume, LoRA, evaluation + archive
#   sbatch jobs/jupiter/magi1/test_changes.sbatch dit --clean-chunks   # same; extra args go to both DiT trainers
# "exact resume" = losses of steps 2.. of (1 step, stop, resume) vs an uninterrupted run: same data and
# noise, so they agree up to GPU nondeterminism; a wrong data order or RNG would differ by far more.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source jobs/jupiter/magi1/magi1_env.sh

MODE="$1"
EXTRA=("${@:2}")   # dit mode: extra trainer args, e.g. --clean-chunks
T="${MAGI1_SCRATCH}/test_changes_${MODE}"
OUT="${MAGI1_OUTPUTS}/tests/${MODE}"
SMOKE_VAE="${MAGI1_SCRATCH}/magi1_smoke/vae"           # finished 6-step VAE run (old format, no metadata)
rm -rf "${T}" "${OUT}"; mkdir -p "${T}" "${OUT}"
echo "job=${SLURM_JOB_ID} node=$(hostname) mode=${MODE} start=$(date -Is)"

losses() {  # losses <log> -> "step loss" lines for steps >= 2
    grep -E "^step +[0-9]+ +loss" "$1" | awk '$2 >= 2 {print $2, $4}'
}
compare() {  # compare <uninterrupted log> <resumed log>
    echo "step  uninterrupted  resumed"
    join <(losses "$1") <(losses "$2") | awk '{d = ($2 > $3 ? $2 - $3 : $3 - $2) / $2; printf "%4d  %12s  %12s   rel diff %.1e\n", $1, $2, $3, d}'
}

if [[ "${MODE}" == vae ]]; then
    VAE_ARGS=(--max-samples 32 --eval-samples 4 --eval-every 2 --log-every 1)

    echo "=== 1. resume from an OLD-format resume.pt (what VAE part 2 of the official run will do)"
    cp -r "${SMOKE_VAE}" "${T}/compat"
    rm -f "${T}/compat/finished.json"
    launch_4gpu -m magi_finetuning.train.train_vae --ckpt-dir "${T}/compat" --out "${OUT}/compat" --steps 8 --resume \
        "${VAE_ARGS[@]}" 2>&1 | tee "${OUT}/compat.log" | grep -E "resumed|^step|done|Error|error" || true
    test -f "${T}/compat/finished.json" && echo "PASS old-format resume finished" || echo "FAIL old-format resume"

    echo "=== 2. exact resume, full VAE + SDS: uninterrupted 4 steps vs (1 step, stop, resume to 4)"
    launch_4gpu -m magi_finetuning.train.train_vae --ckpt-dir "${T}/straight" --out "${OUT}/straight" --steps 4 \
        "${VAE_ARGS[@]}" > "${OUT}/straight.log" 2>&1 || echo "FAIL: straight exited with an error (see ${OUT}/straight.log)"
    launch_4gpu -m magi_finetuning.train.train_vae --ckpt-dir "${T}/resumed" --out "${OUT}/resumed" --steps 4 --max-minutes 0 \
        "${VAE_ARGS[@]}" > "${OUT}/resumed_part1.log" 2>&1 || echo "FAIL: resumed_part1 exited with an error (see ${OUT}/resumed_part1.log)"
    launch_4gpu -m magi_finetuning.train.train_vae --ckpt-dir "${T}/resumed" --out "${OUT}/resumed" --steps 4 --resume \
        "${VAE_ARGS[@]}" > "${OUT}/resumed_part2.log" 2>&1 || echo "FAIL: resumed_part2 exited with an error (see ${OUT}/resumed_part2.log)"
    grep -h "resumed at" "${OUT}/resumed_part2.log" || true
    compare "${OUT}/straight.log" "${OUT}/resumed_part2.log"

    echo "=== 3. adapter VAE, no SDS: 2 steps, then VAE-only evaluation (2 sims + the 3 cases)"
    launch_4gpu -m magi_finetuning.train.train_vae --train adapter --sds-weight 0 --ckpt-dir "${T}/adapter" \
        --out "${OUT}/adapter" --steps 2 "${VAE_ARGS[@]}" 2>&1 | tee "${OUT}/adapter.log" | grep -E "trainable|^step|done|Error" || true
    CUDA_VISIBLE_DEVICES=0 python -m magi_finetuning.eval.evaluate --vae-only --vae-ckpt "${T}/adapter/vae_best.safetensors" \
        --metric-sims 2 --out "${OUT}/adapter_eval" --no-archive 2>&1 | tee "${OUT}/adapter_eval.log" | tail -25 || echo "FAIL: evaluation exited with an error"
    ls "${OUT}/adapter_eval/videos" || true

elif [[ "${MODE}" == dit ]]; then
    DIT_ARGS=(--full-finetune --frames 100 --max-samples 64 --vae-ckpt "${SMOKE_VAE}/vae_best.safetensors" "${EXTRA[@]}")

    echo "=== 1. exact resume, full DiT: uninterrupted 3 steps vs (1 step, stop, resume to 3)"
    launch_4gpu -m magi_finetuning.train.finetune "${DIT_ARGS[@]}" --steps 3 --no-save --out "${OUT}/straight" \
        > "${OUT}/straight.log" 2>&1 || echo "FAIL: straight exited with an error (see ${OUT}/straight.log)"
    launch_4gpu -m magi_finetuning.train.finetune "${DIT_ARGS[@]}" --steps 3 --ckpt-dir "${T}/full" --save-every 1000 \
        --max-minutes 0 --resume --out "${OUT}/resumed" > "${OUT}/resumed_part1.log" 2>&1 || echo "FAIL: resumed_part1 exited with an error"
    launch_4gpu -m magi_finetuning.train.finetune "${DIT_ARGS[@]}" --steps 3 --ckpt-dir "${T}/full" --save-every 1000 \
        --resume --out "${OUT}/resumed" > "${OUT}/resumed_part2.log" 2>&1 || echo "FAIL: resumed_part2 exited with an error"
    grep -h "resumed at\|normalization" "${OUT}/resumed_part2.log" || true
    compare "${OUT}/straight.log" "${OUT}/resumed_part2.log"
    ls "${T}/full" || true

    echo "=== 2. LoRA with checkpoints on SCRATCH: 2 steps"
    launch_4gpu -m magi_finetuning.train.finetune --frames 100 --max-samples 32 --steps 2 --save-every 2 --log-every 1 \
        --vae-ckpt "${SMOKE_VAE}/vae_best.safetensors" --ckpt-dir "${T}/lora" --out "${OUT}/lora" "${EXTRA[@]}" 2>&1 \
        | tee "${OUT}/lora.log" | grep -E "trainable|^step|saved|normalization|Error" || true
    ls "${T}/lora" || true

    echo "=== 3. evaluation of that LoRA (2 sims + 3 cases, 4 Euler steps = 1 per stage of the 4-chunk window), archive, remove it"
    CUDA_VISIBLE_DEVICES=0 python -m magi_finetuning.eval.evaluate --dit-ckpt "${T}/lora/lora_step000002.safetensors" \
        --vae-ckpt "${SMOKE_VAE}/vae_best.safetensors" --metric-sims 2 --num-steps 4 --out "${OUT}/lora_eval" \
        --archive-name test_changes_archive 2>&1 | tee "${OUT}/lora_eval.log" | tail -30 || echo "FAIL: evaluation exited with an error"
    ARCHIVE="${MAGI1_SCRATCH}/magi1_final/test_changes_archive"
    find "${ARCHIVE}" -maxdepth 2 | sort | head -30 || true
    rm -rf "${ARCHIVE}"
fi

rm -rf "${T}"   # checkpoints of the test (tens of GB); logs and small outputs stay in ${OUT}
echo "end=$(date -Is)"
