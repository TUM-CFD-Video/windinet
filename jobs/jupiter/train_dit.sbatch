#!/bin/bash
#SBATCH --job-name=wh_train_dit
#SBATCH --account=e-dev-2026d09-262
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00  # jupiter booster limit
#SBATCH --output=logs/jupiter/%x_%j.log
#SBATCH --error=logs/jupiter/%x_%j.err

# DiT training on jupiter (1 node, 4 GPUs). Companion to
# preprocess_dit_data.sbatch: trains on the latents it wrote under
# /e/scratch/e-dev-2026d09-262/wh_work/dit_preprocessed/<PREPROCESSED_NAME>/.
#
# $1 PREPROCESSED_NAME (required), $2 BASE_CONFIG (default: the 30k-step
# Chapter 6 baseline config). The config's __PREPROCESSED_ROOT__ placeholder
# is substituted at launch, same as the other clusters' DiT launchers.
# STEPS=<n> (env) overrides optimization.steps.
#
# AUTO-RESUME: if output_dir/checkpoints already holds model_weights_step_*
# files, model.load_checkpoint is pointed at that directory, and the trainer
# restores weights + optimizer/scheduler/global_step from the newest one.
# So if the 12h walltime kills a run, resubmitting the exact same command
# continues it. If the newest checkpoint is already at optimization.steps,
# the job exits 0 without training (the trainer would raise otherwise), so
# a queued resume job is harmless when the first job finished in time.
#
# Submit from the repo root, windinet env active:
#   sbatch jobs/jupiter/train_dit.sbatch <PREPROCESSED_NAME> [BASE_CONFIG]

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
if [[ -f sc_venv_template/activate.sh ]]; then
    set +eu  # modules.sh has a failing "module load mpi4py" and unset vars
    source sc_venv_template/activate.sh
    set -eu
fi
PYTHON=python
NUM_GPUS=4

export SCRATCH="${SCRATCH_e_dev_2026d09_262}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
export WINDINET_HF_CACHE=/e/project1/e-dev-2026d09-262/wh_work/ltx_pretrained_wh
export HF_HUB_CACHE="${WINDINET_HF_CACHE}"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

PREPROCESSED_NAME=${1:?usage: sbatch jobs/jupiter/train_dit.sbatch <PREPROCESSED_NAME> [BASE_CONFIG]}
BASE_CONFIG=${2:-configs/dit/train_dit_jupiter_ch6_loss_rmse_h1_30k.yaml}
STEPS=${STEPS:-}
PREPROCESSED_ROOT="/e/scratch/e-dev-2026d09-262/wh_work/dit_preprocessed/${PREPROCESSED_NAME}"
RUN_CONFIG=$(mktemp /tmp/wh_train_dit_XXXX.yaml)
JOB_START_EPOCH=$(date +%s)
trap 'echo "end_time=$(date -Is) elapsed=$(( $(date +%s) - JOB_START_EPOCH ))s"; rm -f "${RUN_CONFIG}"' EXIT
trap 'exit 143' TERM

echo "job=${SLURM_JOB_ID} node=$(hostname) gpus=${CUDA_VISIBLE_DEVICES:-}"
echo "start_time=$(date -Is)"
echo "PREPROCESSED_ROOT: ${PREPROCESSED_ROOT}"
echo "BASE_CONFIG: ${BASE_CONFIG}"

# Prints "done" if the newest checkpoint already reached optimization.steps,
# else "train".
STATUS=$("${PYTHON}" - "${BASE_CONFIG}" "${RUN_CONFIG}" "${PREPROCESSED_ROOT}" "${STEPS}" <<'PY'
import sys
from pathlib import Path
import yaml
src, dst, root, steps = sys.argv[1:5]
cfg = yaml.safe_load(open(src))
cfg["data"]["preprocessed_data_root"] = cfg["data"]["preprocessed_data_root"].replace("__PREPROCESSED_ROOT__", root)
if cfg.get("validation", {}).get("data_root"):
    cfg["validation"]["data_root"] = cfg["validation"]["data_root"].replace("__PREPROCESSED_ROOT__", root)
if steps:
    cfg["optimization"]["steps"] = int(steps)
total = cfg["optimization"]["steps"]
ckpt_dir = Path(cfg["output_dir"]) / "checkpoints"
done_steps = [int(p.stem.split("step_")[1]) for p in ckpt_dir.glob("model_weights_step_*.safetensors")] if ckpt_dir.is_dir() else []
status = "train"
if done_steps:
    cfg["model"]["load_checkpoint"] = str(ckpt_dir)
    print(f"resuming from step {max(done_steps)} in {ckpt_dir}", file=sys.stderr)
    if max(done_steps) >= total:
        status = "done"
yaml.safe_dump(cfg, open(dst, "w"), sort_keys=False)
print(f"preprocessed_data_root={cfg['data']['preprocessed_data_root']} output_dir={cfg['output_dir']} "
      f"steps={total} accum={cfg['optimization']['gradient_accumulation_steps']}", file=sys.stderr)
print(status)
PY
)
if [[ "${STATUS}" == "done" ]]; then
    echo "Already at optimization.steps -- nothing to do."
    exit 0
fi

OUT_DIR=$("${PYTHON}" -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['output_dir'])" "${RUN_CONFIG}")
printf '%s\t%s\t%s\t%s\t%s\n' "$(date -Is)" "${SLURM_JOB_ID}" "${SLURM_JOB_NAME}" \
    "${BASE_CONFIG}" "${OUT_DIR}" >> logs/jupiter/INDEX.tsv

"${PYTHON}" -m accelerate.commands.launch \
    --multi_gpu \
    --num_machines 1 \
    --num_processes "${NUM_GPUS}" \
    --mixed_precision bf16 \
    scripts/train.py "${RUN_CONFIG}"
