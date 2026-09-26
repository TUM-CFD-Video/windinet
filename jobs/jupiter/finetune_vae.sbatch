#!/bin/bash
#SBATCH --job-name=wh_finetune_vae
#SBATCH --account=e-dev-2026d09-262
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00  # 20ep 256res full FT takes ~3.5h here; booster limit is 12h
#SBATCH --output=logs/jupiter/%x_%j.log
#SBATCH --error=logs/jupiter/%x_%j.err

# Full-length VAE finetuning on jupiter (1 node, 4 GPUs). Production sibling
# of finetune_vae_debug.sbatch: same environment, but
#   - output_dir has NO _job<ID> suffix, so the checkpoint path is known
#     before submission and preprocess_dit_data.sbatch can be chained on it:
#     /e/scratch/e-dev-2026d09-262/wh_work/finetune_vae_outputs/<output_dir>/
#   - batch_size 4 x accum 2 x 4 GPUs = effective batch 32 (sng_pvc runs the
#     ch6 arms at batch_size 4 too, 8 tiles x accum 1). Override with
#     BATCH_SIZE=<n> if it doesn't fit in memory -- but keep it the same
#     across the resume jobs of one run.
#   - AUTO-RESUME, kept as a safety net: a 20-epoch 256res full FT takes
#     ~3.5h on jupiter (vs ~16h on sng_pvc's 8 tiles), so one 6h job
#     normally finishes the run; a resume is only needed if it gets killed:
#       * <output_dir>/.train_complete exists -> exit 0 without training
#         (written below only after scripts/finetune_vae.py exits 0).
#       * checkpoints/vae_shockwave_last.{safetensors,state.pt} exist ->
#         resume_from=last, clean_output_dir=false; optimizer/scheduler/
#         epoch/best-metric are restored and metrics.csv is continued.
#       * otherwise a fresh run (the config's clean_output_dir applies).
#     So resubmitting the same command continues a killed run, and a queued
#     resume job is harmless if the previous one already finished. A killed
#     run loses at most the epoch in progress (checkpoints.interval=1).
#
# Defaults to the Chapter 6 baseline (rmse+h1, no SSIM); the dataset is the
# 256x256_ds default in CLUSTER_DEFAULTS["jupiter"].
#
# Submit from the repo root, windinet env active (sbatch inherits it):
#   sbatch jobs/jupiter/finetune_vae.sbatch [configs/finetune_vae/<config>.yaml]
# or the whole VAE -> encode -> DiT chain: bash jobs/jupiter/submit_baseline_pipeline.sh

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
export WANDB_MODE=offline  # compute nodes have no internet
export WINDINET_HF_CACHE=/e/project1/e-dev-2026d09-262/wh_work/ltx_pretrained_wh
export HF_HUB_CACHE="${WINDINET_HF_CACHE}"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"  # windinet is not pip-installed in the venv
export PYTHONUNBUFFERED=1  # keep the log complete if the job is hard-killed
export HDF5_USE_FILE_LOCKING=FALSE  # 4 ranks open train.h5 concurrently

BASE_CONFIG=${1:-configs/finetune_vae/finetune_vae_ch6_loss_rmse_h1_256res.yaml}
BATCH_SIZE=${BATCH_SIZE:-4}
RUN_CONFIG=$(mktemp /tmp/wh_finetune_vae_XXXX.yaml)
JOB_START_EPOCH=$(date +%s)
trap 'echo "end_time=$(date -Is) elapsed=$(( $(date +%s) - JOB_START_EPOCH ))s"; rm -f "${RUN_CONFIG}"' EXIT
trap 'exit 143' TERM

echo "job=${SLURM_JOB_ID} node=$(hostname) gpus=${CUDA_VISIBLE_DEVICES:-}"
echo "start_time=$(date -Is)"

# Prints "done" if this run already finished, else "train".
STATUS=$("${PYTHON}" - "${BASE_CONFIG}" "${RUN_CONFIG}" "${NUM_GPUS}" "${BATCH_SIZE}" <<'EOF'
import sys
from pathlib import Path
import yaml
from windinet.cluster_config import patch_config_for_cluster
src, dst, num_processes, batch_size = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
cfg = yaml.safe_load(open(src))
cfg["optimization"]["batch_size"] = batch_size
patch_config_for_cluster(cfg, cluster="jupiter", num_processes=num_processes, output_suffix="")
out = Path(cfg["output_dir"])
last = out / "checkpoints" / "vae_shockwave_last.safetensors"
status = "train"
if (out / ".train_complete").exists():
    status = "done"
elif last.exists() and last.with_name(last.stem + ".state.pt").exists():
    cfg["resume_from"] = str(last)
    cfg["resume_weights_only"] = False
    cfg["clean_output_dir"] = False
    print(f"resuming from {last}", file=sys.stderr)
yaml.safe_dump(cfg, open(dst, "w"), sort_keys=False)
print(f"{num_processes} GPUs x batch {batch_size} x accum {cfg['optimization']['gradient_accumulation_steps']} "
      f"-> effective batch 32; output_dir {cfg['output_dir']}", file=sys.stderr)
print(status)
EOF
)
if [[ "${STATUS}" == "done" ]]; then
    echo "Run already complete (.train_complete present) -- nothing to do."
    exit 0
fi
nvidia-smi --query-gpu=name,memory.total --format=csv || true

OUT_DIR=$("${PYTHON}" -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['output_dir'])" "${RUN_CONFIG}")
printf '%s\t%s\t%s\t%s\t%s\n' "$(date -Is)" "${SLURM_JOB_ID}" "${SLURM_JOB_NAME}" \
    "${BASE_CONFIG}" "${OUT_DIR}" >> logs/jupiter/INDEX.tsv

"${PYTHON}" -m accelerate.commands.launch \
    --multi_gpu \
    --num_machines 1 \
    --num_processes "${NUM_GPUS}" \
    --mixed_precision bf16 \
    scripts/finetune_vae.py "${RUN_CONFIG}"
touch "${OUT_DIR}/.train_complete"

# Mirror metrics/visualizations/config (not checkpoints/) into the repo
# checkout so they can be browsed and git-tracked, same as sng_pvc.
LOCAL_DIR="finetune_vae_outputs/jupiter/$(basename "${OUT_DIR}")"
mkdir -p "${LOCAL_DIR}"
rsync -a --exclude='checkpoints/' "${OUT_DIR}/" "${LOCAL_DIR}/"
echo "Mirrored (excl. checkpoints/) to ${LOCAL_DIR}"
