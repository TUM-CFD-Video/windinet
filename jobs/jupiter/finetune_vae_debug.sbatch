#!/bin/bash
#SBATCH --job-name=wh_finetune_vae_4gpu
#SBATCH --account=e-dev-2026d09-262
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=logs/jupiter/%x_%j.log
#SBATCH --error=logs/jupiter/%x_%j.err

# Single-node, 4-GPU VAE finetuning launcher. Accelerate spawns four workers
# from one Slurm task. Target effective batch is 32 (batch_size=1 -> accum=8).
# The default 30-minute allocation is for a smoke run; override --time when
# submitting a full training run, within your partition's allowed limit.
#
# Needs, before the first run (compute nodes have no internet):
#   - dataset at /e/project1/e-dev-2026d09-262/datasets/euler_mq_dataset/256x256_ds/
#     (login node: python euler_mq_dataset/jupiter/download_from_hf.py)
#   - LTX weights in /e/project1/e-dev-2026d09-262/wh_work/ltx_pretrained_wh
#     (login node: bash jobs/jupiter/download_pretrained.sh)
# Outputs land under /e/scratch/e-dev-2026d09-262/wh_work/finetune_vae_outputs/
# (windinet/cluster_config.py's CLUSTER_DEFAULTS["jupiter"]).
#
# Before submission:           mkdir -p logs/jupiter
# Submit from the repo root, windinet env active (sbatch inherits it):
#                              sbatch jobs/jupiter/finetune_vae_debug.sbatch <config.yaml>
# Watch with:                  tail -f logs/jupiter/wh_finetune_vae_4gpu_<jobid>.log

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
# Uses the caller's environment (e.g. `conda activate windinet`); the
# sc_venv_template venv is only sourced if a checkout actually has one.
if [[ -f sc_venv_template/activate.sh ]]; then
    set +eu  # modules.sh has a failing "module load mpi4py" and unset vars
    source sc_venv_template/activate.sh
    set -eu
fi
PYTHON=python

export SCRATCH="${SCRATCH_e_dev_2026d09_262}"
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline  # compute nodes have no internet
export WINDINET_HF_CACHE=/e/project1/e-dev-2026d09-262/wh_work/ltx_pretrained_wh
export HF_HUB_CACHE="${WINDINET_HF_CACHE}"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"  # windinet is not pip-installed in the venv

BASE_CONFIG=${1:?usage: sbatch jobs/jupiter/finetune_vae_debug.sbatch configs/finetune_vae/<config>.yaml}
RUN_CONFIG=$(mktemp /tmp/wh_finetune_vae_4gpu_XXXX.yaml)
JOB_START_EPOCH=$(date +%s)
trap 'echo "end_time=$(date -Is) elapsed=$(( $(date +%s) - JOB_START_EPOCH ))s"; rm -f "${RUN_CONFIG}"' EXIT
trap 'exit 143' TERM

"${PYTHON}" - "${BASE_CONFIG}" "${RUN_CONFIG}" <<'PY'
import os, sys, yaml
from windinet.cluster_config import patch_config_for_cluster
src, dst = sys.argv[1], sys.argv[2]
cfg = yaml.safe_load(open(src))
patch_config_for_cluster(
    cfg, cluster="jupiter", num_processes=4,
    output_suffix="_job" + os.environ["SLURM_JOB_ID"], effective_batch=32,
    data_root=os.environ.get("DATA_ROOT"),
)
yaml.safe_dump(cfg, open(dst, "w"), sort_keys=False)
print(f"4 GPUs x batch {cfg['optimization']['batch_size']} x accum {cfg['optimization']['gradient_accumulation_steps']} -> effective batch 32; output_dir {cfg['output_dir']}")
PY

echo "job=${SLURM_JOB_ID} node=$(hostname) gpus=${CUDA_VISIBLE_DEVICES:-}"
echo "start_time=$(date -Is)"

OUT_DIR=$("${PYTHON}" -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['output_dir'])" "${RUN_CONFIG}")
printf '%s\t%s\t%s\t%s\t%s\n' "$(date -Is)" "${SLURM_JOB_ID}" "${SLURM_JOB_NAME}" \
    "${BASE_CONFIG}" "${OUT_DIR}" >> logs/jupiter/INDEX.tsv

"${PYTHON}" -m accelerate.commands.launch \
    --multi_gpu \
    --num_machines 1 \
    --num_processes 4 \
    scripts/finetune_vae.py "${RUN_CONFIG}"
