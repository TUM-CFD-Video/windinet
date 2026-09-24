#!/bin/bash
#SBATCH --job-name=wh_debug
#SBATCH --account=e-dev-2026d09-262
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=logs/jupiter/%x_%j.log
#SBATCH --error=logs/jupiter/%x_%j.log

# 1-GPU debug launcher (JUPITER port of the lundquist debug script). The config
# is a required argument. effective_batch=1 on purpose: overfit diagnostics
# rely on gradient_accumulation_steps: 1.
# NOTE: booster nodes are allocated exclusively (4x GH200), so one GPU here
# still costs a full node.
#
# Needs, before the first run (compute nodes have no internet):
#   - dataset at $SCRATCH/windinet/euler_mq_dataset/128x128_ds/train.h5
#   - LTX weights in ~/.cache/huggingface/hub (download on a login node)
#
# Submit from the repo root:   sbatch jobs/jupiter/run_job.sbatch <config.yaml>
# Watch with:                  tail -f logs/jupiter/wh_debug_<jobid>.log

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
set +eu  # modules.sh has a failing "module load mpi4py" and unset vars
source sc_venv_template/activate.sh
set -eu
PYTHON=python

export SCRATCH="${SCRATCH_e_dev_2026d09_262}"
export HF_HUB_OFFLINE=1
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"  # windinet is not pip-installed in the venv

BASE_CONFIG=${1:?usage: sbatch jobs/jupiter/run_job.sbatch configs/finetune_vae/<config>.yaml}
RUN_CONFIG=$(mktemp /tmp/wh_debug_XXXX.yaml)

"${PYTHON}" - "${BASE_CONFIG}" "${RUN_CONFIG}" <<'PY'
import sys, yaml
from windinet.cluster_config import patch_config_for_cluster
src, dst = sys.argv[1], sys.argv[2]
cfg = yaml.safe_load(open(src))
patch_config_for_cluster(cfg, cluster="jupiter", num_processes=1, output_suffix="", effective_batch=1)
yaml.safe_dump(cfg, open(dst, "w"), sort_keys=False)
print(f"1 GPU x accum {cfg['optimization']['gradient_accumulation_steps']} -> effective batch 1; output_dir {cfg['output_dir']}")
PY

echo "job=${SLURM_JOB_ID} node=$(hostname) gpus=${CUDA_VISIBLE_DEVICES:-}"
JOB_START_EPOCH=$(date +%s)
echo "start_time=$(date -Is)"
trap 'echo "end_time=$(date -Is) elapsed=$(( $(date +%s) - JOB_START_EPOCH ))s"; rm -f "${RUN_CONFIG}"' EXIT
trap 'exit 143' TERM

OUT_DIR=$("${PYTHON}" -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['output_dir'])" "${RUN_CONFIG}")
printf '%s\t%s\t%s\t%s\t%s\n' "$(date -Is)" "${SLURM_JOB_ID}" "${SLURM_JOB_NAME}" \
    "${BASE_CONFIG}" "${OUT_DIR}" >> logs/jupiter/INDEX.tsv

"${PYTHON}" -m accelerate.commands.launch \
    --num_processes 1 \
    scripts/finetune_vae.py "${RUN_CONFIG}"
