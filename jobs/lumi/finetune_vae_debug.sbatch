#!/bin/bash
#SBATCH --job-name=lc_finetune_vae_8gpu
#SBATCH --account=project_465003416
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=56
#SBATCH --time=00:30:00
#SBATCH --output=logs/lumi/%x_%j.log
#SBATCH --error=logs/lumi/%x_%j.err

# Single-node, 8-GPU VAE finetuning on LUMI (effective batch 32). Data, LTX
# weights and outputs live on scratch (CLUSTER_DEFAULTS["lumi"]).
# Submit from the repo root:
#   sbatch jobs/lumi/finetune_vae_debug.sbatch configs/finetune_vae/<config>.yaml
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

NUM_GPUS=8
SCRATCH=/scratch/project_465003416
CONTAINER=${CONTAINER:-/project/project_465003416/venvs/windinet_rocm.sif}
PYTHON=(singularity exec -B "${PWD}":/workspace -B "${SCRATCH}" --pwd /workspace "${CONTAINER}" python)

export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}  # srun no longer inherits --cpus-per-task
export HF_HUB_OFFLINE=1 WANDB_MODE=offline
export WINDINET_HF_CACHE=${SCRATCH}/lc_work/ltx_pretrained_lc HF_HUB_CACHE=${SCRATCH}/lc_work/ltx_pretrained_lc
export MIOPEN_USER_DB_PATH=/tmp/${USER}-miopen MIOPEN_CUSTOM_CACHE_DIR=/tmp/${USER}-miopen
export PYTHONPATH=/workspace

BASE_CONFIG=${1:?usage: sbatch jobs/lumi/finetune_vae_debug.sbatch configs/finetune_vae/<config>.yaml}
RUN_CONFIG=$(mktemp /tmp/lc_finetune_vae_XXXX.yaml)
trap 'rm -f "${RUN_CONFIG}"' EXIT

"${PYTHON[@]}" - "${BASE_CONFIG}" "${RUN_CONFIG}" "${NUM_GPUS}" <<'PY'
import os, sys, yaml
from windinet.cluster_config import patch_config_for_cluster
src, dst, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
cfg = yaml.safe_load(open(src))
patch_config_for_cluster(cfg, cluster="lumi", num_processes=n, output_suffix="_job" + os.environ["SLURM_JOB_ID"])
yaml.safe_dump(cfg, open(dst, "w"), sort_keys=False)
print(f"{n} GPUs x batch {cfg['optimization']['batch_size']} x accum "
      f"{cfg['optimization']['gradient_accumulation_steps']}; output_dir {cfg['output_dir']}")
PY

srun "${PYTHON[@]}" -m accelerate.commands.launch --multi_gpu --num_machines 1 --num_processes "${NUM_GPUS}" \
    /workspace/scripts/finetune_vae.py "${RUN_CONFIG}"
