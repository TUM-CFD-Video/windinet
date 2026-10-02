#!/bin/bash
# Run every line of a command file in parallel inside the current job, one GPU per line, and wait for all.
# Cluster-agnostic: CUDA_VISIBLE_DEVICES selects the GPU on NVIDIA and, through HIP, on AMD too (setting
# ROCR_VISIBLE_DEVICES as well would mask twice). Lines starting with # are skipped.
# Per-command logs: logs/lc/<job name>_<job id>/gpu<i>.log. From the repo root, e.g. on LUMI:
#   sbatch --time=05:00:00 --partition=small-g jobs/lumi/lc/run.sbatch bash jobs/lc/one_per_gpu.sh jobs/lc/vae_baselines.txt
set -uo pipefail
commands=${1:?usage: one_per_gpu.sh <file with one command per line>}
log_dir=logs/lc/${SLURM_JOB_NAME:-local}_${SLURM_JOB_ID:-$$}
mkdir -p "${log_dir}"
gpu=0
while IFS= read -r command; do
    [[ -z "${command}" || "${command}" == \#* ]] && continue
    echo "gpu ${gpu}: ${command}"
    CUDA_VISIBLE_DEVICES=${gpu} bash -c "${command}" > "${log_dir}/gpu${gpu}.log" 2>&1 &
    gpu=$((gpu + 1))
done < "${commands}"
wait
