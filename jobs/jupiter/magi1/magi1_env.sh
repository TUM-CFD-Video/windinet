# Shared environment for the MAGI-1 jobs (sourced by the sbatch files, from the repo root).

IG_WORK="$(dirname "${PWD}")"
set +eu  # modules.sh has a failing "module load mpi4py" and unset vars
source sc_venv_template/activate.sh
set -eu

export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1                                   # compute nodes have no internet
export HF_HOME="${IG_WORK}/.cache/huggingface"            # every cache inside IG_work
export XDG_CACHE_HOME="${IG_WORK}/.cache"
export TRITON_CACHE_DIR="${IG_WORK}/.cache/triton"
export CUDA_CACHE_PATH="${IG_WORK}/.cache/nv"
export MPLCONFIGDIR="${IG_WORK}/.cache/matplotlib"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # less fragmentation at high memory use

source jobs/jupiter/magi1/magi1_paths.sh   # MAGI1_SCRATCH / _OUTPUTS / _LOGS / _OFFICIAL: everything on SCRATCH

launch_4gpu() {  # accelerate launch on all GPUs of this node, then the given python args
    local num_gpus master_port multi=""
    num_gpus=$(nvidia-smi -L | wc -l)
    [[ ${num_gpus} -gt 1 ]] && multi="--multi_gpu"
    master_port=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
    python -m accelerate.commands.launch ${multi} --num_machines 1 --num_processes "${num_gpus}" \
        --main_process_port "${master_port}" --mixed_precision no "$@"
}

touch_scratch() {  # reset the 90-day purge clock on all MAGI-1 checkpoints on SCRATCH
    [[ -d "${MAGI1_SCRATCH}" ]] && find "${MAGI1_SCRATCH}" -exec touch -a -m {} + || true
}
