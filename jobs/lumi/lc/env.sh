# Environment for windinet jobs on LUMI. Sourced by the launchers in this folder, from the repo root.
# Data, LTX weights and outputs live on project scratch under lc_work; the code runs in a ROCm container
# with the repo mounted at /workspace:  "${RUN[@]}" python <script> ...
SCRATCH=/scratch/project_465003416
WORK=${SCRATCH}/lc_work
CONTAINER=${CONTAINER:-/project/project_465003416/venvs/windinet_rocm.sif}
export WINDINET_WORK=${WORK}  # outputs: windinet/cluster_config.py
export WINDINET_HF_CACHE=${WORK}/ltx_pretrained_lc HF_HUB_CACHE=${WORK}/ltx_pretrained_lc
export HF_HUB_OFFLINE=1 WANDB_MODE=offline PYTHONUNBUFFERED=1 HDF5_USE_FILE_LOCKING=FALSE
export MIOPEN_USER_DB_PATH=/tmp/${USER}-miopen MIOPEN_CUSTOM_CACHE_DIR=/tmp/${USER}-miopen
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-1}  # srun no longer inherits --cpus-per-task
export PYTHONPATH=/workspace
RUN=(singularity exec -B "${PWD}":/workspace -B "${SCRATCH}" --pwd /workspace "${CONTAINER}")
