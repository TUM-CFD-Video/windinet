# Environment for windinet jobs on JUPITER. Sourced by the launchers in this folder, from the repo root,
# with the windinet conda env active (sbatch inherits it). Compute nodes have no internet: the dataset sits on
# project storage (euler_mq_dataset/jupiter/download_from_hf.py), the LTX weights under lc_work
# (jobs/jupiter/download_pretrained.sh), outputs go to scratch.
export SCRATCH="${SCRATCH_e_dev_2026d09_262}"
export WINDINET_WORK=${SCRATCH}/lc_work  # outputs: windinet/cluster_config.py
export WINDINET_HF_CACHE=/e/project1/e-dev-2026d09-262/lc_work/ltx_pretrained_lc HF_HUB_CACHE=/e/project1/e-dev-2026d09-262/lc_work/ltx_pretrained_lc
export HF_HUB_OFFLINE=1 WANDB_MODE=offline PYTHONUNBUFFERED=1 HDF5_USE_FILE_LOCKING=FALSE
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"  # windinet is not pip-installed in the env
RUN=(env)  # no container: commands run as they are
