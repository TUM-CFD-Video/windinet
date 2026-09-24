#!/bin/bash
# Populate jupiter's LTX-Video weight cache (LTXV_2B_0.9.6_DEV: transformer,
# VAE fallback from 0.9.5, scheduler config) under project storage, so it
# survives scratch purges and is shared by every jupiter job.
#
# Run ONCE on a LOGIN node (compute nodes have no internet), from the repo root,
# with the windinet env active:
#   bash jobs/jupiter/download_pretrained.sh
#
# Every jobs/jupiter/*.sbatch reads the same directory offline via
# WINDINET_HF_CACHE/HF_HUB_CACHE -- keep LTX_CACHE in sync with them.

set -euo pipefail

LTX_CACHE=/e/project1/e-dev-2026d09-262/wh_work/ltx_pretrained_wh
mkdir -p "${LTX_CACHE}"

# Uses the caller's environment (e.g. `conda activate windinet`); the
# sc_venv_template venv is only sourced if a checkout actually has one.
if [[ -f sc_venv_template/activate.sh ]]; then
    set +eu  # modules.sh has a failing "module load mpi4py" and unset vars
    source sc_venv_template/activate.sh
    set -eu
fi

export WINDINET_HF_CACHE="${LTX_CACHE}"
export HF_HUB_CACHE="${LTX_CACHE}"
unset HF_HUB_OFFLINE
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

python -c "import windinet; from windinet.inference.model_loader import load_ltxv_components; load_ltxv_components('LTXV_2B_0.9.6_DEV')"
# Newer huggingface_hub keeps blobs in ${LTX_CACHE}/blobs/, not per-repo, so
# size the whole cache (~8.4GB expected), not the models--* dirs.
du -sh "${LTX_CACHE}"
