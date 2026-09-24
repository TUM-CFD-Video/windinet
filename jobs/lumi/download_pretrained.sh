#!/bin/bash
# Download LTXV_2B_0.9.6_DEV weights to scratch. Run once on a login node,
# from the repo root: bash jobs/lumi/download_pretrained.sh
set -euo pipefail

LTX_CACHE=/scratch/project_465003416/lc_work/ltx_pretrained_lc
SIF=/project/project_465003416/venvs/windinet.sif
mkdir -p "${LTX_CACHE}"

export WINDINET_HF_CACHE="${LTX_CACHE}" HF_HUB_CACHE="${LTX_CACHE}" HF_HOME="${LTX_CACHE}/.hf_home"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
unset HF_HUB_OFFLINE

singularity exec -B /scratch/project_465003416 "${SIF}" python -c \
    "from windinet.inference.model_loader import load_ltxv_components; load_ltxv_components('LTXV_2B_0.9.6_DEV')"
du -sh "${LTX_CACHE}"  # ~8.4G expected
