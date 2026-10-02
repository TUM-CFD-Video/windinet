#!/bin/bash
# Download the pretrained weights to the scratch HF cache: LTX-Video 2B (Weihao's pipeline) and the Wan 2.1 VAE.
# Run once on a login node (compute nodes are offline), from the repo root: bash jobs/lumi/lc/download_pretrained.sh
set -euo pipefail

HF_CACHE=/scratch/project_465003416/lc_work/hf_cache
SIF=/project/project_465003416/venvs/windinet_rocm.sif
mkdir -p "${HF_CACHE}"

export WINDINET_HF_CACHE="${HF_CACHE}" HF_HUB_CACHE="${HF_CACHE}" HF_HOME="${HF_CACHE}/.hf_home"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
unset HF_HUB_OFFLINE

RUN=(singularity exec -B /scratch/project_465003416 -B "${PWD}":/workspace --pwd /workspace "${SIF}")
"${RUN[@]}" python -c "from windinet.inference.model_loader import load_ltxv_components; load_ltxv_components('LTXV_2B_0.9.6_DEV')"
"${RUN[@]}" hf download Wan-AI/Wan2.1-T2V-1.3B-Diffusers --include "vae/*"  # the VAE alone, ~500 MB
du -sh "${HF_CACHE}"  # ~8.9G expected
