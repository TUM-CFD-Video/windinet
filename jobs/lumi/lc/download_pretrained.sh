#!/bin/bash
# Download the pretrained weights to the scratch HF cache: LTX-Video 2B (Weihao's pipeline) and the Wan 2.1 VAE.
# Run once on a login node (compute nodes are offline), from the repo root: bash jobs/lumi/lc/download_pretrained.sh
set -euo pipefail
source jobs/lumi/lc/env.sh
unset HF_HUB_OFFLINE
export HF_HOME="${HF_HUB_CACHE}/.hf_home"
mkdir -p "${HF_HUB_CACHE}"
"${RUN[@]}" python -c "from windinet.inference.model_loader import load_ltxv_components; load_ltxv_components('LTXV_2B_0.9.6_DEV')"
"${RUN[@]}" hf download Wan-AI/Wan2.1-T2V-1.3B-Diffusers --include "vae/*"  # the VAE alone, ~500 MB
du -sh "${HF_HUB_CACHE}"  # ~8.9G expected
