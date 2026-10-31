"""Download MAGI-1 4.5B (DiT + VAE, no text encoder) to project storage.
Run on a login node from the repo root: python magi_finetuning/utils/download.py"""

import os

os.environ["HF_HOME"] = os.path.abspath("../.cache/huggingface")  # keep HF cache inside IG_work

from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="sand-ai/MAGI-1",
    local_dir="/e/project1/e-dev-2026d09-262/ig_work/magi1_pretrained",  # = settings.MAGI1_WEIGHTS
    allow_patterns=["ckpt/vae/*", "ckpt/magi/4.5B_base/inference_weight/*"],
)
