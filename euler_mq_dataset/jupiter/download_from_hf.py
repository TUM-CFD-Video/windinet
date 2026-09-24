"""
Download one resolution's worth of the euler_mq ShockWave CFD dataset from
HuggingFace (rha6696/euler_mq) onto jupiter's project storage.

Same three top-level resolution folders as euler_mq_dataset/sng_pvc/
download_from_hf.py (see that script's docstring for the dataset-card
detail). Unlike sng_pvc/lrz_ai, jupiter defaults to 256x256_ds -- that is the
resolution this cluster is set up to train on.

Lands at DATA_ROOT/<resolution>/{train,val,...}.h5, where DATA_ROOT is the
shared project dataset dir (not $SCRATCH_e_dev_2026d09_262, which is purged).
windinet/cluster_config.py's CLUSTER_DEFAULTS["jupiter"]["data_root"] points
at DATA_ROOT/256x256_ds/train.h5 -- change both together.

Needs outbound internet access to huggingface.co -- run this from a login
node, not inside a job (compute nodes have no internet, and
jobs/jupiter/*.sbatch set HF_HUB_OFFLINE=1).

Usage (run from the repo root with the windinet env active):
    python euler_mq_dataset/jupiter/download_from_hf.py                 # 256x256_ds (default)
    python euler_mq_dataset/jupiter/download_from_hf.py 128x128_ds
    python euler_mq_dataset/jupiter/download_from_hf.py 512x512_orig
"""

import os

import typer
from huggingface_hub import snapshot_download

RESOLUTIONS = ["128x128_ds", "256x256_ds", "512x512_orig"]

DATA_ROOT = "/e/project1/e-dev-2026d09-262/datasets/euler_mq_dataset"


def main(
    resolution: str = typer.Argument(
        "256x256_ds",
        help=f"Which resolution folder to download. One of: {', '.join(RESOLUTIONS)}.",
    ),
) -> None:
    if resolution not in RESOLUTIONS:
        raise typer.BadParameter(f"resolution must be one of {RESOLUTIONS}, got {resolution!r}")

    os.makedirs(DATA_ROOT, exist_ok=True)

    snapshot_download(
        repo_id="rha6696/euler_mq",
        repo_type="dataset",
        allow_patterns=f"{resolution}/*",
        local_dir=DATA_ROOT,
    )


if __name__ == "__main__":
    typer.run(main)
