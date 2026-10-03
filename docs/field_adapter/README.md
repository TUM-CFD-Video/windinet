# Field adapter and Wan 2.1 VAE stage

A 14-parameter per-pixel map from the Euler fields (ρ, m_x, m_y, p) to one or two RGB images, so the Wan 2.1 VAE
can carry CFD fields; optionally the VAE decoder is fine-tuned behind it. Findings of the adapter study:
[ablations.md](ablations.md), runs in the [field_adapter index](../../results/field_adapter/20_wan_roundtrip/README.md).
VAE baselines on the LTX protocol (train.h5 / test.h5, whole-trajectory VRMSE): [docs/wan/vae_baselines.md](../wan/vae_baselines.md), runs in [results/wan/vae](../../results/wan/vae/README.md).

## Setup

```bash
pip install -e .
```

- Data: `euler_mq_dataset/256x256_ds/{train,test}.h5` (not in git; a symlink to the dataset outside the repo).
  Channel statistics are committed (`results/field_adapter/00_stats/`); recompute with `scripts/compute_channel_stats.py`.
- The Fréchet latent diagnostic (natural-video reference) is no longer in the code; its numbers in ablations.md
  are in the run folders' metrics.json, the last code with it is commit 2c65a29.
- Wan weights come from the HF cache (`Wan-AI/Wan2.1-T2V-1.3B-Diffusers`, VAE only, 500 MB).
- GPU: fp32. 11 GB is enough with `train.micro_batch: 1` (activation checkpointing in `windinet/wan/vae.py`).

## Run

```bash
# adapter and decoder jointly, LTX protocol (configs/wan/vae.yaml); on an 11 GB card add --set train.micro_batch=1
python scripts/wan/train_vae.py configs/wan/vae.yaml --name pairs_joint --desc "pairs, joint, 1000 updates"

# evaluate a saved adapter on the frozen VAE without training
python scripts/wan/train_vae.py configs/wan/vae.yaml --name eval --desc "..." --set train.steps=0 \
    --set load=results/wan/vae/pairs_adapter

# reproduce any run from its own config
python scripts/wan/train_vae.py results/wan/vae/pairs_joint/config.yaml --name repro --overwrite

# the first LUMI baselines: six runs on one node, one GPU each
sbatch --time=05:00:00 --partition=small-g jobs/lumi/lc/run.sbatch bash jobs/lc/one_per_gpu.sh jobs/lc/vae_baselines.txt
```

Every run needs `--name` (the results folder) and `--desc` (one sentence). Any config key can be overridden with
`--set a.b=value`; `adapter.groups` chooses which fields share an image and which are dropped. An existing folder
is not overwritten without `--overwrite`.

## Outputs

`<results_dir>/<name>/`: `config.yaml` (resolved config, re-runnable), `metrics.json` (`init` and `val`: VRMSE per
field on the validation clips; `final`: VRMSE per field on every test simulation's whole trajectory, the LTX
baseline's metric; loss curve, provenance: commit, command, versions, GPU, wall time),
`panel.png`, `curves.png`, `adapter.pt` (in git) and, for VAE fine-tunes, `vae.pt` (280 to 485 MB, not in git).
The folder's `README.md` index is regenerated after each run (`scripts/wan/index_runs.py`).

Tests: `pytest tests/test_wan_vae_stage.py`.
