# WinDiNet: Pretrained Video Models as Differentiable Physics Simulators

<p>
  <a href="https://arxiv.org/abs/2603.21210" target="_blank">
    <img src="https://img.shields.io/badge/arXiv-2603.21210-b31b1b.svg" alt="arXiv Paper"/>
  </a>
</p>

WinDiNet repurposes pretrained video diffusion models as fast, differentiable surrogates for CFD. This fork
targets **ShockWaveNet**: 4-channel compressible Euler fields (density, momentum_x, momentum_y, pressure) with
shocks, conditioned on the scalar `gamma`. Two lines of work live here:

| | Backbone | Where | Status |
|---|---|---|---|
| VAE fine-tuning and DiT training | [LTX-Video](https://github.com/Lightricks/LTX-Video) | `windinet/`, `scripts/`, `configs/{finetune_vae,dit}` | ledger in [docs/weihao/EXPERIMENTS.md](docs/weihao/EXPERIMENTS.md) |
| Field adapter for the frozen Wan 2.1 VAE | [Wan 2.1](https://github.com/Wan-Video/Wan2.1) | `windinet/{field_adapter,wan}`, `scripts/field_adapter`, `configs/field_adapter` | [docs/field_adapter/](docs/field_adapter/README.md) |

## Installation

Python 3.10+ and a CUDA or ROCm GPU.

```bash
pip install -e .              # library and scripts
pip install -e ".[training]"  # adds scipy for VAE fine-tuning
```

LTX-Video and Wan weights download into the Hugging Face cache. Set `WINDINET_HF_CACHE=/path` to put them
elsewhere ([pretrained/README.md](pretrained/README.md)).

## Layout

```
windinet/           library: VAE adapter, losses, scalar conditioning, training; field_adapter/, wan/
scripts/            entry points (finetune_vae.py, preprocess_dataset.py, train.py, inference_shockwave.py,
                    field_adapter/train_adapter.py)
configs/            YAML per experiment family: finetune_vae/, dit/, field_adapter/
jobs/<cluster>/     Slurm launchers per cluster (lundquist, sng_pvc, lrz_ai, jupiter, lumi)
docs/               field_adapter/ (README, ablations), weihao/ (experiment ledger, infra notes, thesis prep)
results/            field adapter runs: one folder per run with config, metrics, figures, adapter weights
finetune_vae_outputs/, dit_preprocessed/, dit_outputs/, logs/   tracked outputs and logs of the LTX pipeline
euler_mq_dataset/   download scripts; the HDF5 data itself is not in git
tests/              pytest
```

## LTX pipeline

Three stages; each config's `output_dir` receives checkpoints, panels, metrics and the resolved config.

```bash
# 1. fine-tune the VAE on the 4-channel fields
python scripts/finetune_vae.py configs/finetune_vae/finetune_vae_baseline.yaml

# 2. encode the dataset into latents with the fine-tuned VAE
python scripts/preprocess_dataset.py /path/to/train.h5 --output-dir /path/to/preprocessed \
    --inflate-checkpoint <output_dir>/checkpoints/vae_shockwave_best.safetensors --eval-sims 675

# 3. train the DiT on the latents, then roll out from an initial frame and gamma
python scripts/train.py configs/dit/train_dit.yaml
python scripts/inference_shockwave.py configs/dit/inference_dit.yaml --h5 /path/to/train.h5 --out_dir predictions/
```

The dataset layout is `<sample_id>/{density,momentum_x,momentum_y,pressure}` in HDF5
(`windinet/training/shockwave_data.py`). Inference refuses a VAE that did not produce the DiT's latents
(`latent_provenance.json`). Two changes to LTX-Video make this work: the 3-to-4-channel VAE adapter
(`windinet/vae_adapter.py`) and Fourier-feature scalar conditioning in place of text
(`windinet/scalar_embeddings.py`).

## Wan field adapter

A 14-parameter per-pixel map from the four fields to two RGB images lets the frozen Wan 2.1 VAE carry CFD
fields at 3 to 10 % error per field; fine-tuning its decoder behind the adapter brings the best run to 6 %.
Commands, outputs and the run index: [docs/field_adapter/README.md](docs/field_adapter/README.md). Findings:
[docs/field_adapter/ablations.md](docs/field_adapter/ablations.md).

## Clusters

Each `jobs/<cluster>/` launcher patches a config through `windinet/cluster_config.py` (data path, workers,
effective batch). Per user, set `WINDINET_WORK=/scratch/.../<you>_work` so outputs land under your own
directory, and `WINDINET_HF_CACHE` for the weights. Cluster notes: [docs/weihao/infra.md](docs/weihao/infra.md).

## Acknowledgements

Built on [LTX-Video-Trainer](https://github.com/Lightricks/LTX-Video-Trainer) by Lightricks, Apache 2.0.
