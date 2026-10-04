# WinDiNet: Pretrained Video Models as Differentiable Physics Simulators

<p>
  <a href="https://arxiv.org/abs/2603.21210" target="_blank">
    <img src="https://img.shields.io/badge/arXiv-2603.21210-b31b1b.svg" alt="arXiv Paper"/>
  </a>
</p>

WinDiNet repurposes the [LTX-Video](https://github.com/Lightricks/LTX-Video) video diffusion transformer as a fast, differentiable surrogate for computational fluid dynamics (CFD) simulations. This fork adapts the original urban-wind-flow model (2-channel u/v velocity + building mask) to **ShockWaveNet**: 4-channel compressible Euler CFD fields (density, momentum_x, momentum_y, pressure) with shocks, 256x256, conditioned on the scalar `gamma`.

The final results of the master's thesis (VAE fine-tuning study + full VAE/DiT pipeline, Chapter 6) are in [THESIS_RESULTS.md](THESIS_RESULTS.md). The thesis LaTeX tables, appendices and figures, and the full experiment history (every run, config, log and the old experiment ledgers), are preserved in the `thesis-final` git tag.

## Installation

**Prerequisites:** Python 3.10+, a GPU/XPU with 48+ GB memory for training.

```bash
pip install -e .

# For training (adds decord, pandas, scipy):
pip install -e ".[training]"
```

### Base weights

By default, LTX-Video weights download to `huggingface_hub`'s own cache
(`~/.cache/huggingface`). Set `WINDINET_HF_CACHE=/path/to/dir`
(`windinet/paths.py`) to redirect them elsewhere -- e.g. jupiter points it at
a shared project-storage cache populated by `jobs/jupiter/download_pretrained.sh`.

## Training

Training has two stages: (1) finetuning the VAE to reconstruct the 4-channel CFD fields, then (2) training the diffusion transformer (DiT) on the resulting latents.

### Stage 1: VAE finetuning

```bash
python scripts/finetune_vae.py configs/finetune_vae/finetune_vae_ch6_loss_rmse_h1_256res.yaml
```

This is the thesis baseline ("Full FT": whole encoder + decoder, 1.0·RMSE + 50·H1, AdamW LR 5e-5 cosine, 20 epochs). The other `configs/finetune_vae/finetune_vae_ch6_*` files are the Chapter 6 ablation arms, each changing exactly one setting. Edit `data.data_root` to point at your shockwave HDF5 dataset (`<sample_id>/{density,momentum_x,momentum_y,pressure}`, see `windinet/training/shockwave_data.py`). Outputs under `output_dir`:

```
<output_dir>/
    checkpoints/vae_shockwave_{best,last,epoch###}.safetensors
    checkpoints/*.state.pt        # optimizer/scheduler/RNG, for resume_from
    visualizations/epoch_####/    # GT/prediction/residual panels
    metrics/{metrics.csv,loss_curves.png}
    training_config.yaml
```

Test-set evaluation: `scripts/eval_vae_test.py` (writes `test_eval_256.json`).

### Stage 2: Data preprocessing

Encode the CFD fields into VAE latents for DiT training:

```bash
python scripts/preprocess_dataset.py /path/to/shockwave_dataset/train.h5 \
    --output-dir /path/to/preprocessed \
    --inflate-checkpoint <output_dir>/checkpoints/vae_shockwave_best.safetensors \
    --eval-sims 500
```

This must use the *finetuned* VAE checkpoint from stage 1 -- see `scripts/preprocess_dataset.py`'s docstring for the output layout (`latents/`, `scalars/`, `normalization.json`, and a `val/` split when `--eval-sims` is set).

### Stage 3: DiT training

```bash
python scripts/train.py configs/dit/train_dit_sng_pvc_ch6_loss_rmse_h1.yaml
```

Set `data.preprocessed_data_root` (and `validation.data_root` to the `val/` split from preprocessing) and `output_dir` in the config. DiT training consumes precomputed latents only and records which VAE checkpoint produced them in `<output_dir>/latent_provenance.json`. `configs/dit/train_dit_sng_pvc_ch6_*` are the per-VAE DiT runs of Chapter 6; `configs/dit/train_dit.yaml` is the generic template.

## Evaluation and inference

```bash
# Pixel-space VRMSE of a full VAE + DiT rollout on the test set
python scripts/eval_dit_vrmse.py configs/dit/inference_dit.yaml --help

# Roll out predictions as .npz fields, then render them
python scripts/inference_shockwave.py configs/dit/inference_dit.yaml \
    --h5 euler_mq_dataset/256x256_ds/test.h5 --out_dir predictions/ --num_samples 8
python scripts/visualize_dit_predictions.py --pred_dir predictions/ ...
```

Inference conditions on frame 0 of a simulation (encoded with the VAE posterior mean) plus `gamma` and rolls out the remaining frames. The VAE checkpoint in the config must be the exact one the DiT's latents were encoded with -- the pipeline refuses to decode (`verify_latent_space`) on a latent-space fingerprint mismatch.

The thesis tables can be regenerated with `scripts/thesis_tables.py`.

## Repository layout

| Path | Contents |
|---|---|
| `windinet/` | Package: VAE adapter, losses, scalar conditioning, VAE/DiT trainers, inference pipeline |
| `scripts/` | Entry points: training, preprocessing, evaluation, latent diagnostics, thesis tables/figures |
| `configs/` | Final Chapter 6 VAE and DiT configs |
| `jobs/{sng_pvc,jupiter}/` | Slurm launchers (`jobs/sng_pvc/ch6_submit.sh` submits the Chapter 6 runs); per-cluster defaults in `windinet/cluster_config.py` |
| `finetune_vae_outputs/`, `logs/` | Metrics, test evals and job logs of the final runs (weights are not tracked) |
| `THESIS_RESULTS.md` | Final thesis numbers |
| `docs/` | Cluster infrastructure notes |

## Architecture

WinDiNet modifies LTX-Video in two ways:

1. **VAE channel adapter** (`windinet/vae_adapter.py`): grows the pretrained 3-channel `conv_in`/`conv_out` to 4 channels (`inflate` mode, the new pressure slot zero-initialised) so the CFD fields pass through natively, then finetunes the VAE with reconstruction losses (`windinet/losses/`, weighted via `windinet/loss_weighting/`), optionally with KL or SDS latent regularisation (`windinet/training/sds_loss.py`).
2. **Scalar conditioning** (`windinet/scalar_embeddings.py`): replaces text conditioning with Fourier-feature-encoded scalar inputs (currently just `gamma`), enabling physical parameterization instead of prompts.

## Acknowledgements

Built on [LTX-Video-Trainer](https://github.com/Lightricks/LTX-Video-Trainer) by Lightricks, licensed under Apache 2.0.
