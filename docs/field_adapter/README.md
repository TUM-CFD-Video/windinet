# Field adapter for the frozen Wan 2.1 VAE

A 14-parameter per-pixel map from the four Euler fields (ρ, m_x, m_y, p) to two RGB images, so the
Wan 2.1 VAE can carry CFD fields; optionally the VAE decoder is fine-tuned behind it.
What was found: [ablations.md](ablations.md). Every run: [results index](../../results/field_adapter/20_wan_roundtrip/README.md).

## Setup

```bash
pip install -e .
```

- Data: `euler_mq_dataset/256x256_ds/test.h5` (not in git). Its channel statistics are committed
  (`results/field_adapter/00_stats/`); recompute with `scripts/compute_channel_stats.py`.
- Natural-video reference statistics for the Fréchet metric are committed
  (`results/field_adapter/30_reference/wan_ref_stats.pt`); to regenerate, run `download_pexels.py`
  (Pexels API key; clips land in `30_reference/pexels/`, not in git) and `ref_latents.py` in `scripts/field_adapter/`.
- GPU: fp32, 11 GB is enough (activation checkpointing in `windinet/wan/vae.py`).

## Run

```bash
# train the adapter (20 min on a 2080 Ti)
python scripts/field_adapter/train_adapter.py configs/field_adapter/eulermq.yaml \
    --name pairs_600 --desc "pairs, lr 3e-3, 600 steps"

# fine-tune the Wan decoder behind a fixed adapter (40 min)
python scripts/field_adapter/train_adapter.py configs/field_adapter/eulermq_finetune.yaml \
    --name dec_clip5 --desc "decoder on 5-frame clips" \
    --set data.clip=true --set data.frames_per_sim=5 --set train.batch_sims=3

# evaluate a saved adapter without training
python scripts/field_adapter/train_adapter.py configs/field_adapter/eulermq.yaml \
    --name eval --desc "..." --set train.steps=0 \
    --set adapter.load=results/field_adapter/20_wan_roundtrip/wan_pairs_2000/adapter.pt

# reproduce any run from its own config
python scripts/field_adapter/train_adapter.py \
    results/field_adapter/20_wan_roundtrip/wan_ft_decoder_pairs2000/config.yaml --name repro
```

Every run needs `--name` (the results folder) and `--desc` (one sentence). On a cluster, prefix the command with
`sbatch jobs/lumi/lc/run.sbatch` or `sbatch jobs/jupiter/lc/run.sbatch`. Any config key can be
overridden with `--set a.b=value`. An existing folder is not overwritten without `--overwrite`.

## Outputs

`results/field_adapter/20_wan_roundtrip/<name>/`: `config.yaml` (resolved config, re-runnable),
`metrics.json` (VRMSE per field, Fréchet distances, loss curve, provenance: commit, command,
versions, GPU, wall time), `panel.png`, `curves.png`, `adapter.pt` (in git) and, for fine-tunes,
`vae.pt` (280 to 485 MB, not in git). The folder's `README.md` index is regenerated after each run
(`scripts/field_adapter/index_runs.py`).

Tests: `pytest tests/test_field_adapter.py`.
