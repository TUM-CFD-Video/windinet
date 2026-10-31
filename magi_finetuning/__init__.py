"""MAGI-1 4.5B finetuning on the Euler (ShockWave) dataset, using MAGI-1's official code (IG_work/MAGI-1).

    compat.py      stand-ins for flash_attn / flashinfer / timm + training patches (imported first, below)
    settings.py    paths and fixed hyperparameters
    data.py        train/val split table, channel normalization, training loader, latent layout
    vae.py         4-channel VAE (grown layers or windinet's adapters), loading a trained one
    dit.py         MAGI-1 DiT + gamma embedding, flow matching (noise, loss), rollout, loading a trained one

    train/         finetune.py   DiT entry point: LoRA (train_lora.py) or --full-finetune (train_full.py)
                   train_vae.py  stage 1: adapt the VAE to the 4 fields
    eval/          evaluate.py   test.h5: rollouts, VRMSE tables, figures, videos; archives the final models
                   report.py     plots, videos, tables
    utils/         runtime.py    multi-GPU setup, job timing, pre-flight checks
                   resume.py     exact resuming across 12 h jobs
                   download.py   fetch MAGI-1 4.5B weights;  channel_stats.py  normalization stats of the train split
    tests/         test_attention.py, test_changes.py (run via jobs/jupiter/magi1/quick_test.sbatch)

Entry points are run as modules from the repo root, e.g. python -m magi_finetuning.train.finetune; the
sbatch files in jobs/jupiter/magi1/ do this.
"""

import magi_finetuning.compat  # noqa: F401  -- must run before anything imports inference.*
