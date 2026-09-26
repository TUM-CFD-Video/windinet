# Chapter 6 — Experiment Plan

Status as of 2026-09-25 (results pulled at commit 6e9a638). The plan is
below. Finished VAE results are in "VAE results" at the end of this file.

## Shared protocol

- **Baseline VAE:** `finetune_vae_ch6_loss_rmse_h1_256res`. It uses full-structure
  FT, the loss rmse 1.0 + h1 50.0 with SSIM off, and LR 5e-5 with a cosine
  schedule to 1e-6. Other settings: 20 epochs, effective batch 32, seed 42.
- **Data:** Euler_MQ `256x256_ds` (the exception is the resolution arm below).
  500 sims are held out (`eval_sims=500`), and DiT preprocessing uses
  `EVAL_SIMS=500` to match.
- **Headline metrics:** `val_vrmse_chmean` for the VAE (pooled `val_vrmse`
  is reported alongside). For the DiT, pixel `vae_dit_vrmse_chmean`.
- **Cluster / launcher:** sng_pvc, using `jobs/sng_pvc/ch6_submit.sh`.
- **SSIM and MLW:** SSIM stays out of the baseline. The MLW arm is dropped.
  Runs trained with the old (buggy) SSIM are not compared against this baseline.

## Group 1 — VAE adaptation strategy

| Arm | Config | Status |
|---|---|---|
| Unfinetuned reference | `finetune_vae_ch6_unfinetuned_256res` | done (543419). It ran with epochs=0, so it has no VAE metrics; its only role is to supply the DiT arm |
| Full-structure FT (= baseline) | `finetune_vae_ch6_loss_rmse_h1_256res` | done (543421) |
| Decoder-only | `finetune_vae_ch6_decoder_only_nossim_256res` | done (545277) |
| Color adapter | `finetune_vae_ch6_color_adapter_nossim_256res` | done (545280) |

## Group 2 — VAE loss ablation

| Arm | Config | Status |
|---|---|---|
| RMSE only | `finetune_vae_ch6_loss_rmse_only_256res` | done (543420) |
| RMSE + H1 (baseline) | `finetune_vae_ch6_loss_rmse_h1_256res` | done (543421) |
| RMSE + H1 + fixed SSIM | `finetune_vae_ch6_loss_rmse_h1_ssimfix_256res` | done (545281) |

## Group 3 — LR sensitivity

| Arm | Config | Status |
|---|---|---|
| 1e-5 | `finetune_vae_ch6_lr_1e5_nossim_256res` | done (545279) |
| 5e-5 (baseline) | `finetune_vae_ch6_loss_rmse_h1_256res` | done |
| 1e-4 | `finetune_vae_ch6_lr_1e4_nossim_256res` | **discarded**: diverged in epoch 2 (see "Discarded runs") |

## Training-resolution comparison

The baseline is trained on `128x128_ds` instead of 256. Only the VAE is
trained here, with no DiT and no cross-resolution eval. Its `val_vrmse` is
compared against the 256res baseline.

| Arm | Config | Status |
|---|---|---|
| 128res baseline | `finetune_vae_ch6_loss_rmse_h1_128res` | running (545466). At the last pull it was in epoch 11/20, with val_vrmse 0.106 |
| 256res baseline | `finetune_vae_ch6_loss_rmse_h1_256res` | done |

## Group 5 — KL / SDS latent regularization

These arms run as a full pipeline: VAE, then encode, then DiT, chained with
afterok via `ch6_submit.sh pipeline g5`. SDS uses the stock teacher only
(raw LTXV 2B), and the sds_cfd arm is dropped.

| Arm | VAE config | VAE status | DiT status |
|---|---|---|---|
| KL 1e-7 | `finetune_vae_ch6_kl_1e7_256res` | done (545286) | encode running (545287), then DiT |
| KL 1e-6 | `finetune_vae_ch6_kl_1e6_256res` | **discarded**: diverged in epoch 2 | cancel encode 545290 and its DiT: they would use the epoch-1 checkpoint |
| KL 1e-5 | `finetune_vae_ch6_kl_1e5_256res` | done (545292) | encode running (545293), then DiT |
| SDS stock w=1e-4 | `finetune_vae_ch6_sds_stock_w1e4_256res` | done (545295) | encode running (545296), then DiT |
| SDS stock w=1e-2 | `finetune_vae_ch6_sds_stock_w1e2_256res` | done (545298) | encode running (545299), then DiT |
| SDS stock w=1 | `finetune_vae_ch6_sds_stock_w1_256res` | done (545301) | encode running (545302), then DiT |

The KL weight is applied in the `kl_1e7` arm. The `kl=0.0000` in the log is
the "Loss weights" line, which prints the weight 1e-7 rounded to 4 decimals.
The raw val_kl falls from 2.6e3 to 814 over training, against ~7.7e5 for the
baseline.

## Group 4 — DiT compatibility (lower priority; the VAE chapter comes first)

Every arm reuses an existing VAE, so this group needs no new VAE training.

| Arm | DiT config | Status |
|---|---|---|
| Baseline VAE | `train_dit_sng_pvc_ch6_loss_rmse_h1` | training (545283) |
| Unfinetuned VAE | `train_dit_sng_pvc_ch6_unfinetuned` | training (545285) |
| One KL arm | from Group 5 | picked once the Group 5 results are in |
| One SDS arm | from Group 5 | picked once the Group 5 results are in |

Once each DiT job has finished, run
`bash jobs/sng_pvc/ch6_submit.sh eval g4_ready g5`.

## Pre-Chapter-6 256res SDS weight sweep (reference only)

The runs are `finetune_vae_whole_structure_baseline_ep20_256res_sds_{w0p0001,w0p01,w1}`
plus their DiTs, and the DiT eval is finished (545274–545276). These VAEs were
trained with the old SSIM, so the sweep is only compared within itself and is
not tabled against the Ch6 baseline.

## Open items for the VAE chapter

- [ ] VAE-only test-set reconstruction eval: script written
      (`scripts/eval_vae_test.py`, all 500 sims of `256x256_ds/test.h5`,
      pooled/chmean/per-channel VRMSE, per-sim values kept). Submit with
      `bash jobs/sng_pvc/ch6_submit.sh vaetest vae_all` (13 arms, including
      unfinetuned and the 128res baseline), then build the table with
      `python scripts/vae_test_table.py`. Results pending.
- [ ] Seed noise floor at 256res, from extra seeds of the baseline. The only
      floor measured so far is ~1.1%, at 128res.
- [ ] Qualitative panels: rsync the visualizations from the cluster.
- [ ] Table script that builds the thesis tables from each run's `metrics/metrics.csv`.
- [ ] 128res baseline: add its row once 545466 finishes.

## VAE results (256res, 500 validation sims)

Every value is from the checkpoint with the best `val_vrmse`, which is also
the one DiT preprocessing encodes with. For every run below, that is the
epoch-20 checkpoint.

- `chmean` is the mean of the four per-channel VRMSEs. The baseline was run
  before the `val_vrmse_chmean` column existed, so its value is derived from
  the four `val_vrmse_<channel>` columns. That derivation is exact: both
  quantities are averaged over the same sims. It was cross-checked against
  the logged column in every run that has it.
- Δ is relative to the baseline's chmean.
- The only measured seed noise floor is about 1.1%, and it was measured at
  128res. Differences within about ±1% should therefore be read as ties.

| Group | Arm | val_vrmse | chmean | Δ chmean | density | mom_x | mom_y | pressure |
|---|---|---|---|---|---|---|---|---|
| G1/G2/G3 | **Baseline**: full FT, RMSE+H1, LR 5e-5 | 0.0575 | **0.0791** | — | 0.0709 | 0.0700 | 0.0705 | 0.1051 |
| G1 | Decoder-only | 0.1013 | 0.1403 | +77.4% | 0.0956 | 0.0867 | 0.0938 | 0.2853 |
| G1 | Color adapter | 0.4042 | 0.4691 | +493% | 0.1031 | 0.0846 | 0.0921 | 1.5967 |
| G2 | RMSE only | 0.0587 | 0.0805 | +1.8% | 0.0748 | 0.0713 | 0.0726 | 0.1036 |
| G2 | RMSE+H1+fixed SSIM (0.15) | 0.0639 | 0.0879 | +11.1% | 0.0787 | 0.0780 | 0.0787 | 0.1161 |
| G3 | LR 1e-5 | 0.0666 | 0.0928 | +17.3% | 0.0785 | 0.0765 | 0.0780 | 0.1382 |
| G5 | KL 1e-7 | 0.0578 | 0.0795 | +0.5% | 0.0710 | 0.0700 | 0.0707 | 0.1064 |
| G5 | KL 1e-5 | 0.0633 | 0.0868 | +9.8% | 0.0788 | 0.0787 | 0.0785 | 0.1113 |
| G5 | SDS stock 1e-4 | 0.0570 | 0.0784 | −0.9% | 0.0705 | 0.0696 | 0.0702 | 0.1034 |
| G5 | SDS stock 1e-2 | 0.0580 | 0.0797 | +0.8% | 0.0717 | 0.0708 | 0.0710 | 0.1056 |
| G5 | SDS stock 1 | 0.0746 | 0.1029 | +30.0% | 0.0943 | 0.0933 | 0.0939 | 0.1299 |

What the table shows (VAE reconstruction only; DiT results are still pending):

- **G1:** full FT is clearly best. Decoder-only is +77% worse, and the gap is
  mostly in pressure (0.285 vs 0.105). Its encoder is completely frozen, and
  the pressure input slot is zero-initialised, so the encoder never sees
  pressure. The color adapter fails on pressure (1.60) while its other three
  channels stay in the decoder-only range.
- **G2:** H1 gives a small gain over RMSE only (+1.8% without it, which is
  close to the noise floor). The fixed SSIM at weight 0.15 costs +11%, which
  supports keeping SSIM out of the baseline.
- **G3:** LR 1e-5 is +17% worse, with pressure worst affected. The LR 1e-4 arm
  diverged (see below), so on this side we can only say that 5e-5 is near the
  upper stable limit.
- **G5 KL:** at 1e-7 it ties the baseline while cutting val_kl by roughly
  1000x (~7.7e5 to 814). At 1e-5 it costs +9.8%, with val_kl at 55.
- **G5 SDS:** 1e-4 and 1e-2 tie the baseline. Weight 1 costs +30%.
- Whether any KL or SDS arm helps downstream can only be judged from
  `vae_dit_vrmse_chmean` once the Group 5 DiTs are evaluated.

### Discarded runs

`kl_1e6` and `lr_1e4_nossim` are excluded from all tables. Both collapsed in
epoch 2, right after the 200-step warmup reached peak LR: val_vrmse jumped
from about 0.15 to about 1.07, which means a near-constant output. Neither
recovered. At epoch 20 `kl_1e6` was at 0.229 and `lr_1e4_nossim` at 0.487,
with val_kl at 5.8e10.

The collapse of `kl_1e6` is a training instability rather than an effect of
the KL term: the 1e-7 and 1e-5 arms, which bracket it, both trained normally.
For `kl_1e6` the best checkpoint is from epoch 1 (val_vrmse 0.154), so the
encode and DiT jobs chained after it would run on an epoch-1 VAE.
