# Wan 2.1 VAE on 2D-SRBA-OOOO

Shock-bubble interaction runs at 256×256, data under `datasets/srba_oooo`, settings in `configs/wan/srba_oooo/vae.yaml`.

## Dataset

| fact | value |
|---|---|
| grid | 256×256, 101 frames, 0.7 µs apart |
| cells | 3 Mach numbers (1.3, 1.7, 2.1) × 5 bubble counts (1 to 5) |
| `train.h5` | 240 sims, 16 per cell; 30 of them validation, 2 per cell, seed 42 |
| `test.h5` | 60 sims, 4 per cell |
| fields | pressure p, velocityX u, velocityY v, log density ln ρ (= ln of the stored density) |
| statistics | `results/wan/srba_oooo/stats/train_256.json`, `scripts/wan/field_stats.py` on all 240 train sims |

The statistics equal the provider's global statistics to machine precision.
The field is uniform until the shock arrives, so a sample from that phase has almost no variance.
Mid-run validation therefore divides each sim's error by the variance of its whole trajectory.
Videos follow the display rule of `docs/video_spec.md` §8.
Numbers below on the 60 test sims are for documentation; choices are made on the 30 validation sims.

## Adapter on the frozen VAE, per-channel velocity scales (runs removed)

Recipe: the adapter alone on the frozen Wan 2.1 VAE, images (ln ρ, p) and (u, v), 2000 steps of 2 single frames,
batch 1 sim, lr 3e-3, no warm-up, no gradient clipping, RMSE + H1, no augmentation, u and v each z-scored on its own
statistics (commit 6707466, job 23706863, `jobs/lc/srba_vae_adapters.txt` at that commit).
Varied: pressure linear or logged inside the adapter, and the SoftClip init quantile.
Train loss is the mean of the last 100 steps; VRMSE is over the 60 test sims, whole trajectories.

| run | p | quantile | train loss | VRMSE mean ± std | p | u | v | ln ρ | median | bench_vae VRMSE |
|---|---|---|---|---|---|---|---|---|---|---|
| `res256/pairs/adapter` (removed) | linear | 0.98 | 0.272 | 0.173 ± 0.016 | 0.060 | 0.076 | 0.491 | 0.064 | 0.175 | 0.114 |
| `res256/pairs/adapter_logp` (removed) | log | 0.98 | 0.269 | 0.168 ± 0.014 | 0.049 | 0.077 | 0.488 | 0.057 | 0.172 | 0.111 |
| `res256/pairs/adapter_q999` (removed) | linear | 0.999 | 0.056 | 0.076 ± 0.011 | 0.049 | 0.072 | 0.123 | 0.059 | 0.074 | 0.074 |
| `res256/pairs/adapter_logp_q999` (removed) | log | 0.999 | 0.055 | 0.076 ± 0.010 | 0.049 | 0.072 | 0.122 | 0.058 | 0.076 | 0.076 |

The quantile decides: 0.999 instead of 0.98 lowers the test VRMSE from 0.173 to 0.076, mostly through v (0.491 to 0.123).
Log pressure changes the mean by at most 0.005.
These adapters normalise u and v separately, so they are not valid under rotation; they were replaced by adapters with
one shared velocity scale and removed.

## Adapter with one shared velocity scale, 8 orientations

Recipe: as in the per-channel section, but u and v share one mean-free scale (dataset fact `Spec.shared_vector_scale`)
and training uses all 8 orientations (`data.augment`).
Jobs: 23707708 (four arms, `jobs/lc/srba_vae_adapters.txt`), 23708168 (quantile sweep, `jobs/lc/srba_vae_adapters_sweep.txt`),
23709196 (validation pass, `jobs/lc/srba_vae_adapters_val.txt`).
Validation VRMSE is over the 30 validation sims, whole trajectories (`<run>/val/metrics.json`, block `val`).
Test VRMSE is over the 60 test sims (`<run>/metrics.json`, block `final`).
Quantile 1.0 is the maximum of the init probe.

| run | p | quantile | val VRMSE mean ± std | p | u | v | ln ρ | test VRMSE mean | bench_vae VRMSE |
|---|---|---|---|---|---|---|---|---|---|
| `res256/pairs/adapter` | linear | 0.98 | 0.082 ± 0.010 | 0.060 | 0.041 | 0.160 | 0.066 | 0.081 | 0.078 |
| `res256/pairs/adapter_logp` | log | 0.98 | 0.076 ± 0.008 | 0.049 | 0.041 | 0.157 | 0.058 | 0.076 | 0.075 |
| `res256/pairs/adapter_q995` | linear | 0.995 | 0.075 ± 0.008 | 0.048 | 0.041 | 0.154 | 0.059 | 0.075 | 0.073 |
| `res256/pairs/adapter_q999` | linear | 0.999 | 0.075 ± 0.008 | 0.048 | 0.041 | 0.152 | 0.058 | 0.075 | 0.073 |
| `res256/pairs/adapter_logp_q999` | log | 0.999 | 0.075 ± 0.008 | 0.049 | 0.042 | 0.152 | 0.058 | 0.075 | 0.074 |
| `res256/pairs/adapter_q9999` | linear | 0.9999 | 0.075 ± 0.009 | 0.054 | 0.041 | 0.150 | 0.054 | 0.075 | 0.073 |
| `res256/pairs/adapter_q99999` | linear | 0.99999 | 0.074 ± 0.008 | 0.051 | 0.042 | 0.149 | 0.055 | 0.074 | 0.072 |
| `res256/pairs/adapter_q100` | linear | 1.0 | 0.077 ± 0.009 | 0.056 | 0.042 | 0.149 | 0.062 | 0.077 | 0.075 |

With the shared scale the quantile hardly matters between 0.995 and 0.99999: validation means 0.074 to 0.075.
Quantile 0.98 (0.082) and the maximum (0.077) are slightly worse.
Log pressure helps only at 0.98 (0.082 to 0.076); at 0.999 both give 0.075.
On the test sims, against the best per-channel adapter (`adapter_logp_q999`, removed), the mean is equal (0.075 vs 0.076),
u is better (0.041 vs 0.072) and v is worse (0.153 vs 0.122).

Choice: `res256/pairs/adapter_q999` (linear pressure, quantile 0.999).
The lowest validation mean is `adapter_q99999` (0.0741 vs 0.0751, 1.4 % lower).
Differences within the 0.995–0.99999 band are below 2 %, and for close results the simpler setting wins.
Quantile 0.999 is already the dataset default in `configs/wan/srba_oooo/vae.yaml`.
All later stages (decoders, latents, DiT) use this adapter.

## Decoders behind the chosen adapter

Recipe: encoder frozen, adapter `res256/pairs/adapter_q999` fixed (`train.adapter_lr=0`), one decoder per image
(`train.vae_per_group=true`), all 8 orientations.
1000 updates of 6 sims with one 5-frame clip each, decoder lr 1e-6 with 100 warm-up steps and cosine decay to 2 %,
gradient clipping 5, RMSE + H1 (defaults of `configs/wan/srba_oooo/vae.yaml`, commit 1925ddc, job 23709385, 3 GPUs per arm).
The arms differ only in the decoder start: the original Wan 2.1 VAE, or the eulermq decoders of
`results/wan/eulermq/vae/res256/pairs/2dec_clip33` loaded by image position (`--set vae_load=...`).
Both datasets name the vector image (x, y), but eulermq's x is the row axis and srba_oooo's x is the column axis,
so the two vector channels are swapped relative to the image axes.
Test VRMSE is over the 60 test sims, whole trajectories (`<run>/metrics.json`, block `final`).
Mid-run validation is the last one (step 1000) on short samples over the 30 validation sims, divided by each sim's
whole-trajectory variance (block `val`); the adapter's `val` block predates that division, its whole-trajectory
validation is in the table above (0.075).

| run | decoders | test VRMSE mean ± std | p | u | v | ln ρ | bench_vae VRMSE | mid-run val VRMSE mean |
|---|---|---|---|---|---|---|---|---|
| `res256/pairs/adapter_q999` | frozen, original Wan | 0.075 ± 0.007 | 0.048 | 0.041 | 0.153 | 0.058 | 0.073 | – |
| `res256/pairs/2dec` | from original Wan | 0.069 ± 0.008 | 0.042 | 0.035 | 0.157 | 0.044 | 0.068 | 0.047 |
| `res256/pairs/2dec_from_eulermq` | from eulermq `2dec_clip33` | 0.063 ± 0.006 | 0.036 | 0.031 | 0.144 | 0.041 | 0.062 | 0.047 |

Test VRMSE mean per Mach number and per bubble count (block `final`; the adapter run predates these keys):

| run | Mach 1.3 | Mach 1.7 | Mach 2.1 | 1 bubble | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|---|---|
| `res256/pairs/adapter_q999` | – | – | – | – | – | – | – | – |
| `res256/pairs/2dec` | 0.076 | 0.065 | 0.067 | 0.073 | 0.064 | 0.067 | 0.069 | 0.073 |
| `res256/pairs/2dec_from_eulermq` | 0.066 | 0.060 | 0.063 | 0.065 | 0.058 | 0.061 | 0.064 | 0.067 |

Against the frozen decoders, the test mean drops by 7.5 % from the Wan start and by 15.9 % from the eulermq start.
The eulermq start beats the Wan start on test by 9.0 % (0.063 vs 0.069) and on every field.
On mid-run validation it is 1.5 % lower (0.0468 vs 0.0475).
v remains the weakest field; from the Wan start it is slightly worse than with frozen decoders (0.157 vs 0.153).
Mach 1.3 has the highest error in both arms.

Choice: `res256/pairs/2dec_from_eulermq`.
It was made on the mid-run validation numbers, with the test numbers pointing the same way; there was no separate
whole-trajectory validation pass.
It continues with 33-frame clips as `res256/pairs/2dec_clip33` (job 23710941, the eulermq recipe: 2000 updates,
8 clips of 33 frames per update over 8 GPUs, decoder lr 3e-6); results to be added.
