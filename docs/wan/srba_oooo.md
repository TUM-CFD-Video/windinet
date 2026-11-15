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
The samples have the training clip length, so the 33-frame row's mid-run value is not comparable to the 5-frame rows.

| run | decoders | test VRMSE mean ± std | p | u | v | ln ρ | bench_vae VRMSE | mid-run val VRMSE mean |
|---|---|---|---|---|---|---|---|---|
| `res256/pairs/adapter_q999` | frozen, original Wan | 0.075 ± 0.007 | 0.048 | 0.041 | 0.153 | 0.058 | 0.073 | – |
| `res256/pairs/2dec` | from original Wan | 0.069 ± 0.008 | 0.042 | 0.035 | 0.157 | 0.044 | 0.068 | 0.047 |
| `res256/pairs/2dec_from_eulermq` | from eulermq `2dec_clip33` | 0.063 ± 0.006 | 0.036 | 0.031 | 0.144 | 0.041 | 0.062 | 0.047 |
| `res256/pairs/2dec_clip33` | from `2dec_from_eulermq`, 33-frame clips | 0.043 ± 0.008 | 0.028 | 0.018 | 0.091 | 0.036 | 0.044 | 0.042 |

Test VRMSE mean per Mach number and per bubble count (block `final`; the adapter run predates these keys):

| run | Mach 1.3 | Mach 1.7 | Mach 2.1 | 1 bubble | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|---|---|
| `res256/pairs/adapter_q999` | – | – | – | – | – | – | – | – |
| `res256/pairs/2dec` | 0.076 | 0.065 | 0.067 | 0.073 | 0.064 | 0.067 | 0.069 | 0.073 |
| `res256/pairs/2dec_from_eulermq` | 0.066 | 0.060 | 0.063 | 0.065 | 0.058 | 0.061 | 0.064 | 0.067 |
| `res256/pairs/2dec_clip33` | 0.039 | 0.042 | 0.049 | 0.036 | 0.039 | 0.043 | 0.048 | 0.052 |

Against the frozen decoders, the test mean drops by 7.5 % from the Wan start and by 15.9 % from the eulermq start.
The eulermq start beats the Wan start on test by 9.0 % (0.063 vs 0.069) and on every field.
On mid-run validation it is 1.5 % lower (0.0468 vs 0.0475).
v remains the weakest field; from the Wan start it is slightly worse than with frozen decoders (0.157 vs 0.153).
Mach 1.3 has the highest error in both arms.

Choice: `res256/pairs/2dec_from_eulermq`.
It was made on the mid-run validation numbers, with the test numbers pointing the same way; there was no separate
whole-trajectory validation pass.
It continues with 33-frame clips as `res256/pairs/2dec_clip33` (job 23711640, the eulermq recipe: 2000 updates,
8 sims with one 33-frame clip each per update over 8 GPUs, decoder lr 3e-6 with 100 warm-up steps and cosine decay to
2 %, gradient clipping 5, commit c13375c; 14.4 s/update, 8.1 h).
Job 23710941 with the same command was cancelled after 36 min: its 8 h limit was too short at 14.5 s/update.

The 33-frame continuation lowers the test mean by 31 % against `2dec_from_eulermq` (0.043 vs 0.063).
v drops by 36 % (0.091 vs 0.144) and stays the weakest field.
The order over Mach numbers turns: Mach 2.1 now has the highest error (0.049) and Mach 1.3 the lowest (0.039).
The error grows with the bubble count, from 0.036 at one bubble to 0.052 at five.

## DiT

Setup: Wan 2.1 T2V 1.3B on the latents of `res256/pairs/adapter_q999` at frame stride 2, train sims under all 8
orientations (1680 train files, 30 validation, 60 test; job 23709387, `jobs/lc/srba_dit.txt`).
Conditioning replaces the text with one embedding per input (4 tokens each): Mach number on [1, 2.5], bubble count on
[0, 6] and the two components of the shock direction on [-1, 1], turned with the orientation
(`windinet/data/specs.py`, `windinet/wan/dit/conditioning.py`).
Two starting points: the original Wan 2.1 1.3B (`wan16k`), and the eulermq DiT
`results/wan/eulermq/dit/res256/pairs/unrolled/unrolled2` (`eulermq16k`), whose transformer weights load while
conditioning and AdamW start fresh.
Both run 16000 updates of 32 sims at lr 1e-4 (100 warm-up steps, cosine to 1e-7), gradient clipping 1, commit c13375c
(jobs 23709994 and 23709995, `configs/wan/srba_oooo/dit.yaml`).
Each continues as `<run>/unrolled`: 4000 updates at lr 1e-5 keeping its AdamW moments, half of each micro-batch on its
own sampler state (unrolled 0.5), decoders `res256/pairs/2dec_clip33` (jobs 23738943 and 23738944).
A rollout gets frame 0 alone as context (one clean latent frame) and predicts frames 2 to 96 at stride 2 in one shot;
scores start at frame 10 (`score_from: 10`).

Validation VRMSE over the 30 validation sims during the 16k stage (`<run>/metrics.json`, block `history`), 2 sampling
steps, decoders of `adapter_q999` (the frozen original Wan decoders):

| run | statistic | 1000 | 2000 | 4000 | 8000 | 12000 | 16000 |
|---|---|---|---|---|---|---|---|
| `res256/pairs/recipe/wan16k` | mean | 0.805 | 0.688 | 0.631 | 0.176 | 0.157 | 0.123 |
| `res256/pairs/recipe/wan16k` | median | 0.724 | 0.645 | 0.468 | 0.126 | 0.116 | 0.114 |
| `res256/pairs/recipe/eulermq16k` | mean | 0.809 | 0.570 | 0.216 | 0.187 | 0.174 | 0.157 |
| `res256/pairs/recipe/eulermq16k` | median | 0.727 | 0.473 | 0.159 | 0.132 | 0.112 | 0.110 |

Validation VRMSE during the unrolled stage, 2 sampling steps, decoders `2dec_clip33`; update 0 is the 16k weights
(block `init`):

| run | statistic | 0 | 1000 | 2000 | 3000 | 4000 |
|---|---|---|---|---|---|---|
| `res256/pairs/recipe/wan16k/unrolled` | mean | 0.103 | 0.104 | 0.104 | 0.102 | 0.102 |
| `res256/pairs/recipe/wan16k/unrolled` | Mach 2.1 / 1 bubble | 0.134 / 0.072 | 0.136 / 0.073 | 0.134 / 0.074 | 0.133 / 0.072 | 0.133 / 0.071 |
| `res256/pairs/recipe/eulermq16k/unrolled` | mean | 0.138 | 0.101 | 0.100 | 0.100 | 0.100 |
| `res256/pairs/recipe/eulermq16k/unrolled` | Mach 2.1 / 1 bubble | 0.244 / 0.265 | 0.130 / 0.077 | 0.130 / 0.077 | 0.131 / 0.078 | 0.132 / 0.078 |

Test results over the 60 test sims (`<run>/metrics.json`, block `final`).
The 16k and unrolled rows are each run's own final evaluation; `bench_steps5_eta1` is a separate evaluation of the final
weights (`configs/wan/dit_eval.yaml`: 5 steps, eta 1; job 23738942 for the 16k runs, job 23764408 for the unrolled
runs).
Per-field VRMSE is The Well's metric on frames 10 to 96; bench is cfd_bench (Harish) on frames 10, 20, ..., 90.
VAE share: the decoded simulation latents against the simulation; DiT share: the rollout against the decoded latents.

| run | decoders | steps | VRMSE mean ± std | median | p | u | v | ln ρ | bench VRMSE | bench RMSE | bench H1 | VAE share | DiT share |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `res256/pairs/recipe/wan16k` | `adapter_q999` | 2 | 0.122 ± 0.030 | 0.111 | 0.067 | 0.059 | 0.259 | 0.103 | 0.118 | 0.025 | 0.640 | 0.092 | 0.086 |
| `res256/pairs/recipe/eulermq16k` | `adapter_q999` | 2 | 0.117 ± 0.028 | 0.110 | 0.065 | 0.057 | 0.247 | 0.101 | 0.114 | 0.023 | 0.636 | 0.092 | 0.080 |
| `res256/pairs/recipe/wan16k/bench_steps5_eta1` | `2dec_clip33` | 5, eta 1 | 0.100 ± 0.035 | 0.094 | 0.053 | 0.046 | 0.217 | 0.085 | 0.096 | 0.021 | 0.516 | 0.068 | 0.075 |
| `res256/pairs/recipe/eulermq16k/bench_steps5_eta1` | `2dec_clip33` | 5, eta 1 | 0.096 ± 0.031 | 0.089 | 0.051 | 0.045 | 0.206 | 0.083 | 0.092 | 0.020 | 0.511 | 0.068 | 0.069 |
| `res256/pairs/recipe/wan16k/unrolled` | `2dec_clip33` | 2 | 0.101 ± 0.034 | 0.095 | 0.053 | 0.046 | 0.221 | 0.085 | 0.097 | 0.021 | 0.516 | 0.068 | 0.076 |
| `res256/pairs/recipe/eulermq16k/unrolled` | `2dec_clip33` | 2 | 0.097 ± 0.031 | 0.091 | 0.051 | 0.045 | 0.210 | 0.083 | 0.094 | 0.020 | 0.512 | 0.068 | 0.071 |
| `res256/pairs/recipe/wan16k/unrolled/bench_steps5_eta1` | `2dec_clip33` | 5, eta 1 | 0.100 ± 0.035 | 0.094 | 0.053 | 0.046 | 0.217 | 0.085 | 0.096 | 0.021 | 0.515 | 0.068 | 0.074 |
| `res256/pairs/recipe/eulermq16k/unrolled/bench_steps5_eta1` | `2dec_clip33` | 5, eta 1 | 0.096 ± 0.031 | 0.089 | 0.051 | 0.045 | 0.206 | 0.083 | 0.092 | 0.020 | 0.510 | 0.068 | 0.069 |

Test VRMSE mean of the 5-step evaluations per Mach number, per bubble count and per frame window:

| run | Mach 1.3 | Mach 1.7 | Mach 2.1 | 1 bubble | 2 | 3 | 4 | 5 | frames 9–26 | 27–56 | 57–100 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `res256/pairs/recipe/wan16k/bench_steps5_eta1` | 0.078 | 0.095 | 0.128 | 0.071 | 0.083 | 0.092 | 0.116 | 0.139 | 0.043 | 0.082 | 0.123 |
| `res256/pairs/recipe/eulermq16k/bench_steps5_eta1` | 0.075 | 0.091 | 0.122 | 0.070 | 0.080 | 0.090 | 0.111 | 0.129 | 0.043 | 0.079 | 0.117 |
| `res256/pairs/recipe/wan16k/unrolled/bench_steps5_eta1` | 0.078 | 0.095 | 0.128 | 0.071 | 0.082 | 0.092 | 0.116 | 0.139 | 0.043 | 0.081 | 0.123 |
| `res256/pairs/recipe/eulermq16k/unrolled/bench_steps5_eta1` | 0.075 | 0.092 | 0.122 | 0.070 | 0.080 | 0.090 | 0.111 | 0.129 | 0.043 | 0.079 | 0.117 |

Comparison: Weihao's LTX pipeline on this dataset gives cfd_bench VRMSE 0.148 for the DiT and 0.072 for the VAE on his
frames 40 to 90 (reported by Weihao, not re-run here); our frame window differs, so the comparison is approximate.

Findings:

- Both starts beat the LTX DiT number on bench VRMSE: 0.096 (`wan16k`) and 0.092 (`eulermq16k`) vs 0.148 at 5 steps.
- On the VAE side, bench VRMSE is 0.044 for the round trip of `2dec_clip33` and 0.067 inside the DiT evaluation, vs
  0.072 for LTX.
- The eulermq start is ahead on test in every row: by 4.1 % at 5 steps (0.096 vs 0.100) and 3.7 % at 2 steps
  (0.117 vs 0.122).
- At the end of the 16k stage the eulermq start had the higher validation mean (0.157 vs 0.123) at a lower median
  (0.110 vs 0.114).
- That mean came from outlier rollouts: validation std 0.205 vs 0.033, Mach 2.1 0.258 and one bubble 0.287.
- `wan16k` had such outliers too until update 14000 (std 0.131), and lost them by update 15000 (std 0.034).
- The decoder switch to `2dec_clip33` alone took the eulermq validation mean from 0.157 to 0.138; the first 1000
  unrolled updates took it to 0.101, with Mach 2.1 0.244 → 0.130 and one bubble 0.265 → 0.077.
- For `wan16k` the unrolled stage changed the validation mean by 0.001 (0.103 → 0.102).
- At 5 steps the unrolled stage leaves the test numbers unchanged: mean 0.100 → 0.100 (`wan16k`) and 0.096 → 0.096
  (`eulermq16k`), bench VRMSE 0.096 → 0.096 and 0.092 → 0.092, all changes 0.1 %.
- 5 steps with eta 1 and the tuned decoders lower the 16k test mean by 18 % against 2 steps with frozen decoders
  (0.100 vs 0.122, 0.096 vs 0.117); this mixes sampler and decoders.
- The VAE share inside the DiT evaluation (0.068) is higher than the round trip of `2dec_clip33` itself (0.043).
- The same gap appears with the frozen decoders (0.092 vs 0.075), so it likely comes from encoding every second frame,
  while the VAE was built and tuned for consecutive frames.
- The VAE share is close to the DiT share at 5 steps (0.068 vs 0.069 for `eulermq16k`).
- v dominates the error: 0.206 against at most 0.083 for the other fields (`eulermq16k`, 5 steps).
- The error grows over the rollout (0.043, 0.079, 0.117 over the three windows), with Mach number (0.075 to 0.122) and
  with bubble count (0.070 to 0.129).
- Best run of the section: `res256/pairs/recipe/eulermq16k/unrolled/bench_steps5_eta1` (test mean 0.096, bench VRMSE
  0.092), within 0.1 % of `eulermq16k/bench_steps5_eta1`.
