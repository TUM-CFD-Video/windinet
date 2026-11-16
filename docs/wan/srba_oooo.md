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
`bench_steps5_eta1_ens16` repeats `bench_steps5_eta1` with 16 rollouts per sim (job 23791201); its row is the
16-rollout mean (block `final`, keys `ens_*`), its rollout 0 equals `bench_steps5_eta1`, and it has no cfd_bench
score of the mean.
Per-field VRMSE is The Well's metric on frames 10 to 96; bench is cfd_bench (Harish) on frames 10, 20, ..., 90.
VAE share: the decoded simulation latents against the simulation; DiT share: the rollout against the decoded latents.

| run | decoders | steps | VRMSE mean ± std | median | p | u | v | ln ρ | bench VRMSE | bench RMSE | bench H1 | VAE share | DiT share |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `res256/pairs/recipe/wan16k` | `adapter_q999` | 2 | 0.122 ± 0.030 | 0.111 | 0.067 | 0.059 | 0.259 | 0.103 | 0.118 | 0.025 | 0.640 | 0.092 | 0.086 |
| `res256/pairs/recipe/eulermq16k` | `adapter_q999` | 2 | 0.117 ± 0.028 | 0.110 | 0.065 | 0.057 | 0.247 | 0.101 | 0.114 | 0.023 | 0.636 | 0.092 | 0.080 |
| `res256/pairs/recipe/wan16k/bench_steps5_eta1` | `2dec_clip33` | 5, eta 1 | 0.100 ± 0.035 | 0.094 | 0.053 | 0.046 | 0.217 | 0.085 | 0.096 | 0.021 | 0.516 | 0.068 | 0.075 |
| `res256/pairs/recipe/eulermq16k/bench_steps5_eta1` | `2dec_clip33` | 5, eta 1 | 0.096 ± 0.031 | 0.089 | 0.051 | 0.045 | 0.206 | 0.083 | 0.092 | 0.020 | 0.511 | 0.068 | 0.069 |
| `res256/pairs/recipe/eulermq16k/bench_steps5_eta1_ens16` | `2dec_clip33` | 5, eta 1, mean of 16 | 0.095 ± 0.032 | 0.088 | 0.050 | 0.044 | 0.203 | 0.082 | – | – | – | 0.068 | – |
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

## LoLA baseline

LoLA (Rozet et al. 2025, "Lost in Latent Space", arXiv 2507.02608) fine-tuned on srba_oooo from its published Euler
c80 weights, scored with our metric code.
Code, design notes and runs live on scratch, outside the repo: `$LOLA = /scratch/project_465003416/lc_work/lola/srba`
(`DESIGN.md`, `JOBS.txt`); LoLA run paths below are relative to `$LOLA`.

Setup (paper values from `DESIGN.md`, which cites paper App. B and the published configs):

| item | paper / published (Euler) | ours on srba_oooo |
|---|---|---|
| start | autoencoder `n6g8kix8_euler_all_dcae_f32c64_large` (c80), latent diffusion `e7xzovde_euler_all_f32c64_vit_large` | same weights |
| grid / latent | 512² → 16²×64 | 256² → 8²×64 |
| fields | energy, density, pressure, momentum x (H), momentum y (W) | ln ρ, ln p, v (H), u (W) in those slots, z-scored, u and v on one shared scale; energy output dropped, energy input folded into pressure |
| label | γ, 4 boundary flags | Mach mapped onto LoLA's γ range, 4 open-boundary flags, (bubbles − 3)/2, flow direction (H, W); new label columns start at zero |
| autoencoder training | PSGD lr 1e-5, batch 64, 262k updates, VMSE (paper) / MAE (checkpoint config), flips, permute, roll | PSGD lr 5e-6, 200 warm-up, cosine, batch 64, 30000 updates, MAE, all 8 orientations |
| diffusion training | Adam lr 1e-4 (paper; the published checkpoint used PSGD), batch 256, 262k updates, windows of 5 states at stride 4, published context mask | Adam, batch 256, bf16 backbone, same windows and mask; 125k updates over three warm-started runs (below) |
| window starts | any | start 0 and starts at or after the shock onset (deviation a) |
| diffusion orientations | flips, permute, roll | k 0 and 2 only, the two that keep the flow direction (deviation b); roll dropped (open boundaries) |
| rollout | context 1, overlap 1, Adams–Bashforth 16 steps | frame 0 only, 26 states (frames 0, 4, ..., 100); overlap 2, 16 steps, chosen on validation |
| checkpoint | last (autoencoder saves only the last; diffusion also keeps the lowest validation loss, evaluation loads the last by default) | lowest validation rollout VRMSE (30 validation sims) |

Both deviations favour LoLA.
(a) The frames before the shock arrives are identical (latent distance to frame 0 is 0.000 up to frame 7, 5 and 4 at
Mach 1.3, 1.7, 2.1), so a window starting there pairs frame 0's context with a future shifted by 1 to 7 frames.
The model of `runs/dm_c80` learned a random phase: on validation its rollout from frame 0 gives 0.446, from the true
frame 8 0.103 (`runs/diag/phase/metrics.json`, checkpoint at update 74000, 4 samples).
(b) The model ignores the direction label and reads the direction from the bubble layout: the label's direction
columns have norm 0.35–0.41 against 4–4.8 for the pretrained columns, and changing the direction label leaves the
output unchanged.
At update 19000 of `runs/dm_c80_fix`, single-bubble validation sims 003 and 037 went the wrong way in 8 of 8 samples
(VRMSE ~4) and 043 in 2 of 8; 18 of 240 samples had VRMSE > 1 (`runs/diag/outliers/dm_c80_fix_last_19000.json`).
LoLA's own Rayleigh–Bénard config also drops permute (flip and roll only).

Autoencoder optimiser A/B (`runs/ab`, 40 min per arm on 2 GPUs, validation VRMSE over 30 sims, frames 12 to 100) and
the run that followed:

| run | optimiser | updates | val VRMSE |
|---|---|---|---|
| `runs/ae_c80` at update 0 | none (zero-shot, published weights) | 0 | 0.119 |
| `runs/ab/A` | PSGD lr 5e-6 | 321 | 0.052 |
| `runs/ab/B` | Adam lr 1e-5 | 1253 | 0.079 |
| `runs/ab/C` | Adam lr 1e-5, transverse velocity weighted σu/σv = 7.8 | 1244 | 0.084 |
| `runs/ab/D` | Adam lr 1e-5, bf16 | 2139 | 0.075 |
| `runs/ae_c80` | PSGD lr 5e-6 (arm A), job 23766039 | 30000 | 0.031 |

Diffusion runs, each warm-started from the one before (weights only); validation is one rollout per sim, 16 steps,
overlap 1, over the 30 validation sims (`<run>/history.json`):

| run | start | change | lr | updates | val VRMSE mean | outcome |
|---|---|---|---|---|---|---|
| `runs/dm_c80` | published diffusion model | – | 1e-4 | 77500 of 115000 | 0.35–1.43 after update 0 | cancelled: phase defect (a) |
| `runs/dm_c80_fix` | `dm_c80` last | pre-shock starts dropped | 1e-4 | 22500 of 50000 | 0.12–0.13 at 10000, 20000, 22500; 0.26–1.22 at the other checkpoints | cancelled: direction defect (b) |
| `runs/dm_c80_dir` | `dm_c80_fix` best (22500) | orientations k 0, 2 only | 5e-5 | 25000 | 0.118 → 0.086 | used |

Which validation sims went the wrong way in `dm_c80_fix` changed between checkpoints, so its validation choice was a
lottery.

Sampler choice on the 30 validation sims, `runs/dm_c80_dir` best (update 25000), 16 samples per sim:

| run | overlap | steps | member mean VRMSE | sample 0 | mean of 16 |
|---|---|---|---|---|---|
| `runs/eval_c80/val_overlap1_steps16` | 1 | 16 | 0.091 | 0.088 | 0.074 |
| `runs/eval_c80/val_overlap1_steps32` | 1 | 32 | 0.092 | 0.089 | 0.074 |
| `runs/eval_c80/val_overlap2_steps16` | 2 | 16 | 0.088 | 0.085 | 0.072 |
| `runs/eval_c80/val_overlap2_steps32` | 2 | 32 | 0.089 | 0.087 | 0.072 |

Choice by member mean: overlap 2, 16 steps.

Test, 60 sims (`<run>/metrics.json`, block `final`).
Ours: one rollout predicts frames 2 to 96 at stride 2, scored on frames 10, 12, ..., 96.
LoLA: 26 states at stride 4, scored on frames 12, 16, ..., 100; "matched" is frames 12, 16, ..., 96.
Each VRMSE divides by the variance of its own scored span.
Member mean: each of the 16 samples scored alone, then averaged (the paper's convention); sample 0 is one draw like ours.
VAE / AE: the decoded encoding of the simulation against the simulation; rollout vs AE: the rollout against that
decoding.

| run | statistic | frames | VRMSE mean ± std | median | p | u | v | ln ρ | spread-skill |
|---|---|---|---|---|---|---|---|---|---|
| `res256/pairs/recipe/eulermq16k/bench_steps5_eta1` | one rollout | 10–96, stride 2 | 0.096 ± 0.031 | 0.089 | 0.051 | 0.045 | 0.206 | 0.083 | – |
| `res256/pairs/recipe/eulermq16k/bench_steps5_eta1_ens16` | mean of 16 | 10–96, stride 2 | 0.095 ± 0.032 | 0.088 | 0.050 | 0.044 | 0.203 | 0.082 | 0.15 |
| `res256/pairs/recipe/eulermq16k/bench_steps5_eta1` | VAE share | 10–96, stride 2 | 0.068 ± 0.013 | 0.064 | 0.040 | 0.036 | 0.145 | 0.051 | – |
| `res256/pairs/recipe/eulermq16k/bench_steps5_eta1` | rollout vs VAE | 10–96, stride 2 | 0.069 ± 0.033 | 0.064 | 0.034 | 0.027 | 0.150 | 0.066 | – |
| `runs/eval_c80/test` | member mean | 12–100, stride 4 | 0.087 ± 0.032 | 0.081 | 0.048 | 0.044 | 0.164 | 0.094 | – |
| `runs/eval_c80/test` | member mean, matched | 12–96, stride 4 | 0.084 ± 0.031 | 0.078 | 0.047 | 0.043 | 0.159 | 0.088 | – |
| `runs/eval_c80/test` | sample 0 | 12–100, stride 4 | 0.086 ± 0.035 | 0.077 | 0.047 | 0.043 | 0.163 | 0.093 | – |
| `runs/eval_c80/test` | mean of 16 | 12–100, stride 4 | 0.071 ± 0.033 | 0.064 | 0.034 | 0.032 | 0.139 | 0.079 | 0.81 |
| `runs/eval_c80/test` | AE share | 12–100, stride 4 | 0.031 ± 0.013 | 0.027 | 0.011 | 0.011 | 0.062 | 0.039 | – |
| `runs/eval_c80/test` | rollout (sample 0) vs AE | 12–100, stride 4 | 0.080 ± 0.034 | 0.072 | 0.045 | 0.041 | 0.149 | 0.086 | – |

Our VRMSE on LoLA's matched frames (12, 16, ..., 96) is not available: `metrics.json` keeps per-frame means, not the
rollouts, and a per-sim VRMSE over another span needs the rollouts again (a GPU evaluation).

cfd_bench VRMSE; the shared frames 20, 40, 60, 80 are the same for both:

| run | statistic | own frames | VRMSE, own frames | VRMSE, frames 20/40/60/80 |
|---|---|---|---|---|
| `res256/pairs/recipe/eulermq16k/bench_steps5_eta1` | one rollout | 10, 20, ..., 90 | 0.092 | 0.092 |
| `res256/pairs/recipe/eulermq16k/bench_steps5_eta1` | VAE share | 10, 20, ..., 90 | 0.067 | – |
| `runs/eval_c80/test` | member mean | 20, 40, ..., 100 | 0.092 | 0.076 |
| `runs/eval_c80/test` | sample 0 | 20, 40, ..., 100 | 0.091 | 0.075 |
| `runs/eval_c80/test` | mean of 16 | 20, 40, ..., 100 | 0.075 | 0.061 |
| `runs/eval_c80/test` | AE share | 20, 40, ..., 100 | 0.032 | – |

Our 16-rollout mean has no cfd_bench score (`bench_steps5_eta1_ens16` scores rollout 0 only).

VRMSE per Mach number, per bubble count and per The Well window (ours: frames 10–26, 28–56, 58–96; LoLA: 12–24,
28–56, 60–100); neither side stores the windows of the member mean or the scalars of the 16-sample mean:

| run | statistic | Mach 1.3 | 1.7 | 2.1 | 1 bubble | 2 | 3 | 4 | 5 | frames 9–26 | 27–56 | 57–100 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `res256/pairs/recipe/eulermq16k/bench_steps5_eta1` | one rollout | 0.075 | 0.091 | 0.122 | 0.070 | 0.080 | 0.090 | 0.111 | 0.129 | 0.043 | 0.079 | 0.117 |
| `res256/pairs/recipe/eulermq16k/bench_steps5_eta1_ens16` | mean of 16 | – | – | – | – | – | – | – | – | 0.041 | 0.078 | 0.116 |
| `res256/pairs/recipe/eulermq16k/bench_steps5_eta1` | VAE share | – | – | – | – | – | – | – | – | 0.039 | 0.066 | 0.075 |
| `runs/eval_c80/test` | member mean | 0.073 | 0.082 | 0.108 | 0.058 | 0.070 | 0.084 | 0.105 | 0.121 | – | – | – |
| `runs/eval_c80/test` | sample 0 | 0.071 | 0.079 | 0.108 | 0.057 | 0.068 | 0.089 | 0.099 | 0.117 | 0.021 | 0.053 | 0.111 |
| `runs/eval_c80/test` | mean of 16 | – | – | – | – | – | – | – | – | 0.012 | 0.039 | 0.093 |
| `runs/eval_c80/test` | AE share | – | – | – | – | – | – | – | – | 0.005 | 0.019 | 0.040 |

Cost on LUMI (one node = 8 GPUs, `sacct` elapsed time); neither side counts the training of its starting weights (LoLA
on Euler; our eulermq decoders and DiT):

| run | jobs | node h | GPU h |
|---|---|---|---|
| `runs/ae_c80`, `dm_c80`, `dm_c80_fix`, `dm_c80_dir`, `eval_c80` | 23766039, 23767303, 23786358, 23791505, 23791506 | 20.9 | 167 |
| `res256/pairs/adapter_q999` (4-arm job), `2dec_from_eulermq` (2-arm job), `2dec_clip33`, latents, `recipe/eulermq16k`, `bench_steps5_eta1` (2-run job) | 23707708, 23709385, 23711640, 23709387, 23709995, 23738942 | 29.6 | 237 |

LoLA's smoke, A/B, pilot and diagnosis jobs are not counted.
Our shared jobs also trained or scored other arms, so our total is an upper bound.
The 60-sim test evaluation with 16 samples took 216 s for LoLA and 1369 s for ours (`provenance.seconds`).

Reading:

- LoLA wins the rollout: member mean 0.087 against our single rollout 0.096 (9 % lower), sample 0 0.086.
- It is clearly lower on v (0.164 vs 0.206), about equal on p and u (within 0.004), lower in every Mach and bubble
  cell, and higher only on ln ρ (0.094 vs 0.083).
- On the shared cfd_bench frames it gives 0.076 against our 0.092.
- Its autoencoder floor is 0.031 against our VAE share of 0.068.
- Its rollout against its own decoding is 0.080 against our 0.069, on different frame spans.
- Its lead therefore likely comes from the autoencoder, not the latent model.
- Its 16-sample mean lowers its sample 0 by 17 % (0.086 → 0.071) at spread-skill 0.81; ours lowers our single
  rollout by 1.4 % (0.096 → 0.095) at spread-skill 0.15.
- Our rollouts are therefore under-dispersed.
- LoLA cost 20.9 node h against our 29.6.
- Caveat: LoLA's sampler was chosen on validation among 4 settings and its checkpoints by validation VRMSE; ours uses
  the last weights at 5 steps, eta 1, chosen on eulermq (`docs/wan/dit_sampling.md`), not on srba validation.
- Caveat: the scored frames differ (stride 2 to frame 96 vs stride 4 to frame 100).
- Caveat: LoLA's matched-frame number (0.084) has no counterpart on our side.
- Caveat: LoLA was fine-tuned with fewer updates than the paper's from-scratch schedule: autoencoder 30000 against
  262k, diffusion 125k in total across the three runs against 262k.
