# Per-pixel field-to-RGB adapter for the frozen Wan 2.1 VAE

Question: can a small deterministic map from the four Euler fields (ρ, m_x, m_y, p)
to RGB let the *frozen* Wan 2.1 VAE carry CFD fields, and make the latents look
like natural video? Answer: yes at 3 to 10 % error; no for the latents.

## Setup

- Adapter: fixed log/z-score per field → learned linear mix to 3 colours → learned tanh scale. Exact inverse.
- Round trip: fields → adapter → Wan encode → decode → adapter⁻¹ → fields, single 256×256 frames.
- Data: euler_mq test shards, 500 simulations (50 per γ); every 5th per γ held out.
- Metric: VRMSE per field in physical units on held-out frames. 0.05 = 5 % of the field's spread, 1.0 = predicting the mean.
- Training: RMSE + H1 in z-scored field space, Adam, gradients through the frozen VAE, 2 frames per step. 400 steps ≈ 20 min on a 2080 Ti.
- Latent reference: 319 Pexels clips (9 frames, 256×256) through the same VAE; Fréchet distance between 16-channel Gaussians.
- Inverse clamped to the dataset's value range; otherwise a few tanh-tail pixels explode through exp.
- Ablated variants (splines, PCA init, latent losses) are removed from the code; commit `0a525c9` has them and is the only place their checkpoints load. Reference run of the current code: `wan_pairs_nowarp_lr3e3_600`.
- Per run: `results/field_adapter/20_wan_roundtrip/<name>/` with config.yaml (the resolved config, re-runnable as is), metrics.json (outcome and provenance), panel.png, curves.png and adapter.pt; fine-tunes also vae.pt, which is not in git. The `run` column of each table names the folder; the [index](../../results/field_adapter/20_wan_roundtrip/README.md) lists every run with a one-line description.

## 1. One image is not enough

A single 4 → 3 map loses pressure; density leaks into it. A per-pixel 3×4
matrix has a null space, and training cannot fill it.

| single 4 → 3 | ρ | m_x | m_y | p | run |
|---|---|---|---|---|---|
| ρ, m_x, m_y → RGB, p dropped, untrained | 0.20 | 0.17 | 0.20 | 2.36 | `wan_single_naive` |
| PCA init, trained | 0.82 | 0.22 | 0.14 | 1.53 | `wan_single_clamp` |

## 2. Two images work; the learned mix halves the error

Two colour images per frame, (ρ, p) and (m_x, m_y). Plain z-score into channels
already gives 7 to 20 %; the learned mix and scale give 3 to 10 %. The 2000-step run
(current code, seeded sampler) gains 0.002 over its last 500 steps: converged. Momentum
stays the hardest field.

| pairs (ρ,p) + (m_x,m_y) | ρ | m_x | m_y | p | mean | run |
|---|---|---|---|---|---|---|
| z-score only, untrained | 0.17 | 0.20 | 0.14 | 0.07 | 0.145 | `wan_pairs_naive` |
| lr 1e-3, 400 steps | 0.08 | 0.17 | 0.12 | 0.06 | 0.107 | `wan_pairs` |
| lr 3e-3, 600 steps | 0.05 | 0.12 | 0.10 | 0.05 | 0.079 | `wan_pairs_lr3e3_600` |
| lr 3e-3, 1200 steps | 0.03 | 0.10 | 0.07 | 0.05 | 0.064 | `wan_pairs_lr3e3_1200` |
| lr 3e-3, 2000 steps | 0.04 | 0.09 | 0.08 | 0.06 | 0.068 | `wan_pairs_2000` |

## 3. Grouping and initialisation

Triplets (m_x, m_y, p) + (ρ, p, E), with E derived via γ and p shared across
both images, end equal to pairs after training (0.104 vs 0.107). Pairs need no
derived field, so pairs are the default.

PCA init (principal axes rescaled to natural-image colour statistics) costs
pressure: it lands on the squeezed axes, where the VAE's constant noise
dominates. Plain z-score init avoids this.

| triplets | ρ | m_x | m_y | p | mean | run |
|---|---|---|---|---|---|---|
| z-score, untrained | 0.29 | 0.22 | 0.18 | 0.11 | 0.200 | `wan_triplets_naive` |
| PCA init, untrained | 0.18 | 0.12 | 0.17 | 0.24 | 0.178 | `wan_triplets_init_clamp` |
| PCA init, trained | 0.06 | 0.08 | 0.17 | 0.16 | 0.116 | `wan_triplets_clamp` |
| z-score init, trained | 0.06 | 0.17 | 0.13 | 0.06 | 0.104 | `wan_triplets_naiveinit` |

3+1, (ρ, m_x, m_y) + (p): pressure alone in a grey image is the best pressure result,
but density pays for sharing an image with both momenta. Same code and budget as
`wan_pairs_nowarp_lr3e3_600` (mean 0.074).

| 3+1 | ρ | m_x | m_y | p | mean | run |
|---|---|---|---|---|---|---|
| untrained | 0.20 | 0.17 | 0.20 | 0.12 | 0.172 | `wan_3plus1_init` |
| lr 3e-3, 600 steps | 0.13 | 0.10 | 0.14 | 0.04 | 0.102 | `wan_3plus1` |

## 4. Splines add nothing

Pairs, lr 3e-3, 600 steps: with a monotone spline per field mean 0.079
(~100 parameters), without 0.074 (14 parameters). Removed.

## 5. The adapter cannot make latents look natural

Physics latents: Fréchet 5.4 to 6.8 from the reference in every run. Scale: two
halves of the reference 0.07, frames vs 9-frame clips 0.9, standard normal 8.3.
A loss pulling latent mean and covariance toward the reference barely moves the
distance and ruins density. Per-pixel maps change colour values, not spatial
structure; latent statistics depend on the latter. Wan's `latents_mean/std` also
do not standardise real clips (per-channel variance 0.23 to 0.48).

| pairs, latent loss λ | ρ | m_x | m_y | p | Fréchet A / B | run |
|---|---|---|---|---|---|---|
| 0 | 0.08 | 0.17 | 0.12 | 0.06 | 6.43 / 6.83 | `wan_pairs` |
| 1 | 0.15 | 0.17 | 0.12 | 0.06 | 5.76 / 6.78 | `wan_pairs_latent1` |
| 10, 200 steps | 0.30 | 0.19 | 0.12 | 0.07 | 5.74 / 6.63 | `wan_pairs_latent10` |

## 6. Frames vs clips

Everything above passes single frames. The DiT will see clips, and Wan compresses
4× in time. Evaluated on the same held-out sims with 5 frames each, spread over the
trajectory (single frames) or consecutive and encoded as one clip. Adapter
`wan_pairs_h1_50` fixed, H1 weight 1, `--set data.clip=true`.

| Wan VAE, evaluated on | ρ | m_x | m_y | p | mean | Fréchet A / B | run |
|---|---|---|---|---|---|---|---|
| frozen, single frames | 0.042 | 0.211 | 0.145 | 0.051 | 0.112 | 6.65 / 6.66 | `clip5_frozen_frames` |
| frozen, 5-frame clips | 0.075 | 0.183 | 0.151 | 0.090 | 0.125 | 9.23 / 9.59 | `clip5_frozen` |
| frozen, 9-frame clips | 0.075 | 0.186 | 0.153 | 0.092 | 0.127 | 11.3 / 11.9 | `clip9_frozen` |
| decoder tuned on frames (section 7), single frames | 0.040 | 0.124 | 0.109 | 0.046 | 0.080 | 6.65 / 6.66 | `clip5_dec_frames` |
| decoder tuned on frames, 5-frame clips | 0.065 | 0.132 | 0.133 | 0.082 | 0.103 | 9.23 / 9.59 | `clip5_dec` |
| decoder tuned on 5-frame clips, 5-frame clips | 0.060 | 0.122 | 0.120 | 0.073 | 0.094 | 9.23 / 9.59 | `wan_ft_decoder_clip5` |
| decoder tuned on 5-frame clips, single frames | 0.041 | 0.132 | 0.109 | 0.047 | 0.082 | 6.65 / 6.66 | `clip5_decclip_frames` |

Temporal compression costs 0.03 to 0.04 on density and pressure, the fields with
sharp shocks, and nothing on momentum; clip length beyond 5 does not matter. The
frame-tuned decoder keeps most of its gain on clips. Tuning the decoder on clips instead
(section 7 recipe, 3 clips × 5 frames per update, 32 min) is better on clips and as good
on single frames, so clips are the right training unit for the decoder. Behind the
converged adapter the same recipe reaches 0.063 on clips (section 7). The Fréchet
distance of clip latents is larger for every model: the reference set is 9-frame clips,
so this is the comparable number, and the adapter's latents are further from natural
video in time than in space.

Clip fine-tuning needs block-wise activation checkpointing inside the Wan decoder
(`windinet/wan/vae.py`): with it a 5-frame clip peaks at 7.8 GB, without it 9 frames do
not fit, and the gradient is unchanged (relative difference 1e-8).

## 7. Fine-tuning the Wan VAE behind the fixed adapter

Adapter fixed (`wan_pairs_h1_50`, trained with the same loss). Recipe copied from
chapter 6 (`configs/finetune_vae/finetune_vae_ch6_*_256res.yaml`): AdamW, peak 5e-5,
1 % linear warm-up, cosine to 1e-6, grad clip 5, RMSE + 50 H1, 16 frames per update.
Differences: fp32 (no bf16 on the 2080 Ti), activation checkpointing by hand, no SSIM,
single frames, 120 updates on the 400 test-shard sims instead of 20 epochs on train.h5.
Config: each run's `config.yaml` (re-runnable with `scripts/wan/train_vae.py`).

The LTX peak lr breaks the Wan decoder within one update (VRMSE 1 to 25 after 60).
Runs without warm-up and clipping at 1e-5 and 1e-6 degraded as well and were discarded.
The gradient is correct: finite differences along it match autograd to 0.2 %, with and
without checkpointing, and lr 0 reproduces the frozen numbers exactly. The loss is
sharp: along the negative gradient of 8 held-out frames it is lowest after a step of
about 1e-3 in weight space (73 M decoder parameters) and worse than the start beyond
3e-3. An Adam update of lr per weight moves about 8500·lr in that norm, so the usable
peak lr is 1e-7 to 1e-6. The RGB-space RMSE has the same trust region, so the
sharpness is the decoder's, not the adapter's tanh inverse. With H1 weight 50 no step
of 3e-4 or more lowers the loss on the frames the gradient came from.

| loss, step along −grad | 0 | 3e-4 | 1e-3 | 3e-3 | 1e-2 |
|---|---|---|---|---|---|
| RMSE + 50 H1 (fields) | 4.19 | 4.28 | 4.68 | 6.45 | 8.12 |
| RMSE (fields) | 0.181 | 0.174 | 0.169 | 0.188 | 0.627 |
| RMSE (RGB) | 0.0091 | 0.0088 | 0.0087 | 0.0130 | 0.0384 |

Peak-lr sweep, decoder only, 15 updates of 8 frames, same recipe otherwise:

| peak lr | ρ | m_x | m_y | p | run |
|---|---|---|---|---|---|
| frozen | 0.042 | 0.189 | 0.134 | 0.057 | `wan_pairs_h1_50` |
| 1e-8 | 0.042 | 0.188 | 0.133 | 0.057 | `lrsweep_dec_1e-8` |
| 1e-7 | 0.042 | 0.188 | 0.133 | 0.056 | `lrsweep_dec_1e-7` |
| 1e-6 | 0.040 | 0.163 | 0.138 | 0.054 | `lrsweep_dec_1e-6` |
| 3e-6 | 0.066 | 0.225 | 0.143 | 0.092 | `lrsweep_dec_3e-6` |
| 1e-7, H1 weight 1 | 0.041 | 0.179 | 0.127 | 0.054 | `lrsweep_dec_1e-7_h1_1` |

The window is narrow: 1e-6 helps, 3e-6 already hurts, and 5e-5 is 50× past the edge.
H1 weight 1 beats 50 at equal lr, as it did for the adapter (section 2 vs `wan_pairs_h1_50`).
Full runs below use peak 1e-6.

Peak 1e-6, 120 updates of 16 frames, adapter fixed:

| Wan VAE | ρ | m_x | m_y | p | mean | Fréchet A / B | time | run |
|---|---|---|---|---|---|---|---|---|
| frozen | 0.042 | 0.189 | 0.134 | 0.057 | 0.106 | 6.62 / 6.59 | – | `wan_pairs_h1_50` |
| decoder fine-tuned | 0.046 | 0.117 | 0.108 | 0.056 | 0.082 | 6.62 / 6.59 | 43 min | `wan_ft_decoder` |
| all fine-tuned | 0.045 | 0.145 | 0.114 | 0.057 | 0.090 | 6.77 / 6.75 | 64 min | `wan_ft_all` |
| decoder + adapter (lr 3e-3) | 0.048 | 0.135 | 0.113 | 0.062 | 0.089 | 6.79 / 6.39 | 58 min | `wan_ft_decoder_adapter` |
| decoder fine-tuned, H1 weight 1 | 0.040 | 0.114 | 0.102 | 0.051 | 0.077 | 6.62 / 6.59 | 43 min | `wan_ft_decoder_h1_1` |

Decoder fine-tuning takes a third off momentum and leaves the other fields and the
latents untouched. Training the encoder as well helps less at 1.5× the cost and moves
the latents slightly away from the reference. Co-training the adapter with the decoder
is no better than keeping it fixed: the two chase each other. H1 weight 1 instead of 50
improves every field again and brings density back under the frozen value. All stay above
the best adapter-only run (section 2, `wan_pairs_lr3e3_1200`: mean 0.064, 1200 steps),
so at this budget the 14-parameter adapter is the better use of the GPU; decoder
fine-tuning is the next step once the adapter has converged.

[decoder fine-tuned, H1 weight 1](../../results/field_adapter/20_wan_roundtrip/wan_ft_decoder_h1_1/panel.png)

### Is the adapter still needed once the VAE is trained?

Same recipe on 5-frame clips, H1 weight 1, 120 updates of 3 clips, but no learned
adapter: z-score into one colour channel per field, identity mix, tanh scale set once
from data (the "untrained" adapter of section 2, saved and shared by both arms).
The encoder may learn the mix itself in the "all" arm, the LTX chapter-6 design.

| adapter | trained VAE part | ρ | m_x | m_y | p | mean | Fréchet A / B | time | run |
|---|---|---|---|---|---|---|---|---|---|
| untrained | none | 0.248 | 0.187 | 0.148 | 0.107 | 0.172 | 8.73 / 9.91 | – | `wan_ft_decoder_noadapter` (init) |
| untrained | decoder | 0.136 | 0.119 | 0.123 | 0.099 | 0.119 | 8.73 / 9.91 | 40 min | `wan_ft_decoder_noadapter` |
| untrained | all | 0.131 | 0.118 | 0.119 | 0.097 | 0.116 | 8.80 / 10.0 | 53 min | `wan_ft_all_noadapter` |
| learned (`wan_pairs_h1_50`) | none | 0.075 | 0.183 | 0.151 | 0.090 | 0.125 | 9.23 / 9.59 | – | `clip5_frozen` |
| learned (`wan_pairs_h1_50`) | decoder | 0.060 | 0.122 | 0.120 | 0.073 | 0.094 | 9.23 / 9.59 | 32 min | `wan_ft_decoder_clip5` |
| learned, 2000 steps (`wan_pairs_2000`) | none | 0.058 | 0.124 | 0.122 | 0.089 | 0.098 | 8.78 / 12.3 | – | `wan_ft_decoder_pairs2000` (init) |
| learned, 2000 steps | decoder | 0.041 | 0.068 | 0.073 | 0.070 | 0.063 | 8.78 / 12.3 | 40 min | `wan_ft_decoder_pairs2000` |
| learned, 2000 steps | decoder, then adapter + decoder jointly | 0.040 | 0.066 | 0.071 | 0.068 | 0.061 | 8.80 / 12.3 | +48 min | `wan_ft_joint_pairs2000` |

Without the learned mix the trained decoder halves density and pressure errors but
stops at twice the adapter's values; training the encoder too gains 0.003 and moves the
latents. The frozen encoder keeps only what the input images make visible, and a
per-field colour channel with a data-set tanh scale is not enough. The 14 parameters in
front are not redundant at this budget. Caveat on the "all" arm: at peak 1e-6 the
encoder moves 0.03 % in 120 updates, too little to learn a new channel mix, so this is
not the LTX setting (5e-5, 20 epochs). A sweep on frames (30 updates, no adapter, full
VAE) shows the encoder tolerates no more than the decoder: 1e-6 reaches 0.117, 1e-5
0.139, 1e-4 0.338 with the latents pulled away from the reference. The encoder route
needs a budget this GPU does not have, and the table stands with that caveat. Weight drift after 120 updates is 0.03 to 0.04 %
of the pretrained norm, concentrated in the last up-block and `conv_out`
([curves](../../results/field_adapter/20_wan_roundtrip/wan_ft_decoder_noadapter/curves.png)).
The training loss is heavy-tailed (shock frames), so per-update loss curves need a
running mean to read.

The converged adapter (2000 steps, section 2) is worth 0.03 on clips before any
fine-tuning, and the decoder behind it reaches 0.063 on 5-frame clips, the best number
of the study, with the latents untouched. A joint polish afterwards, adapter at 1e-4 and
decoder at 1e-6 (relative step 8e-5 and 8e-6 per update, a 10× ratio instead of the
3000× of the failed co-training in the table above), gains 0.002 and moves the latents
by 0.02: within noise. Recipe: adapter alone until converged, then the decoder on
clips; joint training adds nothing at this budget.

Runs from this section onward log the training loss per update, the relative weight
change per step and group, and the drift from the pretrained weights per block
(`curves.png` next to each run).

## 8. Generalisation across gamma

Same recipe as section 2 (lr 3e-3, 600 steps, 2 frames per step), but one whole gamma
held out instead of every 5th sim: 450 train / 50 test sims, `--set data.test_gamma=`.

| held out | ρ | m_x | m_y | p | mean | run |
|---|---|---|---|---|---|---|
| every 5th sim (reference) | 0.054 | 0.119 | 0.096 | 0.063 | 0.083 | `gamma_ref` |
| γ = 1.365, the middle of the range | 0.049 | 0.124 | 0.108 | 0.062 | 0.086 | folder lost, see below |
| γ = 1.76, the upper end | 0.044 | 0.257 | 0.117 | 0.069 | 0.122 | `gamma_1.76_heldout` |

Interpolating in gamma costs nothing. Extrapolating to the stiffest gas doubles the
m_x error and leaves the other fields intact: the 14 parameters are not overfitting the
regimes, the learned momentum scale simply does not reach the widest shocks. The folder
of the γ = 1.365 run was overwritten when the results were reorganised; its row is the
run log entry at the time (`--set data.test_gamma=1.365` reproduces it).

## Summary

- Frozen Wan VAE + two colour images + 14-parameter linear adapter: 3 to 10 % per field, on par with the fine-tuned LTX VAE in this repo (7 to 10 %), at ~8× the latent bandwidth.
- Negative: one image per frame, splines, PCA init, latent-matching losses.
- Latent naturalness must come from the VAE or the diffusion model, not from a per-pixel map.
- Fine-tuning the Wan decoder behind the fixed adapter works only with a peak lr near 1e-6; the chapter-6 value 5e-5 diverges in one update because the decoder's loss landscape is that sharp. Decoder-only beats full fine-tuning, keeps the latents unchanged, and behind the converged adapter reaches 0.063 on 5-frame clips (frozen 0.098).
- The adapter is not redundant once the VAE is trained: without it, decoder fine-tuning stops at 0.119 and full fine-tuning at 0.116 on clips, against 0.063 with it.
- Adapter first, decoder second. Joint training with matched relative steps afterwards gains 0.002; co-training with the adapter at its own lr is worse.
- H1 weight 1 beats 50 for the adapter and for the decoder alike.
- 4× temporal compression costs 0.03 to 0.04 on density and pressure; a decoder tuned on 5-frame clips recovers most of it (0.094 on clips) and loses nothing on frames.
- The adapter interpolates across gamma at no cost; extrapolation to the stiffest gas doubles the m_x error only.
- One Wan encode of a 256×256 frame takes 60 ms on the 2080 Ti in fp32 (5-frame clip 240 ms, decode about 1.6× that).

Panels (GT / reconstruction / residual per field, RGB images in the last column):
[best adapter](../../results/field_adapter/20_wan_roundtrip/wan_pairs_lr3e3_1200/panel.png),
[failed single map](../../results/field_adapter/20_wan_roundtrip/wan_single_clamp/panel.png).
