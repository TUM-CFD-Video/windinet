# Field → RGB adapter for the frozen Wan 2.1 VAE

**Question.** Can a tiny deterministic per-pixel map turn the four Euler fields
(ρ, m_x, m_y, p) into RGB so that the *frozen* Wan 2.1 video VAE carries them
without loss, and can that map also make the latents look like natural video?

**Setup.** Adapter = fixed log/z-score → per-field monotone spline → linear mix →
tanh, exact inverse, 14 to 160 parameters ([bijections.py](../../windinet/field_adapter/bijections.py)).
Round trip = fields → adapter → Wan encode → Wan decode → adapter⁻¹ → fields.
Data: euler_mq 256×256, 500 sims (50 per γ), every 5th sim per γ held out.
Metric: VRMSE per field in physical units on held-out frames (0.05 = 5 % of the
field's own spread, 1.0 = as bad as predicting the mean). Training: RMSE + H1 in
z-scored field space, Adam 1e-3, 400 steps × 2 frames, gradients through the
frozen VAE, ~20 min on a 2080 Ti. Latent reference: 319 Pexels clips encoded
with the same VAE; Fréchet distance between 16-channel latent Gaussians.
The inverse is clamped to the dataset's value range (without it a handful of
tanh-tail pixels blow up through exp and dominate the number).
Every run: `results/field_adapter/20_wan_roundtrip/<name>.{json,png,pt}`.

## Experiments

### 1. Is one 4 → 3 map enough?

*Idea:* if the four fields are redundant enough, three colours could carry them.

*Result:* no. Pressure is lost (VRMSE ≈ 1.5 after training), and the residual
image shows density structure leaking into pressure. Training does not help.

| single 4 → 3 | ρ | m_x | m_y | p |
|---|---|---|---|---|
| naive (ρ, m_x, m_y → RGB, p dropped) | 0.20 | 0.17 | 0.20 | 2.36 |
| PCA init, trained | 0.82 | 0.22 | 0.14 | 1.53 |

*Why:* a per-pixel 3×4 matrix has a null space; whatever combination of fields
lands there is gone. Spatial context could recover it, a per-pixel map cannot.

### 2. Do we need an adapter at all, or is scaling + frozen VAE enough?

*Idea:* z-score each field, put it straight into a colour channel, no learning.

*Result:* the frozen VAE already carries the fields to 7 to 29 % error. Learning
the ~100 adapter parameters halves that on average.

| two groups, 3 colours each | ρ | m_x | m_y | p | mean |
|---|---|---|---|---|---|
| naive pairs (ρ,p) (m_x,m_y), untrained | 0.17 | 0.20 | 0.14 | 0.07 | 0.145 |
| pairs, trained | 0.08 | 0.17 | 0.12 | 0.06 | 0.107 |
| naive triplets (m_x,m_y,p) (ρ,p,E), untrained | 0.29 | 0.22 | 0.18 | 0.11 | 0.200 |
| triplets, trained (naive init) | 0.06 | 0.17 | 0.13 | 0.06 | 0.104 |

*Why:* the VAE's error per colour channel is roughly fixed; the adapter's job is
to spend the colour range where the fields need it (log for ρ and p, a learned
scale, a mix that shares edges across channels). Momentum stays at 0.12 to 0.17
in every row: that is the VAE blurring thin shock lines, and no per-pixel map
changes it.

### 3. Which grouping?

*Idea:* (ρ,p,E)+(m_x,m_y,p) shares the pressure edges across both images (E is
derived from the other fields with γ); (ρ,p)+(m_x,m_y) is simpler, 2 → 3 each.

*Result:* equal within noise after training (mean 0.104 vs 0.107). Pairs need no
derived field and no γ, so pairs is the default.

### 4. What do the pieces buy? (triplets)

| | ρ | m_x | m_y | p | mean |
|---|---|---|---|---|---|
| naive, untrained | 0.29 | 0.22 | 0.18 | 0.11 | 0.200 |
| PCA/luma init, untrained | 0.18 | 0.12 | 0.17 | 0.24 | 0.178 |
| PCA init, mix only, trained | 0.07 | 0.09 | 0.15 | 0.20 | 0.127 |
| PCA init, mix + splines, trained | 0.06 | 0.08 | 0.17 | 0.16 | 0.116 |
| naive init, mix + splines, trained | 0.06 | 0.17 | 0.13 | 0.06 | 0.104 |

Checked again on pairs at lr 3e-3, 600 steps: with splines mean 0.079
(0.05/0.12/0.10/0.05), without splines mean 0.074 (0.04/0.11/0.08/0.06), 14 parameters.

*Why:* the learned mix does all the work; the splines add nothing once training is
strong, so the adapter can be log/z-score → linear mix → tanh scale, nothing else.
The PCA init (rotate to luma/opponent axes and squeeze two axes to natural-image
variance ratios) hurts pressure: pressure lives on the squeezed axes, and the
VAE's fixed noise then dominates it. Starting from the plain z-score init avoids
that and gives the best pressure. Init matters more than the spline.

### 5. Can the adapter make the latents look natural?

*Idea:* add a loss pulling the 16-channel latent mean/variance/covariance toward
the Pexels reference (λ on both terms), with the VAE frozen.

*Noise floor:* Fréchet(half of reference, other half) = 0.07; single frames vs
9-frame clips of the same reference = 0.9; N(0,1) vs reference = 8.3.
Physics latents sit at 5.4 to 6.8 in every run, trained or not.

| pairs, naive init | ρ | m_x | m_y | p | Fréchet A / B |
|---|---|---|---|---|---|
| λ = 0 | 0.08 | 0.17 | 0.12 | 0.06 | 6.43 / 6.83 |
| λ = 1 | 0.15 | 0.17 | 0.12 | 0.06 | 5.76 / 6.78 |
| λ = 10, 200 steps | 0.30 | 0.19 | 0.12 | 0.07 | 5.74 / 6.63 |

*Result:* the latent distance barely moves (6.4 → 5.7 on one group, 6.8 → 6.6 on
the other) while density error doubles at λ = 1 and quadruples at λ = 10.

*Why:* a per-pixel monotone map with ~50 parameters can shift and stretch colour
values, but latent statistics are set by *spatial* structure (shocks are sharp
lines, natural video is textured), which no per-pixel map changes. The gap is
a property of the data, not of the colour assignment. Note also that the Wan
reference latents are not N(0,1) after Wan's own `latents_mean/std` (var 0.23 to
0.48), so "match N(0,1)" would be the wrong target anyway.

### 6. Frames vs clips

*Idea:* all runs above use single frames (F = 1); Wan compresses time 4×. Same 9
consecutive frames of 20 held-out sims, once as one clip, once frame by frame,
with the adapters trained on frames.

| trained on frames, evaluated on | ρ | m_x | m_y | p |
|---|---|---|---|---|
| pairs, frame by frame | 0.09 | 0.15 | 0.13 | 0.07 |
| pairs, 9-frame clip | 0.15 | 0.16 | 0.15 | 0.09 |
| triplets, frame by frame | 0.06 | 0.06 | 0.13 | 0.15 |
| triplets, 9-frame clip | 0.09 | 0.08 | 0.16 | 0.18 |
| best pairs (1200 steps), frame by frame | 0.04 | 0.08 | 0.07 | 0.06 |
| best pairs (1200 steps), 9-frame clip | 0.06 | 0.11 | 0.09 | 0.09 |

*Result:* the adapter transfers to clips unchanged, at a cost of ~0.03 VRMSE per
field from the temporal compression. Density suffers most (moving shocks are
thin in space *and* time).

### 7. How far does more training go?

*Idea:* the 400-step runs were still improving; raise the learning rate and train longer.

| pairs, naive init | ρ | m_x | m_y | p | mean |
|---|---|---|---|---|---|
| lr 1e-3, 400 steps | 0.08 | 0.17 | 0.12 | 0.06 | 0.107 |
| lr 3e-3, 600 steps | 0.05 | 0.12 | 0.10 | 0.05 | 0.079 |
| lr 3e-3, 1200 steps | 0.03 | 0.10 | 0.07 | 0.05 | 0.064 |

*Result:* still improving at 1200 steps; the best adapter so far reaches 3 to 10 %
per field with ~100 parameters. Momentum improves too, so the "VAE limit" on
momentum in experiment 2 is partly an under-training effect.

## Findings

- A per-pixel 4 → 3 map is a negative result: pressure cannot be recovered.
- Two colour images per frame work: the frozen Wan VAE alone (plain z-score,
  nothing learned) reaches 7 to 29 % per-field error; the learned adapter brings it
  to 3 to 10 % and is not converged yet. Pairs (ρ,p)+(m_x,m_y) is as good as the physics-motivated triplets
  and simpler.
- Momentum stays the hardest field (thin shock lines); longer training brings it from 0.17 to 0.10 to 0.12.
- Init matters: PCA/luma init costs pressure; plain z-score init is better.
- The splines are not needed: a linear mix + one scale (14 parameters) matches or beats the spline version.
- The adapter cannot make physics latents look like natural-video latents;
  the latent loss trades reconstruction for almost nothing. If latent-space
  naturalness matters downstream, it has to come from the VAE or the DiT side,
  not from a per-pixel map.

## Images

Pairs, trained, best overall: [wan_pairs_lr3e3_1200.png](../../results/field_adapter/20_wan_roundtrip/wan_pairs_lr3e3_1200.png).
Single 4 → 3 (negative result, density leaks into pressure): [wan_single_clamp.png](../../results/field_adapter/20_wan_roundtrip/wan_single_clamp.png).
Triplets, trained: [wan_triplets_clamp.png](../../results/field_adapter/20_wan_roundtrip/wan_triplets_clamp.png).
Each panel: per field GT / reconstruction / residual, last column the RGB image(s) fed to the VAE.

## Raw run log (auto-appended by `scripts/field_adapter/train_adapter.py`)

| stage | name | vae | adapter | loss weights | VRMSE density | VRMSE momentum_x | VRMSE momentum_y | VRMSE pressure | notes |
|---|---|---|---|---|---|---|---|---|---|
| 20_wan_roundtrip | wan_pairs_lr3e3_600 | wan | pairs, naive, warp=True | rmse=1, h1=1, slope_penalty=0.001 | 0.0473 | 0.1187 | 0.0964 | 0.0520 | latentA_frechet=6.96, latentB_frechet=7.26 |
| 20_wan_roundtrip | wan_pairs_lr3e3_1200 | wan | pairs, naive, warp=True | rmse=1, h1=1, slope_penalty=0.001 | 0.0342 | 0.1031 | 0.0678 | 0.0497 | latentA_frechet=6.65, latentB_frechet=7.26 |
| 20_wan_roundtrip | wan_pairs_nowarp_lr3e3_600 | wan | pairs, naive, warp=False | rmse=1, h1=1, slope_penalty=0.001 | 0.0436 | 0.1128 | 0.0815 | 0.0568 | latentA_frechet=6.38, latentB_frechet=7.44 |
