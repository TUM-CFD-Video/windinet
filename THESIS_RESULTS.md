# Thesis Results — Final Chapter 6 Numbers

Status: **final**, as of 2026-09-27 (commit 8dc5d58 + table regeneration).
This file is the single source of truth for the thesis Results chapter
(Chapter 6) and its appendices. It supersedes every results claim in
`THESIS_PREP.md` (see §9). It is written as input for drafting the thesis
text: the numbers are final, and the notes say how strongly each finding
may be stated.

Thesis focus: the VAE fine-tuning study is the main contribution; the DiT
results show how each VAE choice carries through to the full pipeline.

The generated LaTeX tables are `thesis/tables/{vae_test,vae_val,dit}.tex`
(from `scripts/thesis_tables.py`). Appendices A–C (setup, hyperparameters,
data, metrics, noise floor, complete tables) are already written in
`thesis/appendix.tex`. Chapter 6 should cite those appendices and not
repeat the full tables.

---

## 1. Setup in brief (details: `thesis/appendix.tex`, App. A–B)

- **Base model:** LTX-Video 2B (v0.9.6 dev). The causal video VAE
  compresses 32× in space and 8× in time into 128 latent channels. A
  101-frame 256² simulation (padded to 105 frames) becomes a 14×8×8×128
  latent.
- **Channel adaptation:** the VAE's `conv_in` and `conv_out` are widened
  from 3 (RGB) to 4 channels, in the order ρ, m_x, m_y, p. Pressure sits in
  the new 4th slot, which is zero-initialised; the three pretrained slots
  keep their weights.
- **Data:** Euler multi-quadrant (`euler_mq`), 2D compressible Euler flow
  parameterised by γ. The runs use 256² (one arm uses 128²). Splits: 4000
  train, 500 val (seeded split of the train file), and 500 test (separate
  file, 10 γ values × 50 sims). **All headline numbers are on the test
  set.**
- **Normalisation:** per channel, u' = clip((u−μ)/(5s), −1, 1).
- **VAE baseline ("Full FT"):** the whole encoder and decoder are trained
  with loss 1.0·RMSE + 50.0·H1 (gradient loss). AdamW at LR 5e-5, with 200
  warm-up steps and cosine decay to 1e-6. 20 epochs, batch 32, seed 42.
  Every arm changes exactly one setting.
- **DiT:** the full transformer is fine-tuned with flow matching on the
  posterior-mean latents of each VAE: 8000 steps, LR 1e-5, batch 32. γ
  enters through a Fourier-feature MLP as 4 tokens, and the first latent
  frame is kept clean as the condition. Inference uses 20 steps with no
  CFG. Frame 0 is encoded with the posterior mean, the DiT generates the
  rest, and the decoder maps it back.
- **Metric:** VRMSE = sqrt(mean((û−u)²)/var(u)) per simulation, over all
  frames and pixels. The **headline metric is the channel-mean VRMSE**, the
  mean of the four per-channel VRMSEs, so each field counts equally. The
  metric is computed per simulation and averaged over the 500 test sims;
  SEM is over those 500.
- **Noise floor:** a 3-seed repeat at 128² gave about 1.1%. At 256², the
  same config on two clusters differed by 0.8%. **Differences of ≲1% are
  ties.** The SEMs and paired tests below capture test-set sampling only,
  not training variation. The DiT has only one seed per arm, so its
  training noise floor is unknown.
- **Hardware:** Intel Max 1550 (sng_pvc). The Decoder + conv_in arm ran on
  4× GH200 (jupiter) at the same effective batch size.

---

## 2. VAE reconstruction (test set, 500 sims, 256²)

Channel-mean VRMSE (± SEM). Δ is relative to the baseline. "Paired" gives
the number of test sims (out of 500) on which the arm beats the baseline.

| Group | Arm | Pooled | **Ch-mean ± SEM** | Δ | Paired: arm better | ρ | m_x | m_y | p |
|---|---|---|---|---|---|---|---|---|---|
| Adaptation | Unfinetuned (inflated only) | 0.4774 | 0.5569 ± 0.0117 | +595.9% | 0/500 | 0.1269 | 0.1116 | 0.1302 | 1.8589 |
| Adaptation | Colour adapter | 0.4081 | 0.4643 ± 0.0102 | +480.2% | 0/500 | 0.1050 | 0.0863 | 0.0924 | 1.5736 |
| Adaptation | Decoder only | 0.1061 | 0.1441 ± 0.0023 | +80.0% | 0/500 | 0.0974 | 0.0870 | 0.0938 | 0.2980 |
| Adaptation | Decoder + conv_in | 0.0677 | 0.0929 ± 0.0015 | +16.1% | 0/500 | 0.0843 | 0.0772 | 0.0803 | 0.1299 |
| — | **Full FT (baseline)** | 0.0590 | **0.0800 ± 0.0014** | — | — | 0.0727 | 0.0707 | 0.0704 | 0.1064 |
| Loss | RMSE only (no H1) | 0.0603 | 0.0814 ± 0.0014 | +1.7% | 9/500 | 0.0767 | 0.0718 | 0.0724 | 0.1048 |
| Loss | RMSE + H1 + SSIM (0.15) | 0.0656 | 0.0889 ± 0.0015 | +11.1% | 0/500 | 0.0808 | 0.0787 | 0.0786 | 0.1174 |
| LR | LR 1e-5 | 0.0683 | 0.0938 ± 0.0015 | +17.2% | 0/500 | 0.0805 | 0.0773 | 0.0778 | 0.1395 |
| Resolution | Trained at 128² | 0.0882 | 0.1200 ± 0.0019 | +49.9% | 0/500 | 0.1202 | 0.1009 | 0.1016 | 0.1572 |
| Latent reg. | KL 1e-7 | 0.0593 | 0.0804 ± 0.0014 | +0.5% | 122/500 | 0.0728 | 0.0707 | 0.0705 | 0.1077 |
| Latent reg. | KL 1e-5 | 0.0650 | 0.0877 ± 0.0014 | +9.6% | 0/500 | 0.0807 | 0.0792 | 0.0784 | 0.1126 |
| Latent reg. | SDS 1e-4 | 0.0585 | 0.0793 ± 0.0014 | −0.9% | 488/500 | 0.0723 | 0.0703 | 0.0700 | 0.1046 |
| Latent reg. | SDS 1e-2 | 0.0595 | 0.0806 ± 0.0014 | +0.8% | 59/500 | 0.0734 | 0.0714 | 0.0709 | 0.1069 |
| Latent reg. | SDS 1 | 0.0766 | 0.1040 ± 0.0015 | +29.9% | 0/500 | 0.0963 | 0.0941 | 0.0940 | 0.1316 |

Validation (500 val sims, best checkpoint, which is epoch 20 for every arm)
gives the same ranking. Each arm's Δ agrees with its test Δ to within 0.6
percentage points; the exceptions are the two arms far from the baseline
(Decoder only +77.4% val vs +80.0% test; Colour adapter +493% vs +480%).
The full table is `thesis/tables/vae_val.tex`. Baseline val ch-mean is
0.0791.

About the resolution arm: the model is trained on 128² data, but its
**test** number comes from applying it directly to the 256² test file,
the same file used for every other arm. This works because the VAE is
fully convolutional. It uses its own 128² normalisation statistics, which
are within 0.2% of the 256² ones. Its **validation** number, by contrast,
was measured on 128² data. On both, the model is about 50% worse, so the
conclusion holds whichever way it is measured. The caption of
`thesis/tables/vae_test.tex` states this.

### Findings (VAE)

1. **Full fine-tuning is needed, and the pressure channel is the reason.**
   The unfinetuned VAE's pressure error is 1.86: the pressure slot is zero
   initialised, so the pretrained VAE cannot see or produce pressure. Its
   ρ/m_x/m_y errors are only ~0.11–0.13. The colour adapter (frozen VAE,
   learned 1×1 4↔3 mapping) fixes ρ/m but still fails on pressure (1.57).
   Decoder only reaches 0.298 on pressure, because the frozen encoder never
   sees the pressure input. Unfreezing just `conv_in` closes most of the
   remaining gap (p 0.130, +16.1%), and the full encoder closes the rest
   (p 0.106). Adaptation-depth ordering: unfinetuned > adapter > decoder >
   decoder + conv_in > full FT, with every step a large effect far above
   noise.
2. **Loss:** H1 gives a small but consistent gain over RMSE alone: +1.7%,
   with the baseline better on 491 of 500 sims. The size is at the noise
   floor, so call it "small, consistent", not large. Adding the (corrected)
   SSIM at weight 0.15 costs +11.1%. SSIM is therefore left out of the
   baseline.
3. **Learning rate:** 1e-5 is +17.2% worse after the same 20 epochs, with
   pressure hit hardest (0.140 vs 0.106). 1e-4 diverged (§7). 5e-5 is close
   to the largest stable LR.
4. **Resolution:** the model trained at 128² is +49.9% worse. This is a
   clean single-variable comparison against the current baseline, which
   resolves an open caveat in `THESIS_PREP.md`.
5. **KL:** at 1e-7 it ties the baseline (+0.5%) while cutting the KL term
   about 1000× (val KL ~7.7e5 → 814). Without KL, the posterior variance
   collapses. At 1e-5 it costs +9.6% (val KL 55).
6. **SDS (score distillation from the stock, never-finetuned LTX-Video
   transformer):** 1e-4 and 1e-2 tie the baseline (−0.9%, +0.8%, both
   within noise). At 1 the cost is +29.9%.
7. **Pressure is the hardest channel for every VAE** (0.105–0.14 vs
   0.07–0.08 for the others).

---

## 3. Full pipeline: VAE + DiT (test set, 500 sims)

Each VAE gets its own DiT, trained for 8k steps on that VAE's latents with
identical settings. "VAE only" is encode→decode of the full ground-truth
sequence, i.e. the reconstruction floor. "VAE + DiT" encodes frame 0, rolls
out with the DiT, and decodes. Both columns are channel-mean VRMSE.

| VAE | VAE only | **VAE + DiT ± SEM** | Δ vs baseline | Paired: arm better | Ratio DiT/floor | ρ | m_x | m_y | p | Latent VRMSE* |
|---|---|---|---|---|---|---|---|---|---|---|
| **Full FT (baseline)** | 0.0802 | **0.4984 ± 0.0060** | — | — | 6.2× | 0.514 | 0.506 | 0.516 | 0.459 | 0.558 |
| Unfinetuned | 0.5579 | 0.9501 ± 0.0128 | +90.6% | 0/500 | 1.7× | 0.637 | 0.646 | 0.657 | 1.859 | 0.583 |
| KL 1e-7 | 0.0807 | 0.4891 ± 0.0059 | −1.9% | 323/500 | 6.1× | 0.507 | 0.499 | 0.506 | 0.445 | 0.558 |
| KL 1e-5 | 0.0879 | 0.4733 ± 0.0053 | −5.0% | 369/500 | 5.4× | 0.493 | 0.480 | 0.488 | 0.433 | 0.273 |
| SDS 1e-4 | 0.0795 | 0.4974 ± 0.0060 | −0.2% | 261/500 (p=0.38, n.s.) | 6.3× | 0.512 | 0.504 | 0.514 | 0.459 | 0.556 |
| **SDS 1e-2** | 0.0808 | **0.4583 ± 0.0054** | **−8.1%** | **443/500** | 5.7× | 0.483 | 0.465 | 0.474 | 0.411 | 0.290 |
| SDS 1 | 0.1044 | 0.5678 ± 0.0051 | +13.9% | 85/500 | 5.4× | 0.572 | 0.578 | 0.585 | 0.536 | 0.065 |

Paired Wilcoxon tests against the baseline give p < 1e-14 for every arm
except SDS 1e-4. These p-values cover test-sim sampling only. Each arm has
one DiT seed, so the deltas also contain DiT training noise of unknown size.

\* **Latent VRMSE must not be used to compare VAEs.** It is normalised by
each VAE's own latent variance, and every VAE has a different latent space.
SDS 1 has by far the lowest latent VRMSE (0.065) but the worst pixel error
of the fine-tuned arms. Use pixel VAE + DiT channel-mean only.

### Per-γ VAE + DiT channel-mean (50 test sims per γ)

| γ | Baseline | Unfinetuned | KL 1e-7 | KL 1e-5 | SDS 1e-4 | SDS 1e-2 | SDS 1 | Baseline VAE-only floor |
|---|---|---|---|---|---|---|---|---|
| 1.130 | 0.526 | 0.947 | 0.523 | 0.511 | 0.527 | **0.495** | 0.587 | 0.094 |
| 1.220 | 0.495 | 1.009 | 0.494 | 0.472 | 0.499 | **0.463** | 0.570 | 0.085 |
| 1.300 | 0.486 | 0.948 | 0.477 | 0.463 | 0.479 | **0.443** | 0.552 | 0.081 |
| 1.330 | 0.494 | 1.018 | 0.479 | 0.470 | 0.493 | **0.451** | 0.564 | 0.077 |
| 1.365 | 0.478 | 0.924 | 0.476 | 0.469 | 0.479 | **0.450** | 0.578 | 0.076 |
| 1.400 | 0.487 | 0.866 | 0.478 | 0.465 | 0.487 | **0.456** | 0.562 | 0.077 |
| 1.404 | 0.470 | 0.940 | 0.460 | 0.453 | 0.466 | **0.435** | 0.542 | 0.077 |
| 1.453 | 0.489 | 0.956 | 0.478 | 0.471 | 0.488 | **0.452** | 0.570 | 0.077 |
| 1.597 | 0.540 | 0.928 | 0.526 | 0.486 | 0.541 | **0.483** | 0.582 | 0.081 |
| 1.760 | 0.521 | 0.965 | 0.500 | 0.472 | 0.515 | **0.456** | 0.572 | 0.080 |

### Findings (pipeline)

1. **The DiT is the bottleneck.** For every fine-tuned VAE the rollout error
   (~0.46–0.57) is 5–6× the reconstruction floor (~0.08). VAE improvements of
   a few percent in reconstruction are therefore invisible downstream, while
   properties of the latent space matter more.
2. **Fine-tuning the VAE is still essential.** With the unfinetuned VAE the
   pipeline error is 0.95 (+90.6%). Most of it comes from the pressure
   channel (1.86), which is exactly the VAE's pressure failure. Its low
   1.7× ratio only reflects the bad floor and is not a good result; don't
   present it as one.
3. **Moderate latent regularisation helps the DiT even when it costs, or
   does not change, reconstruction:**
   - **SDS 1e-2 is the best pipeline**: −8.1% vs the baseline, better on
     443/500 sims and at **every one of the 10 γ values**, while its VAE
     reconstruction ties the baseline (+0.8%).
   - KL 1e-5 gives −5.0% downstream, although its VAE is +9.6% worse at
     reconstruction. KL 1e-7 gives −1.9%, which is small and close to the
     plausible noise level.
   - SDS 1e-4 has no effect at either stage. SDS 1 hurts both
     reconstruction (+29.9%) and pipeline (+13.9%). The SDS response is
     therefore non-monotonic: too weak does nothing, too strong degrades
     the decoder's input information.
   - **Reconstruction quality does not predict pipeline quality.** The
     ranking by VAE-only error differs from the ranking by VAE + DiT error.
     This is the central methodological point: a VAE for latent diffusion
     must be judged by the downstream model, not by reconstruction alone.
4. **Channels:** in the pipeline, pressure has the *lowest* error of the
   four fields (0.41–0.46 for the fine-tuned VAEs), the reverse of the
   VAE-only ordering. The DiT's error is spread across ρ, m_x and m_y.
5. **γ dependence:** the extreme γ values (1.13, 1.597, 1.76) are hardest
   for the DiT. The gains from SDS 1e-2 and KL 1e-5 are largest there: at
   γ = 1.76, SDS 1e-2 is −12.5% and KL 1e-5 −9.4%.

Interpretation to offer, with appropriate hedging: KL and SDS both make the
latent distribution better conditioned for the generative model. KL keeps
the posterior from collapsing to near-deterministic, badly scaled latents,
and SDS pulls the latents toward what a pretrained video DiT finds easy to
denoise. The latent statistics support the "better conditioned" part, but
a mechanism is not proven, so present it as a hypothesis.

---

## 4. Figures (final set, in `figures/`)

1. **`figures/ch3_euler_mq_2507.png`** (Ch. 3, dataset): raw physical
   fields ρ, m_x, m_y, p of sim `2507`, γ = 1.40, at t = 0, 50, 100.
2. **`figures/ch6_vae_dit_frame100_2507.png`** (Ch. 6): baseline pipeline
   at frame 100 of the same sim. Panels: GT | VAE only | VAE + DiT |
   |error| VAE only | |error| VAE + DiT. Whole-sim channel-mean VRMSE on
   this sim: VAE only 0.0930, VAE + DiT 0.4668.

**Caption requirement:** sim 2507 is a **training** simulation, picked for
its rich shock structure. The caption must say so, and it must not be
presented as a test result. The styling is viridis for fields and coolwarm
for signed residuals.

All other results (ablation bars, heatmaps, training curves, error vs γ)
are deliberately shown as **tables**, not figures.

---

## 5. Evaluation-protocol note (posterior mean)

The DiT is trained on posterior-mean latents, so at inference the
conditioning frame 0 must be encoded with the **posterior mean** too, not
with a posterior sample (fixed in commit 560f4ad). All numbers in §3 use
the posterior mean. VAEs without KL have a collapsed posterior (σ ≈ 0), so
for them sample and mean coincide; their numbers were unchanged by the fix
(≤ 1e-4). The KL arms were badly affected before the fix: KL 1e-7 scored
1.476 and KL 1e-5 scored 2.058, against 0.489 and 0.473 now. For the thesis
it is enough to state in the method section that conditioning frames are
encoded with the posterior mean, matching training. The earlier numbers
must not appear anywhere.

---

## 6. Suggested Chapter 6 structure

1. Protocol recap (cite App. A–B): test set, channel-mean VRMSE, ~1% noise
   floor.
2. VAE adaptation strategy (Group 1): unfinetuned → adapter → decoder →
   decoder + conv_in → full FT; the pressure-channel story.
3. Loss ablation (Group 2): H1 is a small gain, SSIM hurts.
4. Learning rate (Group 3) and training resolution.
5. Latent regularisation at the VAE level (KL, SDS): mostly ties at small
   weights, cost at large ones.
6. Full pipeline (VAE + DiT): the DiT is the bottleneck; unfinetuned
   collapses; SDS 1e-2 is best, and KL 1e-5 helps despite worse
   reconstruction; reconstruction ≠ generation quality; γ breakdown.
7. Qualitative example (figure 2).
8. Summary of the recommended recipe: full FT, RMSE + H1, LR 5e-5, 256²,
   plus mild latent regularisation (SDS 1e-2, or KL) when a downstream DiT
   is trained.

---

## 7. Discarded / excluded runs (state briefly, e.g. in App. A)

- **LR 1e-4** and **KL 1e-6** collapsed in epoch 2, right after warm-up
  reached peak LR: val VRMSE jumped from ~0.15 to ~1.07 (near-constant
  output) and never recovered. KL 1e-6 is treated as a training
  instability, not a KL effect, because 1e-7 and 1e-5 both trained
  normally. Neither run appears in any table.
- **MLW loss comparison**: dropped (not meaningful under the final
  baseline).
- **Old-SSIM runs**: an earlier SSIM implementation was wrong for signed
  [−1,1] fields. Every run trained with it (the older full_ft,
  decoder_only, color_adapter, lr sweep, and the pre-Ch6 SDS-weight sweep)
  is excluded. All Ch6 numbers use the corrected setup.

---

## 8. Limitations to state honestly

- One seed per arm. The training noise floor comes from a 3-seed study at
  128² (~1.1%) and a two-cluster repeat at 256² (0.8%). The DiT has no
  seed repeats, so pipeline differences of a few percent (KL 1e-7's −1.9%)
  are suggestive only. SDS 1e-2 (−8.1%, consistent across all γ) and
  KL 1e-5 (−5.0%) are well above the VAE-level noise floor.
- DiT training is 8k steps with identical settings for every VAE. It is
  not tuned per VAE and is probably not converged.
- One PDE family (Euler multi-quadrant) and one conditioning scalar (γ).
- One resolution for the DiT (256²). The 512² native resolution was not
  completed.
- No wall-clock comparison against a numerical solver.
- The mechanism by which KL/SDS help the DiT is a hypothesis.

---

## 9. Superseded claims in `THESIS_PREP.md` — do NOT use

`THESIS_PREP.md` (2026-09-22) is still useful for background, literature,
and methodology prose, but all of its **results claims are replaced** by
this file. In particular:

- ✗ "KL regularisation makes the DiT monotonically worse / latent shift /
  KL destroys the value of DiT fine-tuning." This was caused by the
  posterior-sampling evaluation bug (§5). After the fix, KL **improves**
  the pipeline slightly (−1.9% / −5.0%).
- ✗ Every DiT number in its §5.3 tables (5.9×, 15×, 18.8×, 21.4×, 22.7×,
  the lundquist 4.40× / 4.88× / 12.58×). These are older protocols
  (675-sim val subsets, other clusters, 128², old SSIM), and the KL rows
  are affected by the bug.
- ✗ The decoder-on-rollout fine-tune (−15.5%, lundquist). It was never
  confirmed or repeated; leave it out, or at most mention it in one
  sentence as future work.
- ✗ The VAE trajectory numbers (0.0947 → 0.0787 → 0.0543) come from older
  baselines (with SSIM/anchor losses). Use §2 of this file.
- ✗ "Resolution effect not cleanly isolated": now isolated (+49.9% at 128²).
- ✗ lrz_ai and the 512² run: not part of the thesis results.
- Still valid in spirit: the SDS formulation with the stock critic, which
  addresses the circularity concern since the critic never saw this data.
  So is the general lesson that the VAE must be judged downstream, though
  the direction of the KL effect is now reversed.

---

## 10. Source pointers

| What | Where |
|---|---|
| VAE test per-sim results | `finetune_vae_outputs/<cluster>/finetune_vae_ch6_<arm>/test_eval_256.json` |
| VAE training/val curves | `finetune_vae_outputs/<cluster>/finetune_vae_ch6_<arm>/metrics/metrics.csv` |
| DiT eval (final) | `logs/sng_pvc/eval_dit_vrmse/{546059..546065}/vrmse_summary.json` (baseline, unfinetuned, KL 1e-7, KL 1e-5, SDS 1e-4, SDS 1e-2, SDS 1) |
| Figure sim eval | `logs/sng_pvc/eval_dit_vrmse/546066/` (sim 2507, train split) |
| Table generator | `scripts/thesis_tables.py` → `thesis/tables/*.tex` |
| Appendices A–C | `thesis/appendix.tex` |
| Configs | `configs/finetune_vae/finetune_vae_ch6_*`, `configs/dit/train_dit_sng_pvc_ch6_*` |
