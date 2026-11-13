# MAGI-1 finetuning: notes

Module overview: [`__init__.py`](__init__.py). Outputs and checkpoints are on SCRATCH
(`/e/scratch/e-dev-2026d09-262/ig_work`, see its `README.md`); only code is here.

## Training has no step count; sampling does

- **Training** (`train/train_full.py`, `dit.py`). Each chunk gets one random t (t = 0 is noise, t = 1 is clean),
  drawn with the sd3 shift 3 and sorted so that earlier chunks are cleaner. Frame 0 stays clean.
  - The input is `x_t = t·x1 + (1−t)·noise`.
  - The target velocity `x1 − noise` is known exactly, because we made `x_t` ourselves.
  - So there is one forward and one backward pass per sample, with no denoising steps.
  - The model learns every t because t is random per sample.
  - With shift 3: median t = 0.25 (75 % noise), quartiles 0.1 and 0.5, and only about 25 % of samples are more
    image than noise.
- **Sampling** (`dit.rollout`) has to integrate from noise without knowing `x1`, which is where steps come from.
  - It uses Euler steps per chunk with 4 chunks in the window, so `num_steps` must be a multiple of 4.
  - MAGI-1's 64 steps is only the config default. The base model is not trained for any particular step count;
    MAGI's distilled model is (shortcut distillation over 64/32/16/8 steps, plus CFG distillation).
- **Why it blurs.** One noisy `x_t` fits many possible `x1`, so under MSE the model learns the *average*
  velocity, and sampling drifts toward the average outcome.

## Denoising-steps ablation (2026-11-10)

Step-8000 checkpoint (`magi1_dit_full_8000steps_on_F3`), shift 5, 30 test sims + `0456_gamma1.7599999905`.
Rollouts are saved.
Outputs: `magi1_outputs/diagnostics/steps_ablation/steps<N>_shift5/` (results.md, videos of 0456, rollouts/).

| steps | VRMSE mean (all frames) | Δ vs 64, paired | sims better than 64 | mlw | s / trajectory |
|---|---|---|---|---|---|
| 4 | 0.2565 | −0.0087 ± 0.0033 | 26/31 | 65.61 | 3.2 |
| 8 | 0.2590 | −0.0062 ± 0.0017 | 26/31 | 65.37 | 6.1 |
| 16 | 0.2616 | −0.0035 ± 0.0007 | 26/31 | 65.27 | 12.1 |
| 32 | 0.2635 | −0.0016 ± 0.0003 | 27/31 | 65.27 | 24.1 |
| 64 | 0.2652 | 0 | – | 65.28 | 47.3 |
| 128 | 0.2666 | +0.0014 ± 0.0002 | 2/31 | 65.33 | 94.2 |

- **Fewer steps score better on every pointwise metric** (rmse, vrmse, 1-ssim, h1, h2).
- **mlw is the exception.** It compares log-magnitudes of wavelet detail coefficients, which punishes missing
  detail, and it is the only metric that gets worse at 4 steps.
- **Visually, 4 and 64 steps barely differ** (Harish). Wan 5 vs 50 steps differs clearly. MAGI is already blurry
  at 64 and 128 steps, so its blur comes from the model, not the sampler.
- **Shift.** Shift 5 was best of shifts 1–5 at 64 and 128 steps. At 4 steps shifts 7 and 10 score slightly
  lower but are blurrier (see the shift table under Metrics).
- **Compute.** 4 steps makes a 500-sim eval about 70–80 core-h instead of about 530. Training cost does not change.
  Earlier results (64 steps) are not comparable with 4-step numbers without re-scoring.

## Metrics: VRMSE prefers blur

- **Why.** VRMSE is a pointwise squared error, whose optimum is the mean over possible futures, which is blurry.
  A sharp but slightly misplaced feature is penalised twice (where it should be and where it is), while a
  smeared one is penalised once.
- **The other benchmark metrics.** h1 and h2 are pointwise too (relative error of the 1st and 2nd differences).
  mlw is closer to measuring sharpness but still location-matched.
- **Sharpness measures** ([`eval/sharpness.py`](eval/sharpness.py)), reported next to the VRMSE. They compare
  statistics of prediction and truth instead of pixels (prediction / truth: 1 = right, < 1 = blurred).
  - **Power ratio per wavenumber band:** radially averaged power spectrum, bands k 1..3 up to k 64..128.
  - **Gradient ratio:** mean of |grad|, and its 99th percentile (the steepness of the fronts).
  - **Error per band:** where the error sits in wavenumber.
  - **Distance to a reference run:** how far two samplers' rollouts are from each other, in VRMSE units.
  - It runs on the saved `rollouts/` of any `evaluate.py --save-rollouts` folder, without generating again,
    and adds the VAE round trip as the ceiling.
- **Result (2026-11-10, `diagnostics/sharpness/steps/sharpness.md`): fewer steps are more accurate and blurrier.**
  Step 8000, shift 5, 31 sims; ratios are prediction / truth.

  | run | VRMSE | gradient p99 | power k 8..15 | power k 16..31 | power k 32..63 | distance to 64 steps |
  |---|---|---|---|---|---|---|
  | VAE round trip (ceiling) | 0.0794 | 1.018 | 0.988 | 0.943 | 0.814 | – |
  | 4 steps | 0.2565 | 0.947 | 0.913 | 0.832 | 0.746 | 0.114 |
  | 8 steps | 0.2590 | 0.965 | 0.926 | 0.855 | 0.759 | 0.076 |
  | 16 steps | 0.2617 | 0.973 | 0.931 | 0.864 | 0.763 | 0.047 |
  | 32 steps | 0.2635 | 0.977 | 0.933 | 0.871 | 0.767 | 0.027 |
  | 64 steps | 0.2652 | 0.981 | 0.936 | 0.876 | 0.770 | 0 |
  | 128 steps | 0.2666 | 0.984 | 0.939 | 0.880 | 0.773 | 0.022 |

  - VRMSE ranks 4 steps best; the front steepness (gradient p99) and the power at k 16..31 rank 128 steps best.
  - The effect is small: 4 steps lose about 4 points of power at k 16..31 against 64 steps, and every rollout
    is well below the VAE ceiling there (0.83 to 0.88 against 0.94).
  - The mean gradient ratio is not usable: it is about 1.8 even for the VAE round trip, because the decoder
    adds a faint texture to the flat regions of the truth. The same texture holds the k 64..128 ratio at 0.89.
- **Shift at 4 steps** (`diagnostics/sharpness/shift_at_4steps/`, runs in `diagnostics/shift_sweep_4steps/`):

  | shift | 1 | 2 | 3 | 5 | 7 | 10 |
  |---|---|---|---|---|---|---|
  | VRMSE | 0.2692 | 0.2643 | 0.2603 | 0.2565 | 0.2560 | 0.2552 |
  | gradient p99 | 0.947 | 0.952 | 0.951 | 0.947 | 0.942 | 0.936 |
  | power k 16..31 | 0.833 | 0.839 | 0.837 | 0.832 | 0.826 | 0.818 |

  VRMSE keeps falling with the shift but flattens after 5; above 3 the rollouts get blurrier. Shift 5 stays
  the default.
- **Still planned: an ensemble score (CRPS or energy score)** over N noise seeds. It is a proper score, so blur
  is not optimal. At 4 steps, 8 members cost about the same as one 32-step rollout.

## Latent channels: larger relative error in the fine-detail channels

From `diagnostics/latent_sensitivity` and `diagnostics/dit_latent_error` (step 8000, 64 steps, shift 5, 30 sims),
latents in the DiT's space:

| latent channel | std | DiT error, absolute | DiT error / std | share of the latent's power in k >= 8 |
|---|---|---|---|---|
| 13 | 0.352 | 0.176 | 0.50 | 0.295 |
| 11 | 0.407 | 0.172 | 0.42 | 0.273 |
| 0 | 0.416 | 0.152 | 0.36 | 0.178 |
| ... | | | | |
| 8 | 0.829 | 0.190 | 0.23 | 0.096 |
| 5 | 0.953 | 0.201 | 0.21 | 0.081 |
| 2 | 1.009 | 0.172 | 0.17 | 0.075 |

- **The DiT's absolute error is about the same in every channel** (0.15 to 0.20). That is what the unweighted
  velocity MSE asks for.
- **So the relative error is 3× larger in the weakest channel than in the strongest.** Over the 16 channels
  the correlation of std with relative error is −0.92.
- **The weak channels are the fine-detail ones** (correlation of std with the fine-detail share: −0.86).
- **Noise covers them first.** A channel of std s is above the noise only for t > 1 / (1 + s): t > 0.74 for
  channel 13, which is about 10 % of the training samples at shift 3.
- **The decoder does not amplify the weak channels** (part 3 of `diagnostics/sharpness/steps/sharpness.md`:
  noise of the same absolute size, 0.17, in one channel at a time, 10 sims).
  - The decoded change is 0.040 to 0.050 (VRMSE units) for the six weakest channels and 0.056 to 0.066 for the
    six strongest. For pixel error the unweighted loss is therefore about right, and weighting by 1 / std²
    would move effort to channels that matter less for the VRMSE.
  - What the weak channels change is finer: 57 to 68 % of their effect is at k >= 16, against 49 to 54 % for
    the strong ones. Weighting them up could trade VRMSE for sharpness; it is not a free gain.
  - The test uses white noise; the DiT's real error is smoother (52 % of it at latent k 8..16, white noise 79 %).
- **Other candidate: predict the change from frame 0** (Harish): each latent minus the frame-0 latent, rescaled
  per channel. This changes the model's input space, so it may need more than a short continuation. Not tested.

## Physics constraints: what was checked (2026-11-10)

Quick checks on 6 test sims (step 8000, 64 steps, shift 5, physical units) and 58 train sims' cached latents.
"Oracle" means the constraint was imposed with knowledge of the truth, so the gain is an upper bound for any
strong (built-in) or weak (loss penalty) version of it.

| idea | finding | verdict |
|---|---|---|
| Conserve total mass, momentum, energy | The domain is open: the truth's totals drift by 12 to 100 % over 100 frames. The rollouts' totals are off by 0.5 to 2 % (momentum up to 9 %). Oracle: VRMSE 0.2549 -> 0.2500. | at most 2 %, and frame 0 does not give the totals |
| Positivity of density and pressure | at most 0.03 % of pixels are non-positive; clamping changes nothing | nothing to gain |
| Keep untouched regions at frame 0 | 7 % of pixel-frames are still at their initial value (21 to 59 % in frames 1..20) and hold 0.9 % of the error. Oracle: 0.2549 -> 0.2535. | nothing to gain |
| PDE residual as a loss (mass equation, d rho/dt + div m = 0) | Frames are about 4 grid cells of motion apart, so the residual between saved frames is 48 to 66 % of the density change for the truth itself, and the same for the rollout (0.51 vs 0.52, 0.48 vs 0.46). | cannot tell truth from prediction at this frame spacing |
| Hamiltonian NN ideas | Needs a closed, non-dissipative system integrated in time. Here the domain is open, shocks dissipate, and the DiT generates all frames jointly. | does not transfer |
| Change from frame 0, dz_k = z_k - z_0 (Harish) | Removes the channel offsets (largest mean 1.30 -> 0.10) and the frame-mean share of the power (0.70 -> 0.27). But RMS(dz_k) / RMS(z_k) is 0.55, 0.82, 0.97 for latents 1, 2, 3 and 1.26 from latent 15 on: after 12 frames the flow no longer resembles frame 0, so the difference carries frame 0's structure as well. | simpler target for the first 12 frames only |
| Change from the previous latent, z_k - z_(k-1) | std 0.29 against 0.67 for z itself | the compact target, but errors add up along the sum; not tested |

Not checked: the energy field as an extra metric (needs only the 4 fields and gamma), symmetry augmentation
(flips and 90-degree rotations with the momentum components swapped). Training loss and the test one-step loss
are about equal (0.024 and 0.02 to 0.03), so more data variety is not the obvious bottleneck.

## Latent-space experiment: the standardized latent space (`--latent-prior`)

- **Subtracting a known latent changes nothing by itself.** With x' = z - z_0 the noisy input is the old one
  minus t z_0 and the velocity target is the old one minus z_0. The model is given z_0 anyway, so the loss and
  its optimum are the same. What changes the problem is the size of the noise relative to the signal.
- **How the three "change" variants compare** (116 train sims; RMS relative to the latents' own variation):

  | target | chunk 1 (latents 1..5) | chunk 2 | chunk 3 | chunk 4 | overall |
  |---|---|---|---|---|---|
  | z - z_0 (Harish's sketch) | 0.60 .. 1.13 | 1.16 .. 1.23 | 1.24 .. 1.28 | 1.29 .. 1.31 | 1.20 |
  | z - last latent of the previous chunk | 0.60 .. 1.13 | 0.58 .. 0.92 | 0.40 .. 0.66 | 0.34 .. 0.53 | 0.69 |
  | z - previous latent | 0.60 .. 0.74 | 0.42 .. 0.58 | 0.35 .. 0.40 | 0.30 .. 0.34 | – |

  The flow changes fastest at the start and slows down. Differences to a generated latent are small later on,
  but that latent is itself generated at inference, so its error is added to every latent that builds on it.
- **`--latent-prior <latents>/stats.json`** (`dit.LatentPrior`) is the variant that changes the noise-to-signal
  balance without any of that: flow matching in z' = (z - mean_c) / std_c. Like the difference to frame 0 it
  removes the channel offsets (up to 1.3), and it gives every channel the same signal-to-noise ratio (today the
  channels' std ranges from 0.35 to 1.0 under the same unit noise).
  - In the DiT's own space it is two changes: the noise is mean_c + std_c eps (training and sampling), and the
    squared error is divided by std_c^2. Noise std per channel 0.59 to 1.39 (mean square 1), loss weights
    0.39 to 2.19 (mean 1).
  - The network and its inputs do not change, so it continues from step 8000 and compares directly with the
    plain 2e-6 continuation (same start, steps and schedule).
  - A model trained with it must be sampled with it: `evaluate.py` reads it from the run's `args.yaml`. The
    other samplers (`dit_latent_error.py`, `checkpoint_curve.py`) do not pass it yet.
  - **Result (2026-11-11, +2000 steps at 2e-6, scored with 4 steps, shift 5, 31 sims):** no gain. Benchmark
    VRMSE 0.2806 against 0.2660 for the plain run with the same schedule (+0.015 ± 0.011, worse on 23 of 31
    sims) and no sharper (gradient p99 0.947 against 0.952). At 2e-6 the model has little room to adapt to
    the new noise, so this does not rule the idea out at a higher learning rate; it is not a quick win.

## More training, seed spread, entropy rule (2026-11-11)

All with 4 Euler steps, shift 5, on 30 sims + 0456 (job 2842019); benchmark VRMSE on frames 10, 20, ..., 90.
Folders: `diagnostics/big_run_checkpoints/`, `latent_prior_probe/`, `seed_spread/`, `sharpness/after_big_run/`,
`seed_spread_and_entropy/`.

**The +6000 run at 2e-6 over training** (`full_8000_plus6000_lr2e-6_fast_on_F3`; start = step 8000):

| checkpoint | learning rate there | benchmark VRMSE | against the start (paired) | gradient p99 | power k 16..31 |
|---|---|---|---|---|---|
| start | – | 0.2690 | – | 0.947 | 0.832 |
| +1000 | 1.9e-6 | 0.2915 | +0.023 ± 0.005 | 0.932 | 0.806 |
| +2000 | 1.5e-6 | 0.2794 | +0.010 ± 0.005 | 0.950 | 0.838 |
| +3000 | 1.0e-6 | 0.2747 | +0.006 ± 0.005 | 0.951 | 0.835 |
| +4000 | 6e-7 | 0.2665 | -0.003 ± 0.005 | 0.951 | 0.837 |
| +5000 | 2.6e-7 | 0.2550 | -0.014 ± 0.004 | 0.959 | 0.847 |
| +6000 | 1e-7 | 0.2486 | -0.020 ± 0.005 (26 of 31 sims) | 0.955 | 0.842 |

- A restart first makes the rollout worse: the fresh optimizer at 2e-6 costs 0.02 in the first 1000 steps.
  The gain appears only as the learning rate decays below about 5e-7, and it was still growing at the end
  (+5000 to +6000: -0.0065 ± 0.0023).
- So a run is only as good as its last, low-learning-rate part: score checkpoints from the end of the
  schedule, not from the middle. The plain +2000 run decays to 1e-7 within 2000 steps and gains 0.003.
- The gain is not blur: the +6000 model is slightly sharper than the start, and its error is lower in every
  wavenumber band (k 4..7: 0.151 -> 0.127, k 16..31: 0.466 -> 0.429 of the truth's power).

**Seed spread** (`eval/spread_and_entropy.py`; +6000 model, 4 seeds per sim, all frames). The simulation is
deterministic, so any difference between seeds is error.

| measure | value |
|---|---|
| VRMSE of one sample (4 seeds) | 0.2385, 0.2365, 0.2270, 0.2394 |
| VRMSE of the mean of the 4 samples | 0.2125 |
| spread between seeds (std / std of the truth) | 0.117 |
| share of the squared error that is seed variance | 24 % (density 26 %, momentum 25 %, pressure 21 %) |
| the same per quarter of the frames | 17 %, 25 %, 27 %, 27 % |

- Three quarters of the squared error is the same for every seed (the model is consistently wrong), one
  quarter is randomness of the sample. Removing all of the randomness would lower VRMSE by about 13 %
  (0.235 -> 0.205); the mean of 4 samples already gets 11 %, at the price of a blurrier field (not measured).
- Measured at 4 steps only; the spread at 64 steps is not known.

**Entropy rule.** For an exact solution p / density^gamma never falls below its frame-0 minimum (shocks only
raise it). The truth obeys it: 0.0002 % of pixels are more than 1 % below.

| rollouts | pixels > 1 % below | > 5 % below | > 20 % below | VRMSE | with the pressure raised to the floor |
|---|---|---|---|---|---|
| +6000 model, 4 steps | 1.15 % | 0.29 % | 0.05 % | 0.2385 | 0.2384 |
| step 8000, 64 steps | 1.06 % | 0.26 % | 0.05 % | 0.2651 | 0.2650 |
| step 8000, 4 steps | 1.10 % | 0.28 % | 0.05 % | 0.2563 | 0.2563 |

The rollouts break the rule in about 1 % of pixels, but repairing them changes VRMSE by 0.0001: nothing to
gain for accuracy, from a projection at inference or from a loss in training.

## Where the rollout error comes from: true context (2026-11-12)

`eval/teacher_forced.py` (job 2853192; +6000 model, shift 5, the 30 diagnostic sims). Each chunk is generated
with the same start noise and schedule, once with the model's own earlier chunks (free rollout) and once with
the true earlier chunks, noised to the level the sampler has them at ("true context", what training shows the
model). Folder: `diagnostics/teacher_forced/`.

| chunk | frames | latent error, free | true context | VRMSE of its frames, free | true context |
|---|---|---|---|---|---|
| 1 | 1..20 | 0.219 | 0.219 | 0.214 | 0.213 |
| 2 | 21..44 | 0.326 | 0.224 | 0.215 | 0.145 |
| 3 | 45..68 | 0.371 | 0.188 | 0.223 | 0.108 |
| 4 | 69..92 | 0.404 | 0.171 | 0.244 | 0.093 |
| 5 | 93..100 | 0.431 | 0.106 | 0.266 | 0.062 |
| all | 1..100 | 0.356 | 0.201 | 0.240 | 0.148 |

4 Euler steps. With 64 steps on 12 of the sims the picture is the same (latent error 0.205, 0.337, 0.392, 0.425,
0.447 free; 0.205, 0.240, 0.206, 0.186, 0.119 with true context). With fully clean true context the later
chunks are slightly better still (0.204, 0.171, 0.153, 0.103).

- All of the growth from chunk to chunk is error carried over from the earlier chunks. Given the true flow so
  far, the model predicts every later chunk better than it predicts chunk 1 from frame 0.
- So later frames are not harder for the model, and the later chunk positions are not under-trained near the
  clean end (clean context works at least as well).
- Every chunk adds an error of its own of about the same size (0.17 to 0.22 with true context; chunk 1 is not
  special), and the errors of the chunks so far add up almost like independent ones: the root of the summed
  squares of the true-context errors gives 0.220, 0.314, 0.366, 0.404, 0.418 against 0.220, 0.326, 0.371, 0.404,
  0.431 in the free rollout. So the model is hardly thrown off by its own imperfect context beyond carrying the
  error along: little to gain from training on its own rollouts (context noise gave nothing either). What is
  left to improve is the error of one chunk given good context, which more training lowers.
- The true-context numbers are a bound, not a target: in a rollout the context is never the truth, and an
  error in the state at frame 20 also grows in the real flow. The test does not say how much of the carried
  error a model trained on its own rollouts could remove.
- Checks: the sampler in this script reproduces `dit.rollout` exactly (difference 0). Chunk 1 generated without
  the later chunks differs from chunk 1 of the free rollout by at most 0.04 (4 steps) and 0.014 (64 steps) in
  latents whose spread is about 0.66: later chunks have at most a small influence on earlier ones.
- Training draws five independent noise levels and sorts them, chunk 1 the cleanest, so chunk 1 is below
  t = 0.1 in 0.1% of the samples and below t = 0.25 in 3%, although every rollout starts chunk 1 at t = 0 with
  nothing but frame 0. The chunk-1 test below shows this is not where chunk 1's error comes from.

**Chunk-1 test** (`--chunk1-path`, job 2870705; same model and sims, 4 steps): full free rollouts in which chunk 1
is held on the truth's path t z + (1 - t) eps for the first part of its steps and then released (with shift 5
the quarters end at t = 0.0625, 0.167, 0.375).

| chunk 1 on the true path for | latent error chunk 1 | chunk 2 | chunk 3 | chunk 4 | chunk 5 | VRMSE, all frames |
|---|---|---|---|---|---|---|
| none of its steps (free rollout) | 0.220 | 0.326 | 0.371 | 0.404 | 0.431 | 0.240 |
| the first quarter | 0.193 | 0.312 | 0.365 | 0.399 | 0.422 | 0.227 |
| the first half | 0.177 | 0.298 | 0.359 | 0.394 | 0.419 | 0.218 |
| three quarters | 0.155 | 0.277 | 0.349 | 0.388 | 0.416 | 0.209 |
| all of them (chunk 1 = truth) | 0 | 0.224 | 0.324 | 0.372 | 0.407 | 0.188 |

- Chunk 1's error is not decided in its first, rarely trained steps: a true first quarter removes 12% of it,
  and with three quarters true the last quarter alone still leaves 0.155 of 0.220. No support for changing how
  training draws chunk 1's noise level. (At 4 steps each quarter is one Euler step; not repeated at 64 steps.)
- With chunk 1 exactly true the later chunks get the errors the free rollout has one chunk earlier (0.224,
  0.324, 0.372, 0.407 against 0.220, 0.326, 0.371, 0.404): the error depends on how many chunks were generated
  since the last true state, not on which chunk it is.

## The +10,000 run at 5e-6 from the +6000 model (2026-11-12)

`full_14000_plus10000_lr5e-6_fast_on_F3` (500 warmup steps, cosine to 1e-7; 21.2 h, 6,107 core-h), scored like
the +6000 run: 4 Euler steps, shift 5, 30 sims + 0456 (job 2870705; `diagnostics/run10k_checkpoints/`,
`sharpness/after_10k_run/`).

| checkpoint | learning rate there | benchmark VRMSE | against the start (paired) | gradient p99 | power k 16..31 |
|---|---|---|---|---|---|
| start (+6000 model) | – | 0.2486 | – | 0.955 | 0.842 |
| +1000 | 5.0e-6 | 0.3433 | +0.095 ± 0.009 | 0.939 | 0.827 |
| +2000 | 4.7e-6 | 0.3348 | +0.086 ± 0.009 | 0.947 | 0.830 |
| +3000 | 4.2e-6 | 0.3149 | +0.066 ± 0.006 | 0.955 | 0.841 |
| +4000 | 3.5e-6 | 0.2889 | +0.040 ± 0.006 | 0.961 | 0.847 |
| +5000 | 2.8e-6 | 0.2793 | +0.031 ± 0.005 | 0.961 | 0.851 |
| +6000 | 2.0e-6 | 0.2688 | +0.020 ± 0.005 | 0.961 | 0.850 |
| +7000 | 1.2e-6 | 0.2545 | +0.006 ± 0.007 | 0.960 | 0.845 |
| +8000 | 6.2e-7 | 0.2412 | -0.007 ± 0.006 | 0.969 | 0.860 |
| +9000 | 2.3e-7 | 0.2271 | -0.022 ± 0.006 | 0.967 | 0.858 |
| +10000 | 1e-7 | 0.2264 | -0.022 ± 0.006 (27 of 31 sims) | 0.967 | 0.856 |

- Same pattern as the +6000 run, larger at the higher learning rate: the restart costs 0.095, the model is
  back at its start only after 7000 steps, and the gain comes in the last 3000 as the learning rate decays.
- The final model is better and sharper than its start. From the 8000-step model (0.2690) the two
  continuations together give 0.2264. Training loss per 1000 steps fell from 0.0267 to 0.0229.
- Cost per gain: -0.020 for 3,750 core-h at 2e-6 over 6000 steps, -0.022 for 6,107 core-h at 5e-6 over 10,000.

On the 500 test sims (2026-11-13; 64 steps, shift 3, frames 10, 20, ..., 90; job 2878537, 1.8 h, 525 core-h;
`dit_experiments/full_14000_plus10000_lr5e-6_fast_on_F3/eval/`):

| model | all frames | RMSE | VRMSE (median) | 1-SSIM | MLW |
|---|---|---|---|---|---|
| 8000 steps | 0.2751 | 0.0343 | 0.2864 (0.268) | 0.033 | 65.2 |
| + 6000 at 2e-6 | 0.2570 | 0.0320 | 0.2667 (0.249) | 0.031 | 64.4 |
| + 10,000 at 5e-6 | 0.2384 | 0.0297 | 0.2464 (0.232) | 0.029 | 63.5 |
| *Wan 256 (Harish's row 3)* | | *0.028* | *0.234* | *0.025* | *51.2* |

- Paired VRMSE: -0.0204 ± 0.0008 against the +6000 model (476 of 500 sims better), -0.0400 ± 0.0015 against
  the 8000-step model (493 of 500). Each of the 10 gammas is better by 0.016 to 0.024.
- The 4-step score on 31 sims predicted the 500-sim change well (-0.022 there, -0.020 here).

## Symmetry ensemble, fewer steps, train / val / test (2026-11-13)

`magi_finetuning/eval/sampling_sweep.py`, job 2889885 (16 min 50 s, 81 core-h). Outputs:
`diagnostics/sampling_sweep/run10k_step10000/` (summary.md, scores per sim, the fields of sim 0456,
`blur_0456_density.png`). +10,000 model, 4 Euler steps, shift 5, 4 chunks at once, 30 test sims + 0456.

**Symmetries.** The Euler equations and the open boundaries are unchanged by flips and by swapping x and y, so
the rollout from a transformed frame 0, transformed back, is another prediction of the same simulation. The
fields are stored (..., x, y): momentum_x points along the second-to-last array axis, not the last (least squares
of the mass equation on test.h5: R² 0.37 to 0.68 for this pairing, 0.01 to 0.06 for the other). A flip along an
axis negates the momentum along it; the swap exchanges the two momenta. `check_transforms` repeats the
mass-equation fit on every transformed truth before a model is loaded. The initial conditions look symmetric
(150 train sims: 1, 3 or 5 interfaces along each axis with the same position statistics; momentum signs balanced).

| prediction | VRMSE | against one sample (paired) | RMSE | 1-SSIM | MLW | gradient p99 | power k 16..31 |
|---|---|---|---|---|---|---|---|
| one sample | 0.2264 | – | 0.0246 | 0.0233 | 63.7 | 0.967 | 0.856 |
| mean of 2 seeds | 0.2143 | -0.0121 ± 0.0033 (27 of 31 sims) | 0.0229 | 0.0212 | 62.5 | 0.928 | 0.813 |
| mean of 4 seeds | 0.2086 | -0.0177 ± 0.0037 (29 of 31) | 0.0223 | 0.0204 | 61.9 | 0.910 | 0.793 |
| mean of the 4 flips | 0.1919 | -0.0344 ± 0.0040 (30 of 31) | 0.0207 | 0.0181 | 56.6 | 0.808 | 0.746 |
| mean of all 8 symmetries | 0.1890 | -0.0374 ± 0.0038 (30 of 31) | 0.0203 | 0.0175 | 54.5 | 0.775 | 0.716 |
| VAE round trip (the floor) | 0.0785 | | 0.0074 | 0.0069 | 53.4 | 1.018 | 0.943 |

- **The model's error depends on the orientation of the frame.** The 4 flips beat 4 seeds by 0.0167 ± 0.0030 (24
  of 31 sims). Two predictions of one sim differ by 0.137 (RMS / std of the truth, all frames) between seeds and
  by 0.203 between orientations. In squared all-frames error, 4 seeds remove 15%, the 4 flips 28%, all 8
  symmetries 31%.
- **One sample from a transformed frame 0 is slightly worse** than from the untransformed one: +0.012 ± 0.003 on
  average (x flipped +0.001, y flipped +0.014, x and y swapped +0.014 to +0.019). The model has only seen the
  stored orientation.
- **The mean is blurrier**: 99th-percentile gradient 0.78 of the truth's against 0.97. In the picture of sim 0456
  the dotted texture of one sample in smooth regions is gone and shocks are closer to the truth, but the small
  roll-up vortices at late frames are smeared. MLW improves all the same, to 1.2 above the VAE round trip.
- **All 8 against the 4 flips**: -0.0029 ± 0.0027 on VRMSE (within noise), MLW 54.5 against 56.6.
- **Cost**: 8 rollouts per trajectory, about 26 s at 4 steps (one 64-step sample: 47 s).

**Fewer steps** (one sample each, same 31 sims): all worse than 4 steps with 4 chunks at once (0.2264).

| steps per chunk | chunks at once | VRMSE | against 4 steps, 4 chunks (paired) | gradient p99 |
|---|---|---|---|---|
| 1 | 1 | 0.2851 | +0.0587 ± 0.0071 | 0.883 |
| 2 | 1 | 0.2336 | +0.0072 ± 0.0023 | 0.954 |
| 2 | 2 | 0.2338 | +0.0074 ± 0.0025 | 0.952 |
| 4 | 1 | 0.2311 | +0.0048 ± 0.0021 | 0.972 |
| 4 | 2 | 0.2294 | +0.0030 ± 0.0014 | 0.971 |

**Train, val and test sims.** 60 sims of the train split and 60 of the val split (never trained on), evenly over
the gammas, and the 31 test sims; the 8000-step model against the +10,000 model on the same sims. The groups
differ in difficulty (the val sims are easier, also for the VAE), so the change per sim is what compares.

| group | sims | VAE round trip | 8000-step model | +10,000 model | change per sim | relative change per sim |
|---|---|---|---|---|---|---|
| train | 60 | 0.0785 | 0.2539 | 0.1888 | -0.0651 ± 0.0053 (59 of 60 better) | -24.7% ± 1.4% |
| val | 60 | 0.0712 | 0.2230 | 0.1984 | -0.0247 ± 0.0038 (54 of 60) | -10.6% ± 1.2% |
| test | 31 | 0.0785 | 0.2690 | 0.2264 | -0.0426 ± 0.0100 (28 of 31) | -13.7% ± 2.1% |
| test, 64 steps | 500 | | 0.2864 | 0.2464 | -0.0400 ± 0.0015 | -14.1% ± 0.3% |

- **The DiT has started to memorize its training sims**: the change on train sims minus the change on val sims
  is -0.040 ± 0.007. With the 8000-step model the train sims were not ahead (0.254 against 0.223 and 0.269).
- **Sims it has not seen still improve**, by 11 to 14%, about half the rate of the training sims.
- **Consequences.** Rollouts of training sims are now roughly 12 to 16% more accurate than they would be for unseen sims, so a
  decoder trained on rollouts has to use sims the DiT did not train on (the 450 val sims). More passes over the
  same 4,050 sims will pay less; flips and swaps give 8 valid versions of every sim, and the symmetry ensemble
  shows the model is far from treating them alike.

## Training speed options (full FT)

At 18 s per optimizer step (32 samples, 4 GPUs) the DiT ran at roughly 12 % of bf16 peak. Attention is already
FlashAttention through SDPA (`compat.py`). All four options are off by default: runs before 2026-11-10 behave
exactly as before.

| flag | what (code) | expected | cost |
|---|---|---|---|
| `--tf32` | MAGI runs each layer's attention output projection in fp32 on purpose (`dit_module.attn_linear_proj`, about 16 % of the layer FLOPs). TF32 was never enabled, so it ran as true fp32. This flag uses TF32 tensor cores (`dit.set_fp32_matmul`). | 1.5× measured | TF32 keeps 10 mantissa bits (more than bf16), but results are not bit-identical to older runs |
| `--checkpoint-layers N` | recompute only the first N of 34 layers in backward (`dit.enable_gradient_checkpointing`) | 17 layers: about 10 % on top of compile | 1.4 GB more per layer left out (compiled) |
| `--compile` | `torch.compile` of the attention-free parts of every layer (q/k/v projections, norms, rotary; output projection, gating, MLP), one graph shared by all 34 layers (`dit.compile_layers`) | 1.3× on top of `--tf32` | 33 s compile time at the start; training only |
| `--latents DIR` | read pre-computed latents instead of encoding every sample with the VAE (`data.CachedLatents`); `magi1_latents_F3/` on SCRATCH holds all 30 latents per sim, bit-exact | about 5 % (encoding takes 0.12 s of about 2.25 s per sample) | F3 VAE and `--frames 100` only; the data order is unchanged |

- **Design notes for `--compile`.**
  - The attention calls stay eager: they read numpy ranges and `cu_seqlens` lists, which would graph-break or
    freeze data into the graph.
  - The layer is passed as an argument, so dynamo reuses one graph for all layers instead of recompiling per
    layer. It would otherwise fall back to eager after `recompile_limit` (8) recompiles; the limit is also
    raised to 64 as a safety net.
  - `dynamic=False`, because training shapes are fixed by `--frames`.
  - Not used in evaluation, because the sampler's shapes change per window stage.
- **Measured** (`tests/test_speed.py`, job 2812042, one GPU, one sample at the training shape):

  | variant | s per sample | speedup | peak GB in the test (GPU: 102) |
  |---|---|---|---|
  | before (fp32 projection, eager, recompute all 34) | 2.04 | 1.00× | 54.3 |
  | `--tf32` | 1.33 | 1.53× | 54.3 |
  | `--tf32 --compile` | 0.99 | 2.05× | 48.3 |
  | `--tf32 --compile --checkpoint-layers 17` | 0.89 | 2.30× | 71.6 |
  | `--tf32 --compile --checkpoint-layers 8` | 0.81 | 2.52× | 84.7 |
  | `--tf32 --compile --checkpoint-layers 0` | 0.74 | 2.74× | 96.3 |

  - **Memory.** Real 4-GPU training reports a 73 GB peak for the first row, 19 GB above the test, so add
    about that to the last column. `--checkpoint-layers 17` should fit; 8 and 0 are too close to the limit.
  - **Latent cache.** Stored latents are bit-identical to the ones encoded on the fly (4 sims); the encoding
    it saves is 0.120 s per sample.
  - **Compile.** 4 graphs shared by all 34 layers, no graph breaks, 33 s to compile.
  - **Gradients are not bit-identical.** The loss agrees to 0.02 % (`--tf32`) and 0.01 % (`--compile`), but
    the gradient differs by 6.6 % and 5.9 % in relative L2 norm (cosine 0.998), against 0.7 % between two
    identical runs. Both are above the 5 % limit the test was written with, so it reports 5/7. `--tf32`
    only changes the precision of one matmul per layer and gives the same size of difference as `--compile`,
    which points to bf16 rounding and not to a wrong graph (that would change the loss). Partial
    checkpointing stays at the noise floor (0.7 %).
  - **Not checked yet:** 4 GPUs with DDP, and the loss curve of a real run against an old one.

  ```bash
  sbatch jobs/jupiter/magi1/quick_test.sbatch -m magi_finetuning.tests.test_speed
  ```
- **Using them**, once the check passes:

  ```bash
  bash jobs/jupiter/magi1/magi1_dit_experiment.sh --name <run> --vae-ckpt <vae> --tf32 --compile --checkpoint-layers <N> \
      --latents $MAGI1_SCRATCH/magi1_latents_F3 ...
  ```

## Compute budget

The budget (`e-dev-2026d09-262`) has to last until about May 2027.
- Check what is left with `jutil user cpuquota -p e-dev-2026d09-262`.
- Billing is per whole node (288 core-h per node-hour), even for 1-GPU jobs, so pack small jobs 4 per node.
- Reference costs:

  | job | core-h |
  |---|---|
  | 2000-step run | about 2,900 |
  | 8000-step run | about 11,500 |
  | 500-sim eval, 64 steps | about 530 |
  | 500-sim eval, 4 steps | about 75 |
