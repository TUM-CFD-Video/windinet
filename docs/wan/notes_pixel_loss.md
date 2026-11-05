# Density error and a decoded auxiliary loss

## 1 Problem

- Wan 2.1 T2V 1.3B DiT, fine-tuned in the frozen Wan VAE latent space as a surrogate for 2D compressible Euler multi-quadrant Riemann problems (The Well).
- Fields rho, mx, my, p mapped to RGB by a fixed adapter; two groups, (rho,p) and (mx,my), one latent each.
- Loss: rectified-flow MSE in latent space, x_sigma = (1-sigma) z + sigma eps, target eps - z, logit-normal sigma, shift 3. Sampler: UniPC 2 steps (network at sigma = 1 and 0.75).

Per-field VRMSE, best 256 model stage3_d4 (2-step, 500 test sims, 256 res, overall 0.232):

| field | DiT VRMSE |
|---|---|
| density | 0.240 |
| mx | 0.233 |
| my | 0.246 |
| pressure | 0.170 |

- VAE round-trip floor at 512: density 0.034, lowest of all fields. Pairs decoder floor at 256: 0.080 to 0.098 overall.
- Density has the lowest floor but the highest DiT error. Pressure is the easy field.

## 2 What is ruled out (measured)

| candidate | evidence | verdict |
|---|---|---|
| Metric normalisation | persistence check, 20 test sims (table below) | VRMSE does not punish density |
| Sample variance | ens4 vs single at 512: density -5.4 %, mx -5.6 %, my -5.7 %, p -2.4 %; saturates at n = 4 | error is bias, not variance |
| Shared image with pressure | singles (own latent per field): density 0.267 vs 0.249 at 256, other fields level | sharing is not the cause |
| Decoder | density floor is the lowest | not the decoder |

| field | persistence VRMSE | median trajectory variance | std/\|mean\| |
|---|---|---|---|
| rho | 1.53 | 0.133 | 0.42 |
| mx | 1.52 | 0.060 | 1.14 |
| my | 1.51 | 0.083 | 1.14 |
| p | 2.29 | 0.069 | 0.33 |

## 3 Hypothesis

- Density carries the contact discontinuities and rollups. These are advected.
- A small velocity error displaces a sharp density edge; under a pointwise metric it costs its full jump twice. Pressure is continuous across contacts and forgives the same displacement.
- The latent MSE weights smooth bulk content (density spectral error: low band 0.06, mid/high about 0.5). It has no notion of what an edge costs in pixel space.
- Lever: sharpness and placement of edges, not density per se. Thesis framing: pressure is easy because it is smooth across contacts; density pays for misplaced advected interfaces.

## 4 Experiment A: decoded-space auxiliary loss (one-step x0 estimate)

LPL-style. Steps and reasons:

| step | choice | reason |
|---|---|---|
| 1 | keep the latent flow loss | sampler needs a proper velocity field; aux term reshapes, never replaces |
| 2 | x0_hat = x_sigma - sigma * v | exact identity, zero cost; decoding x_sigma would be wrong; a second forward pass costs twice the compute |
| 3a | decode 2 of 13 latent frames | full decode = 49-frame 3D VAE fwd+bwd per sample, about the DiT's own cost; activations do not fit beside the DiT at batch 4; 2 frames keep overhead near 20 % |
| 3b | random frames, not fixed | all positions covered over 4000 updates; variant: bias to late frames |
| 3c | single-latent decode | differs slightly from chunked full decode (causal VAE: latent 0 -> 1 pixel frame, later latents -> 4); fine for a gradient, never for evaluation |
| 3d | (rho,p) group only | density is the target, pressure free in the same image; momentum group doubles cost for a field that is not the problem |
| 3e | frozen decoder, gradients through it | DiT learns; decoder must not drift from the eval decoder |
| 4 | inverse adapter, per-field std normalisation | loss in physical space, not RGB; adapter's log transform is the suspected under-weighting; std matches VRMSE |
| 5 | lambda about 0.1 | term is 10 to 20 % of the total at step 0; one value, no sweep |
| 6 | mask aux term at sigma > 0.8 | see below |
| 7 | run | stage3_d4 + 4000 updates at lr 3e-5, plain latents, partner stage4/control (0.246 at 2 steps); about +20 % per update, about 5 h at 256 |

Why the sigma mask:

- At sigma near 1, x0_hat is the conditional mean, nearly the dataset mean. Decoded: a smeared field with no shock positions. Pixel loss is large on every sample regardless of velocity quality; gradients contradict across samples and average to noise.
- Below 0.8, x0_hat has interfaces at roughly the right place. The gradient says "sharpen, move two pixels".
- 0.8 keeps the sampler's sigma 0.75 evaluation inside the band. The sigma = 1 evaluation is pure prior and cannot be helped.
- With shift 3: about 61 % of draws fall below 0.8, only 14 % below 0.5. Alternative: soft weight (1-sigma)^2.

Central weakness: no clean latent.

- x0_hat at any sigma is a conditional mean. Decoded, it is blurry where the trajectory is ambiguous. The loss cannot reach zero. The gradient pulls to the same averaged target the latent MSE already optimises. A is a re-weighting through the decoder metric, not a sharpness objective.
- Less bad here than for image generation: first frame given, PDE deterministic, ensemble gain only 5 %, so samples are close to their mean.
- Real danger: the frozen decoder is off-manifold at x0_hat from sigma 0.7. Its artefacts can be "fixed" by moving the latent in non-physical directions (decoder hacking).
- Mitigations: tighter mask (sigma < 0.5; the aux subset then needs its own sigma draw), small weight, watch latent loss and 5-step eval.

## 5 Experiment B: loss on the generated 2-step sample

- Run the 2-step sampler inside training on a fraction of the batch. Decode 2 random frames of the sample; decoded H1/gradient loss against ground truth.
- Backprop through both sampler steps. With K = 2 the exploding-gradient argument of 50-step chains disappears.
- Trains exactly the operating point; merges A with C. Latent flow loss stays at weight 1 on the whole batch. Decoded term 10 to 20 % of the total.

| item | value |
|---|---|
| cost per update | about 2.5x plain (2 DiT forwards, 1 to 2 backwards, decoder fwd/bwd on 2 to 4 frames) |
| 4000 updates | about 10 h at 256 |
| alternative | 2000 updates, compared at equal wall clock and at equal updates |

- Risks: pulls the model towards a deterministic 2-step regressor; 5-step sampler and ensemble spread degrade; collapse if the base loss is dropped.
- Checks: evaluate 5 steps and ens4 too. If those degrade while 2-step improves: restrict the decoded term to the last 2 to 4 transformer blocks or a LoRA (PIRF/DRaFT recipe); shorter EMA.

## 6 Experiment C: train at the sampler sigmas

- 70 % of sigma draws set to 1.0 or 0.75 (half each), before frame_sigmas, so context masking is unchanged.
- 30 % keep the logit-normal draw. Nothing else changes.
- Motivation: more steps are worse (5-step 0.251 vs 2-step 0.232), so the velocity field is inconsistent across sigma. Make the two used evaluations better trained.
- Cost: zero extra per update, about 4 h. Risk: forgets the rest of the schedule, 5-step gets worse. Acceptable, but evaluate 5 steps for the thesis.

## 7 Literature

From an Opus research agent; arXiv HTML read with WebFetch unless marked otherwise.

Decoded / perceptual losses on diffusion models:

| paper | setup | result |
|---|---|---|
| LPL, Berrada et al., ICLR 2025 | x0_hat = z_t - sigma_t v; MSE on frozen-decoder features of x0_hat vs z0, per-channel standardised, layer weights 2^(-r_l/r_1), outlier mask; weight about 3 (about 1/5 of total); hard mask sigma_t <= tau, best tau 3 to 6; post-training only (600k pretrain, then 200k at 256 / 120k at 512) | FID ImageNet-256 2.98 -> 2.72, ImageNet-512 4.88 -> 3.79, CC12M-256 7.81 -> 6.22, CC12M-512 8.79 -> 7.27, S320M 8.81 -> 8.30, Flow-OT ImageNet-512 4.54 -> 3.61, equal wall-clock 4.62 -> 3.84; gains at f > 150 |
| Zhang et al. 2024 | L_latent + lambda L_pixel, L2 between decoded noisy latents with true vs predicted noise; lambda 8 (DiT), 2 (Emu); no sigma mask | human eval only: DiT SFT visual-flaw win rate 17.7 % -> 64.2 % |
| PixelGen | pixel diffusion, LPIPS 0.1 + P-DINO 0.01 on x-prediction, off for noisiest 30 % of t | FID 23.67 base, +LPIPS 10.00, +P-DINO 7.46, gate 0.3 7.53 (recall 0.58 -> 0.60), gate 0.6 10.72 |
| Cas-DM | LPIPS on x0 of a DDPM | ImageNet64 FID 65.1 -> 41.4; naive version worse on CelebA-HQ (+1.7), LSUN Bedroom (+2.9) |
| Lee, Lin, Fanti, NeurIPS 2024 | LPIPS-Huber premetric, 1 to 2 step rectified flow, pixel space | about 72 to 75 % 1-NFE FID gain on CIFAR-10 (table not verified) |

- LPL negatives: plain RGB pixel MSE on the decoded output was worse than baseline (no number), large memory; LatentLPIPS/self-perceptual worse (FID 5.8 to 13.4 vs 4.88); early decoder blocks hurt, blocks 3 to 4 carry the gain; min-SNR alone 5.04 (worse), so not a reweighting effect; needed shorter EMA.
- LPL cost: memory 42.5 % -> 74.3 %, throughput 16.6 -> 9.6 img/s, about 1.5x FLOPs. FID rises again at high weight. Tau: the agent infers noise-to-signal, about 0.75 to 0.86 in flow time (not stated by the authors).
- Zhang: decoding predicted x0 at large t gave blurrier images (high variance), an argument for masking high sigma. Pixel-only loss: sharp subjects, over-blurred backgrounds; adding the latent loss fixed it.
- PixelGen: the gate buys recall; gating too much hurts. Cas-DM: metric gradient through the shared backbone disturbs the eps prediction. Fix: separate x0 refiner with stop-gradient.
- REPA-E: backprop of the diffusion loss into the VAE collapses the latent; not our case (decoder frozen). LADD: moved the discriminator to latent features to avoid decoder backprop cost (from summaries only).

PDE surrogates:

| paper | setup | result |
|---|---|---|
| LOLA, Rozet et al. (same dataset) | latent-only, AE with variance-normalised MSE, no decoded loss tested | Euler VRMSE lead 1-20 / 21-60 / 61-100: AE x80 0.011 / 0.014 / 0.020; diffusion x80 0.075 / 0.199 / 0.395; diffusion x320 0.070 / 0.192 / 0.371; pixel neural solver 0.138 / 0.397 / 1.102 |
| WinDiNet, Perini et al. (LTX baseline) | physics loss (divergence, wall, lambda 10) in decoder FT only, not the DiT | round-trip 0.188 -> 0.084 (decoder FT) -> 0.070 (physics); full pipeline 0.531 -> 0.520 |
| LE-PDE, Wu et al., NeurIPS 2022 | decoded multi-step loss + latent consistency loss | 1D / 2D NS error: full 1.127 / 0.186; no decoded loss 3.337 / 0.216; no consistency 6.386 / 0.232 |
| PIDM, Bastek et al., ICLR 2025 | pixel space, PDE residual on E[x0\|x_t], weight 1/(2 Sigma_t), no hard mask, min-SNR-5 on data term | Darcy residual about two orders lower; topology optimisation 33 to 35 % lower; cost +23 % (mean) or +69 % (2-step sample) |
| LDMPDE | PDE residual on decoded latent, fine-tune last 200M params | residuals already low; RMSE slightly worse; no ablation |
| Physics-Informed Distillation | residual loss | diffusion loss up, no PDE gain (snippet only, unverified) |
| Kastor (incl. Euler multi-quadrants) | gradient-difference loss (H1-like) | +13.3 % fCRPS on one dataset; diffusion baseline at 5 steps since more steps degraded CRPS |
| Latent Generative Solvers | latent-only loss | none |

- LOLA: emulator error barely depends on compression; high-frequency spectral error is limited by the AE. Bottleneck is the dynamics. A decoded loss helps only if the DiT misplaces discontinuities in a way the latent MSE under-weights.
- LE-PDE: both terms needed. PIDM notes the Jensen gap R(E[x0|x_t]) != E[R].
- No paper found that adds a decoded physical-space loss to a frozen-VAE latent diffusion PDE surrogate and reports a gain.
- Masking: hard in LPL and PixelGen, soft in PIDM; min-SNR and Kingma & Gao standard (from memory, not re-read). Frame subsampling for decoding: VADER. Memory tricks: LoRA, fp16 frozen weights, gradient checkpointing, CPU offload (VADER, DRaFT, AlignProp).

Backprop through the sampler:

- DRaFT (SD 1.4, 50 DDIM steps, LoRA rank 8/32, checkpointing): K in {1,5,10,30,50}, smaller K better; gradients explode with K, for K > 30 nearly orthogonal to K = 1. HPSv2: base 0.2695, DRaFT-50 0.3033, DRaFT-1 0.3467, DRaFT-5 0.3500, DRaFT-LV 0.3536 (K = 1, re-noise n = 2, about 2x more efficient for about 10 % more compute). Reward hacking: collapse to near-identical images; mitigation LoRA scale below 1.
- VADER (video): K = 1, random frame subsampling before decoding, LoRA, checkpointing. Aesthetic 4.61 -> 7.31 train, 4.49 -> 7.12 test; no reward hacking observed. AlignProp (withdrawn, superseded by VADER): randomized truncation, K range unverified.
- ReFL / ImageReward: no-grad sampling to a random late step, one step with gradient as x0'. Final-step-only "very unstable". Reward weight 1e-3, regularised with the pretraining loss. Win rate 58.8 % vs SD 1.4.
- PIRF (physics reward): last 1 or 2 steps; plain truncation still reward-hacked (artificial voids in Darcy). Fix: update only high-resolution layers, EMA/early-stop. Darcy residual 1.29 vs 4.56 CoCoGen, Kolmogorov 19.28 vs 76.66.
- List et al., unrolled PDE surrogates: fully differentiable unrolling best, non-differentiable close, gradients drift with horizon. Consistency models: LPIPS over L2, iCT Pseudo-Huber (from memory).

Agent synthesis:

- Moderate expectation, not firm. Only clean positive evidence for a decoder-space term on a frozen-VAE latent model is LPL (feature space, high-frequency FID). Plain decoded RGB MSE: worse (LPL), blur-prone (Zhang). No PDE result for this recipe. LOLA says emulator error dominates.
- Common settings: post-training phase, weight about 20 % of total, hard sigma mask on the noisiest region, weight tuned by magnitude ratio.
- Pitfalls: MSE optimum is still the posterior mean; shared-backbone disturbance (Cas-DM); 1.5 to 2x cost; features or gradients helped where pixel MSE did not (favours H1); shorter EMA.
- For a near-deterministic PDE with a 2-step sampler the literature leans to B plus latent flow loss plus an H1/gradient term. A is the cheaper, better-documented fallback.
- Unverified: AlignProp K range, Text2PDE/ACDM losses, consistency-model LPIPS numbers, LPL's exact sigma convention.

## 8 Plan and expectation

Claude's assessment, not measured.

Order:

1. Edge decomposition on existing predictions, CPU on the login node. Split density VRMSE into pixels near strong gradients vs smooth regions; recompute under a 2-pixel best-shift tolerance and a 4x4 blur. If the error sits at edges and collapses under tolerance, placement is the issue and B is the right tool.
2. Run B (generated sample, H1 on decoded rho and p, latent loss kept, 2 random frames, weight 10 to 20 %) from stage3_d4 against stage4/control. Evaluate 2 steps, 5 steps, ens4.
3. C in parallel on a second node if budget allows. A as fallback. Budget: about 10 node hours for B, 4 for C.

| outcome | probability | reading |
|---|---|---|
| modest gain | about 50 % | density -2 to -5 %, overall 0.246 -> about 0.240 vs stage4/control; 500 sims resolve about 0.003 |
| no gain or slight loss | about 35 % | latent MSE already tracks decoded error; residual error is late-trajectory placement of chaotic rollups; ensemble bias result and LOLA ratio point here |
| clear gain above 5 % | about 15 % | only if the latent objective under-weights thin density structures; the decomposition would show it |

- First change of the week to the objective rather than data layout or sampler; all of those were neutral or negative.
- Biggest untested lever remains a latent space trained on the data (LOLA style), so the latent MSE sees the edges.

## 9 References

- Decoded / perceptual losses: LPL, latent perceptual loss, arxiv.org/abs/2411.04873; Zhang et al., pixel-space post-training, arxiv.org/abs/2409.17565; PixelGen, arxiv.org/abs/2602.02493
- Cas-DM, cascaded diffusion with x0 refiner, arxiv.org/abs/2401.02414; Lee, Lin, Fanti, improved rectified flow training, arxiv.org/abs/2405.20320
- VAE and latents: REPA-E, end-to-end VAE and diffusion, arxiv.org/abs/2504.10483; LADD, latent adversarial diffusion distillation, arxiv.org/abs/2403.12015
- PDE latent models: LOLA, latent emulation of PDEs, arxiv.org/abs/2507.02608; WinDiNet (LTX baseline), arxiv.org/abs/2603.21210; LE-PDE, arxiv.org/abs/2206.07681; Latent Generative Solvers, arxiv.org/abs/2602.11229
- Physics losses: PIDM, physics-informed diffusion models, arxiv.org/abs/2403.14404; LDMPDE, arxiv.org/abs/2510.23866; Physics-Informed Distillation, arxiv.org/abs/2505.22391; Kastor, arxiv.org/abs/2608.06107
- Weighting: min-SNR, arxiv.org/abs/2303.09556; Kingma & Gao, weighted ELBO, arxiv.org/abs/2303.00848
- Sampler backprop: DRaFT, arxiv.org/abs/2309.17400; AlignProp (withdrawn), arxiv.org/abs/2310.03739; VADER, video reward gradients, arxiv.org/abs/2407.08737; ReFL / ImageReward, arxiv.org/abs/2304.05977
- PIRF, physics-informed reward fine-tuning, arxiv.org/abs/2509.20570; List et al., unrolled training of PDE surrogates, arxiv.org/abs/2402.12971
