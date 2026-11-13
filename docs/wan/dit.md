# Wan DiT stage: results

Test VRMSE (mean over fields, physical units) on 500 test sims; `run` is the folder under `results/wan/eulermq/dit/`.
**v2** = protocol v2: scored from frame 10, per-sim noise seeds, cfd_bench beside it (`bench`), paired CIs between
runs. **v1** = the older scoring from frame 0 with one shared seed: about 0.02 lower than v2 for the same model,
comparable only within its own table. Sampling settings: [dit_sampling.md](dit_sampling.md).

## Zero-shot and rgb3 (res256)

Pretrained Wan 2.1 without fine-tuning: `zero_shot/*` (videos only, no metrics). Li's recipe on rgb3 latents
(Li: 0.498), v1, 20 steps:

| run | what | VRMSE |
|---|---|---|
| res256/rgb3/baseline | lr 1e-5, 8000 updates | 0.586 |
| res256/rgb3/baseline/stage2_anneal | + 2000 updates cosine tail | 0.437 |
| res256/rgb3/lr_ablation/lr3e-5 | 2000 updates at 3e-5 | 0.641 |
| res256/rgb3/lr_ablation/lr1e-4 | 2000 updates at 1e-4 | 0.564 |
| res256/rgb3/lr_ablation/lr3e-4 | 2000 updates at 3e-4 | 0.578 |
| res256/rgb3/lr_ablation/lr1e-3 | 2000 updates at 1e-3 | 0.868 |

## Pairs, stride-1 era (res256, v1)

| run | what | VRMSE |
|---|---|---|
| res256/pairs/stride1_era/baseline/stage2_streamlr | widened patchify at 1e-4 | 0.553 |
| res256/pairs/stride1_era/baseline/stage3_anneal | cosine tail | 0.432 |
| res256/pairs/stride1_era/baseline/stage4_lr1e-4 | 4000 more at 1e-4 | 0.390 |
| res256/pairs/stride1_era/baseline/stage4_ema | stage 4 with EMA 0.998 | 0.388 |
| res256/pairs/stride1_era/baseline/stage4_adaln | gamma through adaLN | 0.394 |

Sampler ablation on stage4_lr1e-4 (`res256/pairs/stride1_era/sampling/*`): steps 1 / 2 / 5 / 10 / 20 / 50 =
0.383 / **0.363** / 0.382 / 0.388 / 0.390 / 0.390; time shift 1 / 1.5 / 2 / 5 at 20 steps = 0.389 / 0.389 / 0.391 /
0.389.

## Pairs recipe (res256)

| run | what | v1, 5 steps | v2, 2 steps | bench |
|---|---|---|---|---|
| res256/pairs/recipe/stage1_stride1 | stage 1, frame stride 1 | 0.325 | – | – |
| res256/pairs/recipe/stage1 | stage 1, frame stride 2 | 0.307 | – | – |
| res256/pairs/recipe/stage2_lr3e-5 | stage 2 | 0.281 | – | – |
| res256/pairs/recipe/stage3_d4 | stage 3, D4 augmentation | 0.247 | 0.254 | 0.234 |

stage3_d4: 5 steps v2 0.280 (`v2_5s`); v1 ensemble of 4 / 8 samples 0.220 / 0.218 (`ens4`, `ens8`).

## Stage-4 arms (res256, from stage3_d4)

| run | what | v2, 2 steps | bench | LDE | front shift |
|---|---|---|---|---|---|
| res256/pairs/stage4/control | 4000 updates at 3e-5 | 0.270 | 0.249 | 0.362 | 7.66 px |
| res256/pairs/stage4/d4_lr1e-5 | 16000 updates at 1e-5 | **0.244** | 0.225 | 0.358 | 7.26 px |
| res256/pairs/stage4/ctx2 | two context latents | 0.248 | 0.229 | 0.369 | 7.52 px |
| res256/pairs/stage4/sigma1 | sigma draw ablation (step 3000) | 0.254 | 0.234 | 0.370 | 7.41 px |
| res256/pairs/stage4/forcing | Diffusion Forcing history | 0.285 | 0.261 | 0.380 | 7.98 px |
| res256/pairs/stage4/stride4 | frame stride 4 | 0.290 | 0.284 | 0.432 | 8.39 px |
| res256/pairs/stage4/ar7 | 7-latent autoregressive windows | 0.262 (v1) | – | – | – |

Chunked rollouts (v1): control one-shot 0.263, chunks of 4 with history noise 0 / 0.02 = 2.67 / 0.97;
forcing one-shot 0.276, chunks of 4 = 0.271 for history noise 0 to 0.1 (`*/chunk4/*`).

## Pixel and front loss (res256)

| run | what | v2, 2 steps | bench | LDE | front shift |
|---|---|---|---|---|---|
| res256/pairs/pixel_loss/pixel | stage4/control + L1+GDL on decoded x0, 1/5 of the loss | 0.265 | 0.247 | 0.360 | 7.66 px |
| res256/pairs/pixel_loss/pixel_2x | same, 2/5 of the loss | 0.263 | 0.246 | 0.361 | 7.64 px |
| res256/pairs/pixel_loss/v2_pixel | from stage3_d4, 6-latent windows | 0.250 | 0.233 | 0.377 | 7.52 px |
| res256/pairs/pixel_loss/v2_front | v2_pixel + front-distance term | 0.253 | 0.235 | 0.374 | 7.60 px |

Paired: v2_pixel − stage3_d4/v2 = −0.0034*, v2_front − v2_pixel = +0.0024*. Notes:
[notes_pixel_loss.md](notes_pixel_loss.md).

## Unrolled training (res256, from stage4/d4_lr1e-5)

| run | what | v2, 2 steps | 5 steps | bench | LDE | front shift |
|---|---|---|---|---|---|---|
| res256/pairs/unrolled/control | 4000 plain updates at 1e-5 | 0.247 | 0.270 | 0.229 | 0.356 | 7.31 px |
| res256/pairs/unrolled/unrolled | half of each batch on its own sampler state (chain of 2–5 steps) | 0.232 | 0.234 | 0.214 | 0.389 | 7.05 px |
| res256/pairs/unrolled/unrolled2 | unrolled + 6000 updates at 3e-6 with its AdamW state | **0.231** | 0.228 (η 1) | 0.213 | 0.380 | 7.03 px |
| res256/pairs/unrolled/delta | from stage4/d4_lr1e-5, 8000 unrolled updates on latent deltas | 0.282 | – | 0.254 | 0.440 | 8.63 px |

Paired: unrolled − control = −0.015* (2 steps), −0.036* (5 steps); delta − unrolled = +0.050*.

## Continuations of unrolled2 (res256, v2, 2 steps)

Each run starts from unrolled2 and keeps its unrolled recipe; Δ is the paired difference to `cont` over the 500
test sims, matched by sim id, with a 95 % CI.

| run | what changed | updates, lr | VRMSE | Δ vs cont |
|---|---|---|---|---|
| res256/pairs/unrolled/unrolled2/cont | nothing: the control | 4000, 1e-5 | 0.231 | – |
| res256/pairs/unrolled/unrolled2/detail5 | flow loss per latent cell weighted 1 + 5 × its share of front (Löhner > 0.2 on density, pressure) or vortex (Q > 1) pixels | 4000, 1e-5 | 0.232 | +0.0002 [−0.0003, +0.0007] |
| res256/pairs/unrolled/unrolled2/late_hard | later latent frames weighted up (late_weight 1), train sims drawn with weight 0.5 + 0.5 × error/mean from unrolled2's own train rollouts | 4000, 1e-5 | 0.233 | +0.0012 [+0.0004, +0.0020] |
| res256/pairs/unrolled/unrolled2/fixed_noise_eval | unrolled2 sampled from one noise draw shared by all sims (η 0), no training | 0 | 0.230 | −0.0010 [−0.0022, +0.0002] |
| res256/pairs/unrolled/unrolled2/fixed_noise | one noise draw shared in training and sampling: a regression model | 4000, 1e-5 | 0.229 | −0.0019 [−0.0028, −0.0011] |
| res256/pairs/unrolled/unrolled2/learned_noise | as fixed_noise, the shared draw trained with the model at lr 1e-3 | 4000, 1e-5 | 0.228 | −0.0036 [−0.0046, −0.0026] |

`cont` (0.231) does not improve on unrolled2 (0.231).
fixed_noise_eval − unrolled2 = −0.0004 [−0.0015, +0.0007]: a shared draw alone changes nothing.
learned_noise − fixed_noise = −0.0017 [−0.0024, −0.0009].
Learning the shared noise gives the only gain, 0.004 below `cont`.
`late_hard` reads its sim weights from `temp/sim_weights_unrolled2.json`, which is gitignored.
`detail5` reads its masks from `/scratch/project_465003416/lc_work/detail_masks/lohner0.2_q1`, written by
`scripts/wan/detail_masks.py`.

## Pruned students (res256, from unrolled/unrolled2)

At most half the teacher's 1419 M parameters, cut and distilled: [dit_pruning.md](dit_pruning.md).

| run | what | params | v2, 2 steps | bench | LDE | front shift |
|---|---|---|---|---|---|---|
| res256/pairs/pruned/inner_c2_f3776 | cross heads 12 → 2, FFN 8960 → 3776, 1500 updates with teacher KD | 705 M | 0.253 | 0.231 | 0.445 | 7.67 px |
| res256/pairs/pruned/inner_c2_f3776/long | + 6000 updates at 1e-5 | 705 M | 0.243 | 0.223 | 0.421 | 7.44 px |

Paired: inner_c2_f3776 − unrolled2 = +0.022*, inner_c2_f3776/long − unrolled2 = +0.012*. Depth (14 blocks) and stream (760) students were stopped as clearly
behind.

## 512 and singles

| run | what | v1, 5 steps | v1, 2 steps | bench |
|---|---|---|---|---|
| res512/pairs/stage4 | stage3_d4 continued at 512 | 0.241 | 0.225 | – |
| res512/pairs/stage5_d4 | + D4 latents (stopped at 3000, no test) | – | – | – |
| res512/pairs/stage6_d4 | stage5_d4 step 3000 + 2000 | 0.241 | 0.229 | 0.232 |
| res256/singles/stage1_d4 | one field per image, four latent streams | 0.264 | 0.238 | 0.374 |

res512/pairs/stage6_d4 as mean of 4 samples (v1, 2 steps): 0.211 (`ens4`).

## Corrector on stage3_d4 rollouts (v1)

| run | what | VRMSE |
|---|---|---|
| corrector/control | per-frame U-Net on Wan frames t−1, t, t+1 | 0.232 |
| corrector/ar | + its own previous frame | 0.232 |
| corrector/ar_w16_h4 | 4 own past frames, 16-frame windows | 0.232 |
| corrector/ar_w16_h4_f0 | + true frame 0 | 0.232 |
| corrector/final_tf | teacher-forced stage of final_ar (no test) | – |
| corrector/final_ar | AR training from final_tf | **0.224** |

Wan alone on these rollouts: 0.232.

Warp head (runs removed). The corrector predicted two extra channels, a displacement field 16 · tanh(·) in pixels,
zero at the start. Wan's frame t was warped by it (bilinear, edge values continued) before the residual was added,
so fronts could be moved instead of redrawn. Both arms continued final_ar for 3000 AR updates (lr 1e-4, 200
warmup, EMA 0.995, MSE, batch 32, history 1 plus true frame 0) on the same four stage3_d4 rollout sets and were
scored on the same 500 test sims.

| run | what | VRMSE |
|---|---|---|
| corrector/final_ar/control (removed) | final_ar + 3000 AR updates | 0.223 |
| corrector/final_ar/warp16 (removed) | same with the warp head, up to 16 px | 0.222 |

The warp head gained 0.001 VRMSE, below anything we act on, so its code was removed.

## LoLA on our 512 test set (v1, `results/lola/c80/*`)

| run | what | VRMSE |
|---|---|---|
| c80/ae | LoLA c80 autoencoder round trip: the floor of its emulators | 0.008 |
| c80/sm | its latent neural solver, deterministic, frame 0 given, stride 4 | 0.205 |
| c80/dm16 | its latent diffusion model, 16-sample ensemble mean | 0.171 |
