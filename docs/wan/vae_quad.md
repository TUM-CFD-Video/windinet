# Quad VAE: four fields in one widened Wan 2.1 VAE pass

The four fields (ρ, m_x, m_y, p) enter one 4-channel image; `widen_io` adds a zero-initialised fourth input slot to
`encoder.conv_in` and a fourth output row to `decoder.conv_out`. One latent of 16 channels per trajectory, against two
for pairs and four for singles. Protocol of the quad rows in [vae_baselines.md](vae_baselines.md): test VRMSE on the
first 100 test simulations, whole trajectories at frame stride 2; `run` is the folder under `results/wan/vae/`.

## How Wan trained its VAE

Wan report (arXiv 2503.20314, sec. 4.1) and the diffusers `AutoencoderKLWan` code.

| item | Wan 2.1 |
|---|---|
| losses | L1 3, KL 3e-6, LPIPS 3; 3D-discriminator GAN only in the final stage (weight not stated) |
| stages | 2D image VAE → inflated to 3D causal → 128², 5 frames → mixed resolutions and frame counts with GAN |
| lr, batch, steps | not stated |
| causality | chunks of 1 + 4 + 4 … frames, 2 cached frames per causal conv, RMSNorm instead of GroupNorm |
| size | ≈ 127 M parameters, 4 × 8 × 8 compression, 16 channels |
| latent normalisation | not in the model: the pipeline uses (z − latents_mean) / latents_std per channel (ours: `WanVAE.encode`) |
| output | `decode` clamps to [−1, 1] |

Wan trains most of its VAE on 5-frame clips and only the final stage on long ones.
KL at 3e-6 next to L1 at 3 is a weak regulariser; Lola (arXiv 2507.02608) drops it and LPIPS/GAN for physics fields.

## How others widen a pretrained VAE or U-Net

| work | change | init of the new weights | training |
|---|---|---|---|
| SD inpainting | U-Net conv_in 4 → 9 | zero | 440 k steps |
| Marigold (arXiv 2312.02145) | U-Net conv_in doubled | copy and halve | 18 k steps, lr 3e-5, VAE frozen |
| CC-Pan (arXiv 2602.04473) | SD-VAE to one band | luminance projection of RGB in and out convs | latent scale re-estimated |
| Wan-Alpha (arXiv 2509.24979) | RGBA video | alpha replicated to RGB through the frozen encoder, learned merge | separate decoders |
| climate downscaling (arXiv 2605.11531) | many variables | each variable replicated to RGB, VAE frozen | none |
| Lola (arXiv 2507.02608), Euler multi-quadrants | own AE, no pretrained VAE | – | MSE only, VRMSE 0.011 at 80× compression |

Most work avoids putting independent fields into one pass; it widens only the first and last convolutions; none
reports an io-only warm-up stage.

## Diagnosis of res256/quad/full_io1e-4 (0.186)

| check | evidence | verdict |
|---|---|---|
| capacity | quad compresses 4 fields 64× (pairs 32×, singles 16×); Li's LTX with 4 fields at 256× reaches 0.080 (his space) after 20 epochs | not the limit |
| adapter stage missing | pairs and singles trained their mix 2000 steps on the frozen VAE first; quad started from the identity mix with the adapter fixed | main gap at the start |
| colour coupling | with the identity mix the frozen VAE gives val ρ 0.167, m_x 0.171, m_y 0.230 (singles grayscale: ρ 0.029); independent fields in R, G, B use the low-fidelity chroma path | needs encoder training |
| pressure via the zero slot | val p 1.40 → 0.29 over 4000 updates, noisy (0.30–1.46); other fields 0.11–0.13 | slow and unstable |
| too few updates | val still falling at 4000 (0.180 → 0.165 over the last 600) | yes |
| cost | 12 s per update, 33-frame clips, fp32, every block recomputed | limits the schedule |
| decoder clamp | output clamped to [−1, 1], the adapter's tanh stays inside | no issue |

Field correlations (z-scored, train): ρ–p 0.54, m_x–m_y 0.33, others below 0.13; the smallest principal component
holds 9 % of the variance, so four nearly independent fields share one 16-channel latent.

## Training speed

Seconds per update after MIOpen tuning, one node (8 GCDs), encoder + decoder unless noted; dev-g smokes 23495740,
23495898, 23496207 (10 updates each).

| setting | frames per GCD | s / update | ms per frame | peak GB |
|---|---|---|---|---|
| 33-frame clip, fp32, checkpointing (the 0.186 run) | 33 | 10.4 | 315 | 23 |
| same, bf16 autocast, checkpointing | 33 | 8.7 | 264 | 18 |
| same, bf16, no checkpointing | 33 | 7.3 | 221 | 53 |
| 4 × 9-frame clips, fp32, checkpointing | 36 | 11.7 | 326 | 31 |
| io convs only, 4 × 5-frame clips, bf16, no checkpointing | 20 | 2.75 | 138 | 32 |

- bf16 with every layer in bf16 diverged within 10 updates (test VRMSE 6.6 vs 0.47).
- Rounding of the decoder output near ±1 is amplified by the adapter's atanh; with the two io convolutions kept in fp32, bf16 trains like fp32 (loss 0.477 vs 0.485).
- Short clips do not make a frame cheaper (326 vs 315 ms); they buy more updates for the same frames.
- New clip shapes cost about 8 minutes of one-time MIOpen tuning.

## Results

Test VRMSE, 100 test sims, whole trajectories at stride 2. Latents: mean over channels of |mean| and of std of the
normalised latents on 20 test sims (N(0, 1) = 0 and 1), and the relative change from the run's starting VAE.

| run | what changed | VRMSE | ρ | m_x | m_y | p | latent \|mean\| / std | drift | node-h |
|---|---|---|---|---|---|---|---|---|---|
| res256/quad/full_io1e-4 | old: 33-frame clips, fp32, 4000 updates | 0.186 | 0.171 | 0.155 | 0.168 | 0.250 | – | – | 13.5 |
| res256/quad/io | io convs + learned mix (3e-3) on the frozen body, 5-frame clips, 2000 updates | 0.375 | 0.481 | 0.226 | 0.278 | 0.515 | 0.62 / 0.50 (start 0.79 / 0.56) | 0.29 | 1.6 |
| res256/quad/full9 | from the widened VAE, 9-frame clips, 16 per update, 8000 updates, bf16 (3.97 s/update) | 0.179 | 0.153 | 0.175 | 0.163 | 0.223 | 0.41 / 0.97 (start 0.47 / 0.97) | 0.47 | 9.3 |
| res256/quad/full9/clip33 | full9 continued on 33-frame clips, 8 per update, body 3e-6, io 3e-5, 1000 updates (7.3 s/update) | 0.164 | 0.145 | 0.148 | 0.154 | 0.206 | 0.40 / 0.95 | 0.03 (from full9) | 2.3 |
| res256/quad/full9/clip33/long | clip33 continued at body 1e-5, 33-frame clips; stopped at 1600, no test | – | – | – | – | – | – | – | 3.1 |
| res256/quad/full9/clip33/tail | clip33 continued at body 3e-6, 33-frame clips; stopped at 1050 (val median flat at 0.098), no test | – | – | – | – | – | – | – | 2.1 |
| res256/quad/full9_b32 | full9 with 32 clips per update, per-channel scale; stopped at 2300, no test | – | – | – | – | – | – | – | 6.6 |
| res256/quad/full33 | 33-frame clips from the pretrained widened VAE, per-channel scale, 8 per update, body 1e-5; stopped at 4050 of 11000 (val 0.34, median 0.15), no test | – | – | – | – | – | – | – | 8.3 |
| res256/quad/full9_lr3e-5 | as full9, body 3e-5; stopped at update 1600, no test | – | – | – | – | – | – | – | 1.8 |

- On short clips, full9 matches the old 33-frame run (0.179 vs 0.186) with 31 % less compute; its val flattened at 0.19–0.22 over the last 2000 updates.
- Medians over sims: full9 0.107, full9/clip33 0.099, pairs/full/stage2 0.055, singles/decoders 0.034.
- The 33-frame stage cuts the late-frame window 57–100 from 0.233 to 0.161 and the mean from 0.179 to 0.164.
- Training the interface alone on the frozen body plateaus at train loss 0.17 from update 1000 and loses ρ.

## Why 0.08 was not reached

- Best quad: `res256/quad/full9/clip33`, test 0.164, median 0.099; pairs 0.080 / 0.055, singles 0.052 / 0.034.
- The gap is in the bulk of the sims, not in a tail: the quad's median is 1.8× the pairs median.
- Pressure enters through a slot the pretrained encoder ignores; it is the worst field in every run (0.21–0.25) and its val swings by 3× between checks.
- One 16-channel latent holds four nearly independent fields (64× compression, pairs 32×, singles 16×); Li's one-pass LTX VAE reached 0.080 (his space) only after ≈ 8 M training frames, our quad runs saw ≈ 1.4 M.
- The cheap levers are measured: io-only stage 0.375, luminance init worse at step 0, body lr 3e-5 worse than 1e-5, 32 clips per update worse per node-hour, low-lr 33-frame tails flat at val median 0.098, re-warming to 1e-5 undoes the fit.
- What is left needs LTX-scale training (≈ 5× our frames) or a larger latent; neither fits this round's evidence for a single affordable run.
- The anchor arm was not run: Lars has to choose between the implemented N(0, 1) target and an anchor to the start encoder's latents.

## Decisions and job reviews

- Lars, 2026-11-05: the 4-channel adapter mix stays fixed at the identity in every further quad run; only the widened io convolutions and the VAE body learn. conv_in is already a linear mix over its inputs, so a learned mix is redundant, and the identity keeps one field per channel. The tanh scale is one data-initialised scalar shared by the four channels, not one per field. (Replaced by the per-channel scale below.)
- Jobs 23496790 / 23496791 (full stages from `res256/quad/io`, which learned the mix at 3e-3) cancelled before they started.
- Job 23497466 (`res256/quad/full9`, control) was submitted before the reviewer rule; it uses the fixed identity mix.
- Review before job 23498050: change. Proposed were (A) an interface stage with the fixed mix or (B) the anchor arm. The io-only stage already plateaued at 0.375, and the anchor belongs on a settled recipe; run a body-lr arm of the control (3e-5 vs 1e-5) instead, since too few updates is the confirmed gap.
- Review of `res256/quad/full9_perfield` (per-field balanced RMSE + H1, would replace ctrl): skip. ctrl had already repaired the early damage by update 1500 (ρ 0.143, m_y 0.213 = step 0), and per-field balance would cut pressure's gradient share from ≈ 84 % to 25 % while pressure is the limiting field; code reverted.
- Step-0 init check (CPU, 6 val sims): a luminance (mean-of-RGB) init of the 4th slot gives ρ 2.91 / m_x 0.60 / m_y 1.10 / p 4.33 against 0.162 / 0.172 / 0.245 / 2.50 for the zero slot; not used.
- Val outliers are sims with tiny truth variance in the first frames (sim 1863: p std 0.028, p VRMSE 98.7); decoder outputs above 0.99 in magnitude are ≤ 0.7 % of pixels, so atanh saturation is not the cause.
- Stopped job 23498050 (`res256/quad/full9_lr3e-5`, body 3e-5) at update ≈ 1600: val 0.653 → 0.597 → 0.659 at 500 / 1000 / 1500, spread over sims growing (std 3.1 → 3.7), p 1.61 → 1.86, behind ctrl (body 1e-5) at every step from 1000 (0.471 at 1500). Checkpoint (update 1500) and log kept.
- Review of `res256/quad/full9_anchor` (ctrl + loss.latent 0.01, 9.5 node-h): skip until ctrl's test settles the recipe. The anchor pulls toward N(0, 1), ≈ 0.6 away from the start encoder's moments, so it would raise drift from the start rather than prevent it; open question for Lars: anchor to the start encoder's latents, or toward N(0, 1) judged by the DiT.
- Lars, 2026-11-05: the SoftClip scale is per channel (each channel's 0.98 quantile of |y| maps to 0.9), default for every grouping; required in every further quad run. Adapter files with one shared scale still load (broadcast). This replaces the shared scalar of the first 2026-11-05 decision. Runs before this (`io`, `full9`, `full9_lr3e-5`) used the shared scale.
- Step-0 check of the per-channel scale (CPU, frozen widened VAE, 10 val sims, frames 20–36 at stride 2): z-RMSE ρ / m_x / m_y / p 0.067 / 0.156 / 0.100 / 0.757 shared → 0.058 / 0.087 / 0.082 / 0.757 per channel; m_x pixels with |tanh| > 0.99 2.8 % → 0.08 % on the first sim; pressure (zero slot) unchanged.
- Review before job 23504575 (`res256/quad/full9_b32`, Lars's recipe: 32 clips per update, per-channel scale; ctrl's median eval chained first): go, 8000 updates. ctrl was still improving at 8000 and a higher lr is not the fix, so length is the lever; ≈ 19 node-h.
- Job 23504575 ran ctrl's median eval, then went out of memory at update 1 with 4 clips per GCD without checkpointing; resubmitted as 23504623 with block checkpointing (the fallback named in the review), recipe unchanged.
- Review of baseline median evals + `res256/quad/full9/clip33` (ctrl continued on 33-frame clips, body 3e-6, io 3e-5, 1000 updates): change job 1 (fold the evals of pairs/full/stage2 and singles/decoders, each from its own config, into job 2's allocation), go job 2. The late-frame window rises only after 9-frame training (57–100: 0.233 vs 0.195 for the 33-frame run).
- ctrl median over sims (eval-only `res256/quad/full9/median`): 0.107 (ρ 0.099, m_x 0.121, m_y 0.112, p 0.122) against a mean of 0.179.
- Baseline medians (eval-only, 100 test sims, stride 2; means reproduced): `res256/pairs/full/stage2/median` mean 0.080, median 0.055 (ρ 0.039, m_x 0.070, m_y 0.074, p 0.042); `res256/singles/decoders/median` mean 0.052, median 0.034 (0.039 / 0.038 / 0.037 / 0.028). The quad's median (0.107) is twice the pairs median: the gap is in the bulk of the sims, not only in a tail.
- Review before job 23508417 (`res256/quad/full9/clip33/long`, 4000 more 33-frame updates at body 1e-5, ≈ 9 node-h): go, with a stop rule: cancel if 33-frame val is not below ≈ 0.13 by update 2000. clip33's val was still falling at its end; expected test ≈ 0.10–0.12, not 0.08.
- Stopped job 23508417 (`res256/quad/full9/clip33/long`) at update ≈ 1600 of 4000: re-warming the body to 1e-5 threw val back from 0.141 to 0.513 (500), then 0.220 (1000) and 0.243 (1500; median 0.147); the review's stop rule (< 0.13 by 2000) was out of reach. Checkpoint (update 1500) and log kept. At 33 frames a body lr of 1e-5 undoes the fit; the low-lr tail (3e-6) is what helped.
- Review before job 23510173 (`res256/quad/full9/clip33/tail`, low-lr 33-frame tail from clip33): change, 1500 updates instead of 3000 (≈ 3.5 node-h), stop if the 33-frame val median is not below 0.095 by update 1000; clip33's val median had nearly flattened (0.108 → 0.098), so a long tail is not expected to reach 0.08.
- Lars, 2026-11-06: the next quad run trains on 33-frame clips from the start (ctrl's whole-trajectory ρ 0.153 against 0.095 on 9-frame val).
- Review: cancel job 23504623 (`res256/quad/full9_b32`, update ≈ 2300, ≈ 6.6 node-h): go. At equal compute ctrl was ahead on every field (b32 at 2000 ≈ ctrl at 4800 in node-h: ρ / m_x / m_y 0.157 / 0.246 / 0.216 vs ≈ 0.11 / 0.16 / 0.17), so 32 clips per update loses per node-hour. Checkpoint (update 2000) and log kept.
- Review of `res256/quad/full33` (job 23510994): change from 3 clips per GCD with checkpointing × 3000 updates to 1 clip per GCD without checkpointing × 11000 updates (same ≈ 23 node-h; updates, not batch, were the gap; 221 vs 264 ms per frame); stop if 33-frame val is not below 0.165 by update 4000.
- Stopped job 23510173 (`res256/quad/full9/clip33/tail`) at update ≈ 1050 by its stop rule: 33-frame val median 0.110, 0.106, 0.101, 0.098 at 250–1000 (needed < 0.095); clip33 ended at 0.098, so the low-lr tail is flat. Checkpoint (update 1000) and log kept.
- Review of `res256/quad/full33_anchor` (full33 + loss.latent 0.01, now): change, start only after full33 passes its stop rule at update 4000 and Lars has chosen the anchor target (N(0, 1), as implemented, or the start encoder's latents, which needs code).
- Stopped job 23510994 (`res256/quad/full33`) at update ≈ 4050 of 11000 by its stop rule: 33-frame val 0.52, 0.47, 0.41, 0.31, 0.59, 0.31, 0.35, 0.34 at 500–4000 (median 0.244 → 0.153), needed < 0.165. Caveat: the rule compares with the old run's annealed end, while full33's lr was still ≈ 80 % of peak. Checkpoint (update 4000) and log kept.

The anchor target is left open, and no further quad runs are planned: the project moved to the SRBA dataset on 2026-11-13.
