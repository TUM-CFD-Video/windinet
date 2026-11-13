# Pruning and distilling the Wan DiT: literature and plan

This is the literature note written before the pruning study; the implemented study and its results are in
[dit_pruning.md](dit_pruning.md). Model: Wan 2.1 T2V 1.3B (`windinet/wan/__init__.py`), 30 blocks, hidden 1536,
12 heads × 128, FFN 8960, patch (1, 2, 2) (checked against the HF config). Numbers marked *est.* are mine, the rest
are from the cited papers.

## Where the compute is (est.)

| item | value |
|---|---|
| tokens per sample, 256², pairs, 9 latent frames | 16 × 16 × 9 ≈ 2300 |
| parameters per block: self-attn / cross-attn / FFN | 9.4 M / 9.4 M / 27.5 M (FFN ≈ 60 %) |
| attention scores (QKᵀ, AV) share of compute at 2300 tokens | ≈ 15 % |
| our fine-tune cost, `unrolled` (4000 updates × 32 clips) | 6.1 node-h |

Depth and FFN width are the levers; attention sparsity can win at most ≈ 15 % at this sequence length.

## Prioritised list (cheapest first)

| # | what | size / speed | expected quality cost | recovery | our compute (est.) |
|---|---|---|---|---|---|
| 1 | cache the cross-attention K, V per sample across sampler steps (empty prompt + gamma tokens) | 0 params, ≈ 2 % (K/V projections of the text tokens only) | none (exact) | none | 0 |
| 2 | block-importance scan: drop each block alone, score 2-step test VRMSE on 50 sims | – | diagnostic | none | ≈ 1 node-h |
| 3 | depth prune 30 → 20 blocks (every third removed, first and last kept) + output KD from our teacher | 1.39 → 0.93 B, ≈ 1.5× | small after KD (TinyFusion, Minitron) | 4000–8000 updates | 8–15 node-h |
| 4 | same with feature KD (each student block matches the teacher's 2-block group output, outlier mask) | as 3 | 5.79 → 3.73 FID gain over plain fine-tune in TinyFusion at equal steps | as 3 | +20 % per step |
| 5 | FFN width 8960 → 4480 (activation importance on 1024 calibration clips, Minitron) + KD | −30 % params and FLOPs | moderate; one DiT study finds FFN more sensitive than heads | 4000–8000 updates | 8–15 node-h |
| 6 | head pruning 12 → 9 per block (learned or L2 importance) | −6–8 % compute | ≈ none: MobileWan 23 % heads, VBench 83.12 → 83.16 | short | 3–5 node-h |
| 7 | depth 30 → 15 (every other block) + feature KD | 0.7 B, ≈ 2× | noticeable; TinyFusion DiT-XL 28 → 14: FID 2.27 → 2.86 after 500 k steps | ≥ our full fine-tune | 30–60 node-h |
| 8 | 2 → 1 sampling step (rCM / DMD on our teacher) | 2× | loss expected; 1 step 72.8 vs 2 steps 76.9 VBench (2507.13343); our 1-step was 0.383 vs 0.363 at 2 (v1) | 2000–5000 updates | 10–20 node-h |
| 9 | int8 / fp8 quantisation | memory only | – | – | skip on MI250X: int8 peak equals bf16 (383 TFLOPS), no fp8 |
| 10 | sparse attention, token merging | ≤ 10–15 % | risk of smeared fronts | none | skip: CUDA kernels, attention is small here |

Rows 3–7 combine: depth first, then width on the shortened model (Minitron: iterate from the nearest size).

## How to move our trained weights into a smaller model fast

- Student = teacher with blocks deleted (`transformer.blocks` is a `ModuleList`; drop indices, keep patchify, gamma
  tokens, head). No new weights, no re-init.
- Keep the first and last blocks; every-other-layer is the strong cheap baseline in TinyFusion ("oracle" FID 7.43 vs
  22.28 for ShortGPT-style cosine selection after 100 k steps).
- Choose blocks after a short recovery, not by the loss right after removal: TinyFusion shows the immediate loss
  does not predict the recovered quality; Minitron sees rankings flip within ≈ 200 steps.
- Loss: our current flow loss plus MSE to the teacher's velocity at the same (x_t, t, history); add per-block feature
  KD for cuts deeper than a third.
- Teacher outputs on the fly (one extra forward of the 1.3 B, no backward): ≈ +35 % per step.
- Recovery budget: start at 4000 updates (one `unrolled`-sized run, ≈ 6 node-h plus the teacher forward), extend if
  the 2-step VRMSE is still falling.
- Judge on 2-step rollout VRMSE (v2, 500 sims) and the paired CI against the teacher, not on one-step loss.

## Risks

- Wan is depth-sensitive: MobileWan (Wan 2.2 5B) found block pruning hurt and pruned heads instead.
- Deep blocks carry motion (VDMini); our error is dominated by rollout drift, so cutting deep blocks may cost most where
  we are weakest.
- Pruned models need recovery data at the scale of a fine-tune; with no new data the KD target is the only extra signal.
- A 1.5–2× faster DiT matters only if DiT time dominates: at 2 steps the VAE decode is a comparable share of inference.

## Sources

- ShortGPT, block influence: arXiv 2403.03853
- Gromov et al., deeper layers: arXiv 2403.17887
- LayerDrop: arXiv 1909.11556
- Minitron, pruning + KD: arXiv 2407.14679
- TinyFusion, learnable depth pruning for DiTs: arXiv 2412.01199
- LD-Pruner: arXiv 2404.11936
- BK-SDM: arXiv 2305.15798
- LAPTOP-Diff: arXiv 2404.11098
- EcoDiff: arXiv 2412.02852
- VDMini, video diffusion pruning: arXiv 2411.18375
- MobileWan: arXiv 2607.06173
- Taming DiT for mobile video: arXiv 2507.13343
- MobileVD: arXiv 2412.07583
- SnapGen-V: arXiv 2412.10494
- On-device Sora: arXiv 2502.04363
- Self-Forcing: arXiv 2506.08009; rCM: arXiv 2510.08431; MagicDistillation: arXiv 2503.13319
- ViDiT-Q: arXiv 2406.02540; SVDQuant: arXiv 2411.05007
- Sparse VideoGen: arXiv 2502.01776; SVG2: arXiv 2505.18875
- MI250X peak rates: amd.com MI250X product page
