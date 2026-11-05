# Wan 2.1 VAE baselines on the LTX protocol

Round trip fields → adapter → Wan 2.1 VAE → adapter⁻¹ → fields, on the splits and trajectories of the LTX baseline of
Weihao Li's thesis (`docs/references/weihao_li_master_thesis_2026.pdf`).

Protocol (his): euler_mq 256×256, `train.h5` split 4000 train / 500 val (seed 42), `test.h5` 500 simulations (50 per γ),
whole 101-frame trajectories, VRMSE per field per simulation (thesis eq. 5.11) in physical units, averaged over
simulations and the fields. His numbers, LTX-Video VAE, 20 epochs over the 4000 training trajectories: decoder-only
0.144, full fine-tune (encoder + decoder) 0.080. They are in his clipped normalised space (thesis eq. 3.14), not in
physical units, and not directly comparable to ours.
Field statistics (adapter z-scoring, clamp bounds) in every run below came from `eulermq_256_test_stats.json`,
computed on test.h5; since 2026-10-03 the config uses the train-set file. Means and stds differ by at most 3 %.

Ours: adapter trained alone on the frozen VAE (2000 steps of 2 single frames, lr 3e-3), then the decoder behind the fixed
adapter (1000 updates of 6 clips × 5 frames, AdamW, peak lr 1e-6, 100 warm-up steps, cosine, RMSE + H1, grad clip 5),
30 k training frames in total against his 8.1 M. Commands: `jobs/lc/vae_job1_baselines.txt`, `jobs/lc/vae_job2_decoder.txt`.

## 500 test trajectories

| VAE | fields per image | VAE training | VRMSE | run |
|---|---|---|---|---|
| LTX (Li) | 4 in one pass | decoder, 20 epochs | (0.144) | thesis |
| LTX (Li) | 4 in one pass | encoder + decoder, 20 epochs | (0.080) | thesis |
| Wan 2.1 frozen | ρ, m_x, m_y (p dropped) | none | 0.140 | `res256/rgb3/adapter_test500` |
| Wan 2.1 | ρ, m_x, m_y | decoder 1e-6 | 0.111 | `res256/rgb3/two_stage_test500` |
| Wan 2.1 | ρ, m_x, m_y | decoder 3e-6 | 0.108 | `res256/rgb3/two_stage_lr3e-6_test500` |
| Wan 2.1 frozen | (ρ, p) + (m_x, m_y) | none | 0.094 | `res256/pairs/adapter_test500` |
| Wan 2.1 | (ρ, p) + (m_x, m_y) | one decoder, 1e-6 | 0.073 | `res256/pairs/two_stage_test500` |
| Wan 2.1 | (ρ, p) + (m_x, m_y) | one decoder per image, 1e-6 | 0.072 | `res256/pairs/2dec_test500` |
| Wan 2.1 | (ρ, p) + (m_x, m_y) | as above, continued on 33-frame clips, 2000 × 8 clips on 8 GPUs, 3e-6 | 0.060 | `res256/pairs/2dec_clip33` |

LTX values in parentheses are in his clipped space, not physical units.
Caveat for the comparison: the pairs setting makes two VAE passes and two latents per trajectory; his model makes one.
Wan's latent is 8× spatial / 4× temporal with 16 channels, LTX's 32× / 8× with 128, so Wan's is 8× larger.

## Ablations on the first 100 test trajectories

The 100 are the lowest-γ simulations of the sorted test ids and are harder than the full set (rgb3: 0.138 vs 0.111).
Every arm: 1000 updates of 6 clips × 5 frames behind the stage-1 adapter. "val" = 100 validation simulations, 5-frame
clips, at the last update.

| arm | adapter | decoder | loss | val | test | run |
|---|---|---|---|---|---|---|
| rgb3 frozen, z-score init | untrained | frozen | – | 0.259 | 0.379 | `res256/rgb3/frozen` |
| rgb3 adapter only | trained | frozen | – | – | 0.176 | `res256/rgb3/adapter` |
| rgb3 two-stage | fixed | 1e-6 | RMSE + H1 | 0.094 | 0.138 | `res256/rgb3/two_stage` |
| rgb3, adapter continues | lr 1e-4 | 1e-6 | RMSE + H1 | 0.091 | 0.134 | `res256/rgb3/two_stage_joint` |
| rgb3, lr 3e-6 | fixed | 3e-6 | RMSE + H1 | 0.091 | 0.133 | `res256/rgb3/two_stage_lr3e-6` |
| rgb3, his loss | fixed | 1e-6 | RMSE + 50 H1 | 0.106 | 0.164 | `res256/rgb3/two_stage_h1_50` |
| pairs frozen, z-score init | untrained | frozen | – | 0.335 | 0.299 | `res256/pairs/frozen` |
| pairs two-stage | fixed | one, 1e-6 | RMSE + H1 | 0.082 | 0.089 | `res256/pairs/two_stage` |
| pairs, adapter continues | lr 1e-4 | one, 1e-6 | RMSE + H1 | 0.080 | 0.087 | `res256/pairs/two_stage_joint` |
| pairs, decoder per image | fixed | two, 1e-6 | RMSE + H1 | 0.075 | 0.088 | `res256/pairs/2dec` |
| pairs, both | lr 1e-4 | two, 1e-6 | RMSE + H1 | 0.073 | 0.086 | `res256/pairs/2dec_joint` |

## Findings

- The frozen Wan VAE behind the pairs adapter reaches 0.094 on the 500 trajectories.
- Decoder tuning takes that to 0.072, with 0.4 % of his training frames and the encoder untouched.
- Four fields in two images beat three fields in one image by 0.038 (0.073 vs 0.111); the extra field is not the reason,
  pairs is better on every shared field (ρ 0.050 vs 0.102, m_x 0.084 vs 0.105, m_y 0.099 vs 0.127).
- Decoder tuning on 5-frame clips gains 0.05 on validation clips but 0.029 (rgb3) and 0.020 (pairs) on trajectories: a
  decoder tuned with 2 latent frames of temporal context transfers only in part to 26. Continuing on 33-frame clips
  (9 latent frames) takes 0.072 to 0.060 on the 500 trajectories; the per-field numbers are ρ 0.041, p 0.045, m_x 0.072,
  m_y 0.083.
- One decoder per image is worth 0.007 on validation clips and 0.002 on 500 trajectories.
- Letting the adapter continue at lr 1e-4 while the decoder trains changes the trajectory result by at most 0.004; the
  adapter stays fixed, so the DiT's latents do not depend on the decoder run.
- Decoder peak lr 3e-6 is 0.003 better than 1e-6 on 500 trajectories; 50·H1 is 0.013 (val) to 0.026 (test) worse than
  1·H1.
- After decoder tuning m_y is the worst field in every run (0.083 in the best run against 0.041 for ρ).

## Reproduce a test on all 500 trajectories

```bash
torchrun --standalone --nproc_per_node=8 scripts/wan/train_vae.py configs/wan/vae.yaml \
    --name res256/pairs/2dec_test500 --desc "..." --set load=results/wan/vae/res256/pairs/2dec --set train.vae_per_group=true \
    --set train.steps=0 --set train.vae_parts=none --set data.test_sims=null
```

## Later VAE runs

Decoders for the DiT line, the 512 data and rollouts, and the other image layouts; test VRMSE, `run` under
`results/wan/vae/`.

| run | what | VRMSE |
|---|---|---|
| res256/pairs/2dec_clip33/stride1 | round-trip floor of the pairs decoders, 100 test sims, frame stride 1 | 0.075 |
| res256/pairs/2dec_clip33/stride2 | same at frame stride 2 | 0.098 |
| res256/pairs/2dec_clip33/eval_stride4 | same on whole trajectories at stride 4 | 0.120 |
| res256/pairs/full | encoder + decoder fine-tuned from 2dec_clip33 at 1e-5 | 0.085 |
| res256/pairs/full/stage2 | + 2000 updates at 5e-6 | 0.080 |
| res512/pairs/2dec | per-image decoders continued on the native 512 data | 0.067 |
| res256/pairs/2dec_clip33/rollouts/init | decoders untrained, on stage3_d4's test rollouts | 0.232 |
| res256/pairs/2dec_clip33/rollouts/amse | fine-tuned on stage3_d4's train rollouts, AMSE (Subich et al. 2025) | 0.229 |
| res256/pairs/2dec_clip33/rollouts/rmse_h1 | same, RMSE + H1 | 0.220 |
| res256/pairs/2dec_clip33/rollouts/l1_gdl | same, L1 + gradient-difference L1 | 0.220 |
| res256/singles/adapter | one field per image, adapter on the frozen VAE | 0.066 |
| res256/singles/decoders | one decoder per field behind it | 0.052 |
| res256/singles/decoders/stride1 | its round-trip floor at frame stride 1 | 0.044 |
| res256/quad/full | four fields in one image, VAE widened to 4 channels, encoder + decoder | 0.200 |
| res256/quad/full_io1e-4 | same, in/out convolutions at 1e-4 | 0.186 |

The `*/diag` folders under `rollouts/` re-evaluate init, l1_gdl and rmse_h1 for the newer metrics (same VRMSE).
Four fields in one image lose to two images of two fields (0.19–0.20 against 0.07–0.09).
