# Wan 2.1 VAE baselines on the LTX protocol

Round trip fields → adapter → Wan 2.1 VAE → adapter⁻¹ → fields, measured as the LTX baseline of Weihao Li's thesis
(`docs/references/weihao_li_master_thesis_2026.pdf`) measures it, so the two VAEs can be compared on one number.

Protocol (his): euler_mq 256×256, `train.h5` split 4000 train / 500 val (seed 42), `test.h5` 500 simulations (50 per γ),
whole 101-frame trajectories, VRMSE per field per simulation (thesis eq. 5.11) in his 5σ-clipped space, averaged over
simulations and the four fields. The physical-unit VRMSE is given next to it. His numbers, LTX-Video VAE, 20 epochs over
the 4000 training trajectories: decoder-only 0.144, full fine-tune (encoder + decoder) 0.080.

Ours: adapter trained alone on the frozen VAE (2000 steps of 2 single frames, lr 3e-3), then the decoder behind the fixed
adapter (1000 updates of 6 clips × 5 frames, AdamW, peak lr 1e-6, 100 warm-up steps, cosine, RMSE + H1, grad clip 5),
30 k training frames in total against his 8.1 M. Commands: `jobs/lc/vae_job1_baselines.txt`, `jobs/lc/vae_job2_decoder.txt`.

## 500 test trajectories

| VAE | fields per image | VAE training | clipped | physical | run |
|---|---|---|---|---|---|
| LTX (Li) | 4 in one pass | decoder, 20 epochs | 0.144 | – | thesis |
| LTX (Li) | 4 in one pass | encoder + decoder, 20 epochs | 0.080 | – | thesis |
| Wan 2.1 frozen | ρ, m_x, m_y (p dropped) | none | 0.0945 | 0.140 | `rgb3_adapter_test500` |
| Wan 2.1 | ρ, m_x, m_y | decoder 1e-6 | 0.0889 | 0.111 | `rgb3_two_stage_test500` |
| Wan 2.1 | ρ, m_x, m_y | decoder 3e-6 | 0.0889 | 0.108 | `rgb3_two_stage_lr3e-6_test500` |
| Wan 2.1 frozen | (ρ, p) + (m_x, m_y) | none | 0.0706 | 0.094 | `pairs_adapter_test500` |
| Wan 2.1 | (ρ, p) + (m_x, m_y) | one decoder, 1e-6 | 0.0617 | 0.073 | `pairs_two_stage_test500` |
| Wan 2.1 | (ρ, p) + (m_x, m_y) | one decoder per image, 1e-6 | 0.0603 | 0.072 | `pairs_2dec_test500` |
| Wan 2.1 | (ρ, p) + (m_x, m_y) | as above, continued on 33-frame clips, 2000 × 8 clips on 8 GPUs, 3e-6 | 0.0519 | 0.060 | `pairs_2dec_clip33` |

Caveat for the comparison: the pairs setting makes two VAE passes and two latents per trajectory; his model makes one.
Wan's latent is 8× spatial / 4× temporal with 16 channels, LTX's 32× / 8× with 128, so Wan's is 8× larger.

## Ablations on the first 100 test trajectories

The 100 are the lowest-γ simulations of the sorted test ids and are harder than the full set (rgb3: 0.097 vs 0.089).
Every arm: 1000 updates of 6 clips × 5 frames behind the stage-1 adapter. "val" = 100 validation simulations, 5-frame
clips, at the last update.

| arm | adapter | decoder | loss | val (clipped) | test (clipped) | run |
|---|---|---|---|---|---|---|
| rgb3 frozen, z-score init | untrained | frozen | – | 0.153 | 0.158 | `rgb3_frozen` |
| rgb3 adapter only | trained | frozen | – | – | 0.104 | `rgb3_adapter` |
| rgb3 two-stage | fixed | 1e-6 | RMSE + H1 | 0.085 | 0.098 | `rgb3_two_stage` |
| rgb3, adapter continues | lr 1e-4 | 1e-6 | RMSE + H1 | 0.084 | 0.097 | `rgb3_two_stage_joint` |
| rgb3, lr 3e-6 | fixed | 3e-6 | RMSE + H1 | 0.082 | 0.097 | `rgb3_two_stage_lr3e-6` |
| rgb3, his loss | fixed | 1e-6 | RMSE + 50 H1 | 0.089 | 0.099 | `rgb3_two_stage_h1_50` |
| pairs frozen, z-score init | untrained | frozen | – | 0.224 | 0.130 | `pairs_frozen` |
| pairs two-stage | fixed | one, 1e-6 | RMSE + H1 | 0.079 | 0.069 | `pairs_two_stage` |
| pairs, adapter continues | lr 1e-4 | one, 1e-6 | RMSE + H1 | 0.077 | 0.068 | `pairs_two_stage_joint` |
| pairs, decoder per image | fixed | two, 1e-6 | RMSE + H1 | 0.072 | 0.067 | `pairs_2dec` |
| pairs, both | lr 1e-4 | two, 1e-6 | RMSE + H1 | 0.070 | 0.067 | `pairs_2dec_joint` |

## Findings

- The frozen Wan VAE behind the pairs adapter (0.071) is already below the LTX full fine-tune (0.080).
- Decoder tuning adds 0.010 on top of that (0.060), with 0.4 % of his training frames and the encoder untouched.
- Four fields in two images beat three fields in one image by 0.03; the extra field is not the reason, pressure is the
  hardest field and sits in the four-field mean.
- Decoder tuning on 5-frame clips gains 0.04 on clips but 0.006 (rgb3) to 0.010 (pairs) on trajectories: a decoder tuned
  with 2 latent frames of temporal context does not transfer to 26. Continuing on 33-frame clips (9 latent frames) takes
  0.0603 to 0.0519 on the 500 trajectories; the per-field numbers are ρ 0.041, p 0.045, m_x 0.072, m_y 0.083.
- One decoder per image is worth 0.007 on clips and 0.0014 on 500 trajectories.
- Letting the adapter continue at lr 1e-4 while the decoder trains changes the trajectory result by at most 0.002; the
  adapter stays fixed, so the DiT's latents do not depend on the decoder run.
- Decoder peak lr 3e-6 and 1e-6 give the same 500-trajectory number; 50·H1 is 0.002 to 0.004 worse than 1·H1.
- Pressure is the worst field in every setting (0.053 in the best run against 0.045 for ρ): its within-simulation
  variance is the smallest, so the same error at the shock front weighs more.

## Reproduce a test on all 500 trajectories

```bash
torchrun --standalone --nproc_per_node=8 scripts/wan/train_vae.py configs/wan/vae.yaml \
    --name pairs_2dec_test500 --desc "..." --set load=results/wan/vae/pairs_2dec --set train.vae_per_group=true \
    --set train.steps=0 --set train.vae_parts=none --set data.test_sims=null
```
