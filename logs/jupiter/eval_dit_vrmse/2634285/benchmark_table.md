| Model | RMSE | VRMSE | 1 – SSIM | MLW | H1 semi | H2 semi |
|---|---|---|---|---|---|---|
| shockwave_dit_jupiter_baseline_s44_lr1e4_cond9 | 0.045 ± 0.033 | 0.576 ± 0.392 | 0.034 ± 0.035 | 61.8 ± 9.5 | 2.30e-04 ± 3.56e-04 | 1.60e-04 ± 2.72e-04 |
| shockwave_dit_jupiter_baseline_s44_lr1e4_cond9: its VAE round trip | 0.006 ± 0.005 | 0.137 ± 0.765 | 0.005 ± 0.007 | 39.2 ± 13.5 | 6.31e-05 ± 1.14e-04 | 9.61e-05 ± 1.77e-04 |

shockwave_dit_jupiter_baseline_s44_lr1e4_cond9: frames [40, 50, 60, 70, 80, 90], clamp=True
shockwave_dit_jupiter_baseline_s44_lr1e4_cond9: generation per trajectory median 0.28 s (range 0.26-1.18), mean 0.29 s over 500 sims (143 s total); per generated frame 0.003 s (median / 96 frames); NVIDIA GH200 120GB, 5 steps, batch 1

shockwave_dit_jupiter_baseline_s44_lr1e4_cond9 per frame (VAE+DiT, mean ± std over sims)
| Frame | RMSE | VRMSE | 1 – SSIM | H1 semi | H2 semi |
|---|---|---|---|---|---|
| 40 | 0.042 ± 0.031 | 0.460 ± 0.381 | 0.040 ± 0.039 | 3.20e-04 ± 4.83e-04 | 2.37e-04 ± 3.75e-04 |
| 50 | 0.043 ± 0.032 | 0.620 ± 2.060 | 0.038 ± 0.038 | 2.76e-04 ± 4.38e-04 | 1.95e-04 ± 3.34e-04 |
| 60 | 0.044 ± 0.034 | 0.636 ± 0.739 | 0.035 ± 0.038 | 2.44e-04 ± 4.03e-04 | 1.68e-04 ± 3.02e-04 |
| 70 | 0.044 ± 0.034 | 1.016 ± 7.453 | 0.033 ± 0.036 | 2.09e-04 ± 3.53e-04 | 1.42e-04 ± 2.66e-04 |
| 80 | 0.045 ± 0.035 | 1.171 ± 8.294 | 0.030 ± 0.033 | 1.76e-04 ± 2.95e-04 | 1.18e-04 ± 2.22e-04 |
| 90 | 0.045 ± 0.036 | 1.380 ± 9.376 | 0.028 ± 0.031 | 1.56e-04 ± 2.61e-04 | 1.02e-04 ± 1.96e-04 |
