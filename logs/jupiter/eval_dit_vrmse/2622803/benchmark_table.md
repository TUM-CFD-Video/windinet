| Model | RMSE | VRMSE | 1 – SSIM | MLW | H1 semi | H2 semi |
|---|---|---|---|---|---|---|
| shockwave_dit_jupiter_srba_loss_h2_10 | 0.033 ± 0.022 | 0.148 ± 0.052 | 0.025 ± 0.023 | 102.5 ± 22.4 | 1.00e-03 ± 9.47e-04 | 1.68e-03 ± 1.40e-03 |
| shockwave_dit_jupiter_srba_loss_h2_10: its VAE round trip | 0.014 ± 0.008 | 0.072 ± 0.020 | 0.010 ± 0.008 | 99.7 ± 24.4 | 4.81e-04 ± 4.34e-04 | 1.09e-03 ± 8.94e-04 |

shockwave_dit_jupiter_srba_loss_h2_10: frames [40, 50, 60, 70, 80, 90], clamp=True
shockwave_dit_jupiter_srba_loss_h2_10: generation per trajectory median 0.31 s (range 0.31-1.26), mean 0.33 s over 60 sims (20 s total); per generated frame 0.003 s (median / 96 frames); NVIDIA GH200 120GB, 5 steps, batch 1

shockwave_dit_jupiter_srba_loss_h2_10 per frame (VAE+DiT, mean ± std over sims)
| Frame | RMSE | VRMSE | 1 – SSIM | H1 semi | H2 semi |
|---|---|---|---|---|---|
| 40 | 0.018 ± 0.014 | 0.106 ± 0.056 | 0.009 ± 0.009 | 5.75e-04 ± 6.02e-04 | 1.15e-03 ± 1.07e-03 |
| 50 | 0.025 ± 0.018 | 0.115 ± 0.042 | 0.016 ± 0.016 | 8.28e-04 ± 8.77e-04 | 1.51e-03 ± 1.41e-03 |
| 60 | 0.031 ± 0.022 | 0.134 ± 0.051 | 0.023 ± 0.023 | 1.01e-03 ± 1.01e-03 | 1.71e-03 ± 1.52e-03 |
| 70 | 0.035 ± 0.024 | 0.160 ± 0.062 | 0.030 ± 0.028 | 1.14e-03 ± 1.11e-03 | 1.86e-03 ± 1.61e-03 |
| 80 | 0.039 ± 0.026 | 0.213 ± 0.099 | 0.035 ± 0.032 | 1.24e-03 ± 1.13e-03 | 1.98e-03 ± 1.60e-03 |
| 90 | 0.043 ± 0.027 | 0.289 ± 0.159 | 0.038 ± 0.030 | 1.23e-03 ± 9.98e-04 | 1.86e-03 ± 1.31e-03 |
