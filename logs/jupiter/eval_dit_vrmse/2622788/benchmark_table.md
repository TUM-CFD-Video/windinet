| Model | RMSE | VRMSE | 1 – SSIM | MLW | H1 semi | H2 semi |
|---|---|---|---|---|---|---|
| shockwave_dit_jupiter_srba_loss_h1_100 | 0.033 ± 0.023 | 0.150 ± 0.052 | 0.025 ± 0.023 | 103.3 ± 22.4 | 1.02e-03 ± 9.55e-04 | 1.69e-03 ± 1.41e-03 |
| shockwave_dit_jupiter_srba_loss_h1_100: its VAE round trip | 0.014 ± 0.008 | 0.071 ± 0.020 | 0.009 ± 0.008 | 98.9 ± 24.2 | 4.70e-04 ± 4.30e-04 | 1.07e-03 ± 8.87e-04 |

shockwave_dit_jupiter_srba_loss_h1_100: frames [40, 50, 60, 70, 80, 90], clamp=True
shockwave_dit_jupiter_srba_loss_h1_100: generation per trajectory median 0.27 s (range 0.26-1.08), mean 0.29 s over 60 sims (17 s total); per generated frame 0.003 s (median / 96 frames); NVIDIA GH200 120GB, 5 steps, batch 1

shockwave_dit_jupiter_srba_loss_h1_100 per frame (VAE+DiT, mean ± std over sims)
| Frame | RMSE | VRMSE | 1 – SSIM | H1 semi | H2 semi |
|---|---|---|---|---|---|
| 40 | 0.018 ± 0.014 | 0.105 ± 0.055 | 0.009 ± 0.009 | 5.78e-04 ± 6.18e-04 | 1.15e-03 ± 1.09e-03 |
| 50 | 0.025 ± 0.019 | 0.115 ± 0.044 | 0.016 ± 0.016 | 8.40e-04 ± 9.01e-04 | 1.52e-03 ± 1.43e-03 |
| 60 | 0.031 ± 0.022 | 0.136 ± 0.052 | 0.023 ± 0.023 | 1.01e-03 ± 1.00e-03 | 1.71e-03 ± 1.49e-03 |
| 70 | 0.036 ± 0.025 | 0.161 ± 0.062 | 0.030 ± 0.027 | 1.15e-03 ± 1.11e-03 | 1.86e-03 ± 1.59e-03 |
| 80 | 0.040 ± 0.027 | 0.216 ± 0.100 | 0.035 ± 0.032 | 1.26e-03 ± 1.13e-03 | 2.00e-03 ± 1.59e-03 |
| 90 | 0.043 ± 0.028 | 0.293 ± 0.160 | 0.038 ± 0.030 | 1.26e-03 ± 1.02e-03 | 1.90e-03 ± 1.33e-03 |
