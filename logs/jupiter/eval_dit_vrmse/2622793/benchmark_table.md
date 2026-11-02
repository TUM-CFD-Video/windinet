| Model | RMSE | VRMSE | 1 – SSIM | MLW | H1 semi | H2 semi |
|---|---|---|---|---|---|---|
| shockwave_dit_jupiter_srba_loss_h1_200 | 0.036 ± 0.028 | 0.163 ± 0.099 | 0.027 ± 0.024 | 104.2 ± 23.1 | 1.06e-03 ± 9.84e-04 | 1.75e-03 ± 1.46e-03 |
| shockwave_dit_jupiter_srba_loss_h1_200: its VAE round trip | 0.014 ± 0.008 | 0.071 ± 0.020 | 0.009 ± 0.008 | 100.3 ± 24.8 | 4.58e-04 ± 4.22e-04 | 1.04e-03 ± 8.70e-04 |

shockwave_dit_jupiter_srba_loss_h1_200: frames [40, 50, 60, 70, 80, 90], clamp=True
shockwave_dit_jupiter_srba_loss_h1_200: generation per trajectory median 0.28 s (range 0.27-0.47), mean 0.28 s over 60 sims (17 s total); per generated frame 0.003 s (median / 96 frames); NVIDIA GH200 120GB, 5 steps, batch 1

shockwave_dit_jupiter_srba_loss_h1_200 per frame (VAE+DiT, mean ± std over sims)
| Frame | RMSE | VRMSE | 1 – SSIM | H1 semi | H2 semi |
|---|---|---|---|---|---|
| 40 | 0.022 ± 0.022 | 0.120 ± 0.096 | 0.011 ± 0.012 | 6.41e-04 ± 6.77e-04 | 1.24e-03 ± 1.17e-03 |
| 50 | 0.029 ± 0.028 | 0.130 ± 0.099 | 0.017 ± 0.018 | 8.91e-04 ± 9.25e-04 | 1.59e-03 ± 1.47e-03 |
| 60 | 0.034 ± 0.029 | 0.148 ± 0.103 | 0.025 ± 0.025 | 1.07e-03 ± 1.06e-03 | 1.79e-03 ± 1.59e-03 |
| 70 | 0.038 ± 0.030 | 0.175 ± 0.121 | 0.031 ± 0.030 | 1.19e-03 ± 1.14e-03 | 1.93e-03 ± 1.67e-03 |
| 80 | 0.042 ± 0.032 | 0.243 ± 0.230 | 0.037 ± 0.033 | 1.30e-03 ± 1.17e-03 | 2.06e-03 ± 1.66e-03 |
| 90 | 0.045 ± 0.028 | 0.307 ± 0.191 | 0.039 ± 0.030 | 1.28e-03 ± 1.01e-03 | 1.92e-03 ± 1.33e-03 |
