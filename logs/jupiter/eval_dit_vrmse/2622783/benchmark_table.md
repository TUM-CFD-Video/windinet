| Model | RMSE | VRMSE | 1 – SSIM | MLW | H1 semi | H2 semi |
|---|---|---|---|---|---|---|
| shockwave_dit_jupiter_srba_loss_h1_10 | 0.034 ± 0.022 | 0.152 ± 0.052 | 0.026 ± 0.023 | 104.3 ± 22.3 | 1.03e-03 ± 9.39e-04 | 1.71e-03 ± 1.37e-03 |
| shockwave_dit_jupiter_srba_loss_h1_10: its VAE round trip | 0.015 ± 0.008 | 0.077 ± 0.019 | 0.011 ± 0.009 | 97.6 ± 23.4 | 5.48e-04 ± 4.66e-04 | 1.23e-03 ± 9.60e-04 |

shockwave_dit_jupiter_srba_loss_h1_10: frames [40, 50, 60, 70, 80, 90], clamp=True
shockwave_dit_jupiter_srba_loss_h1_10: generation per trajectory median 0.30 s (range 0.29-0.48), mean 0.30 s over 60 sims (18 s total); per generated frame 0.003 s (median / 96 frames); NVIDIA GH200 120GB, 5 steps, batch 1

shockwave_dit_jupiter_srba_loss_h1_10 per frame (VAE+DiT, mean ± std over sims)
| Frame | RMSE | VRMSE | 1 – SSIM | H1 semi | H2 semi |
|---|---|---|---|---|---|
| 40 | 0.019 ± 0.013 | 0.111 ± 0.055 | 0.010 ± 0.009 | 6.08e-04 ± 5.91e-04 | 1.23e-03 ± 1.06e-03 |
| 50 | 0.026 ± 0.018 | 0.119 ± 0.041 | 0.017 ± 0.016 | 8.60e-04 ± 8.69e-04 | 1.56e-03 ± 1.38e-03 |
| 60 | 0.032 ± 0.022 | 0.141 ± 0.053 | 0.025 ± 0.023 | 1.06e-03 ± 1.00e-03 | 1.78e-03 ± 1.47e-03 |
| 70 | 0.036 ± 0.025 | 0.164 ± 0.064 | 0.031 ± 0.028 | 1.16e-03 ± 1.10e-03 | 1.87e-03 ± 1.56e-03 |
| 80 | 0.040 ± 0.027 | 0.215 ± 0.097 | 0.036 ± 0.032 | 1.24e-03 ± 1.12e-03 | 1.97e-03 ± 1.55e-03 |
| 90 | 0.043 ± 0.027 | 0.291 ± 0.157 | 0.038 ± 0.030 | 1.22e-03 ± 9.93e-04 | 1.85e-03 ± 1.29e-03 |
