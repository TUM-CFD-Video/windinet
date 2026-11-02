| Model | RMSE | VRMSE | 1 – SSIM | MLW | H1 semi | H2 semi |
|---|---|---|---|---|---|---|
| shockwave_dit_jupiter_srba_loss_ssim0p2 | 0.033 ± 0.023 | 0.148 ± 0.053 | 0.025 ± 0.023 | 100.4 ± 21.7 | 1.00e-03 ± 9.45e-04 | 1.67e-03 ± 1.38e-03 |
| shockwave_dit_jupiter_srba_loss_ssim0p2: its VAE round trip | 0.014 ± 0.008 | 0.072 ± 0.020 | 0.009 ± 0.008 | 96.7 ± 23.6 | 4.95e-04 ± 4.43e-04 | 1.12e-03 ± 9.11e-04 |

shockwave_dit_jupiter_srba_loss_ssim0p2: frames [40, 50, 60, 70, 80, 90], clamp=True
shockwave_dit_jupiter_srba_loss_ssim0p2: generation per trajectory median 0.29 s (range 0.28-1.52), mean 0.31 s over 60 sims (18 s total); per generated frame 0.003 s (median / 96 frames); NVIDIA GH200 120GB, 5 steps, batch 1

shockwave_dit_jupiter_srba_loss_ssim0p2 per frame (VAE+DiT, mean ± std over sims)
| Frame | RMSE | VRMSE | 1 – SSIM | H1 semi | H2 semi |
|---|---|---|---|---|---|
| 40 | 0.018 ± 0.014 | 0.106 ± 0.059 | 0.009 ± 0.009 | 5.75e-04 ± 6.00e-04 | 1.16e-03 ± 1.06e-03 |
| 50 | 0.025 ± 0.018 | 0.114 ± 0.042 | 0.016 ± 0.016 | 8.33e-04 ± 8.78e-04 | 1.51e-03 ± 1.40e-03 |
| 60 | 0.031 ± 0.022 | 0.134 ± 0.052 | 0.023 ± 0.023 | 1.00e-03 ± 9.97e-04 | 1.69e-03 ± 1.46e-03 |
| 70 | 0.035 ± 0.025 | 0.158 ± 0.064 | 0.029 ± 0.028 | 1.12e-03 ± 1.11e-03 | 1.82e-03 ± 1.57e-03 |
| 80 | 0.040 ± 0.027 | 0.213 ± 0.101 | 0.035 ± 0.032 | 1.24e-03 ± 1.13e-03 | 1.97e-03 ± 1.57e-03 |
| 90 | 0.043 ± 0.028 | 0.289 ± 0.161 | 0.038 ± 0.030 | 1.23e-03 ± 1.00e-03 | 1.86e-03 ± 1.30e-03 |
