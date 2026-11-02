| Model | RMSE | VRMSE | 1 – SSIM | MLW | H1 semi | H2 semi |
|---|---|---|---|---|---|---|
| shockwave_dit_jupiter_srba_loss_h2_50 | 0.033 ± 0.023 | 0.150 ± 0.051 | 0.025 ± 0.023 | 100.9 ± 21.1 | 1.00e-03 ± 9.57e-04 | 1.68e-03 ± 1.42e-03 |
| shockwave_dit_jupiter_srba_loss_h2_50: its VAE round trip | 0.014 ± 0.008 | 0.073 ± 0.020 | 0.010 ± 0.008 | 98.8 ± 24.1 | 4.68e-04 ± 4.23e-04 | 1.05e-03 ± 8.67e-04 |

shockwave_dit_jupiter_srba_loss_h2_50: frames [40, 50, 60, 70, 80, 90], clamp=True
shockwave_dit_jupiter_srba_loss_h2_50: generation per trajectory median 0.30 s (range 0.29-1.42), mean 0.33 s over 60 sims (20 s total); per generated frame 0.003 s (median / 96 frames); NVIDIA GH200 120GB, 5 steps, batch 1

shockwave_dit_jupiter_srba_loss_h2_50 per frame (VAE+DiT, mean ± std over sims)
| Frame | RMSE | VRMSE | 1 – SSIM | H1 semi | H2 semi |
|---|---|---|---|---|---|
| 40 | 0.019 ± 0.014 | 0.109 ± 0.058 | 0.010 ± 0.010 | 5.78e-04 ± 6.23e-04 | 1.15e-03 ± 1.10e-03 |
| 50 | 0.025 ± 0.018 | 0.117 ± 0.042 | 0.016 ± 0.017 | 8.39e-04 ± 9.10e-04 | 1.52e-03 ± 1.46e-03 |
| 60 | 0.031 ± 0.022 | 0.137 ± 0.051 | 0.023 ± 0.023 | 1.01e-03 ± 1.02e-03 | 1.71e-03 ± 1.54e-03 |
| 70 | 0.035 ± 0.024 | 0.161 ± 0.062 | 0.029 ± 0.028 | 1.13e-03 ± 1.10e-03 | 1.85e-03 ± 1.60e-03 |
| 80 | 0.040 ± 0.027 | 0.215 ± 0.099 | 0.035 ± 0.032 | 1.24e-03 ± 1.14e-03 | 1.99e-03 ± 1.62e-03 |
| 90 | 0.043 ± 0.027 | 0.291 ± 0.157 | 0.038 ± 0.030 | 1.23e-03 ± 9.94e-04 | 1.87e-03 ± 1.31e-03 |
