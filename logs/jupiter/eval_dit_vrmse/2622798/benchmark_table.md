| Model | RMSE | VRMSE | 1 – SSIM | MLW | H1 semi | H2 semi |
|---|---|---|---|---|---|---|
| shockwave_dit_jupiter_srba_loss_h2_1 | 0.033 ± 0.022 | 0.151 ± 0.051 | 0.026 ± 0.023 | 103.4 ± 22.4 | 1.03e-03 ± 9.51e-04 | 1.71e-03 ± 1.41e-03 |
| shockwave_dit_jupiter_srba_loss_h2_1: its VAE round trip | 0.014 ± 0.008 | 0.072 ± 0.020 | 0.010 ± 0.008 | 98.8 ± 24.0 | 4.89e-04 ± 4.39e-04 | 1.11e-03 ± 9.04e-04 |

shockwave_dit_jupiter_srba_loss_h2_1: frames [40, 50, 60, 70, 80, 90], clamp=True
shockwave_dit_jupiter_srba_loss_h2_1: generation per trajectory median 0.32 s (range 0.30-1.08), mean 0.34 s over 60 sims (20 s total); per generated frame 0.003 s (median / 96 frames); NVIDIA GH200 120GB, 5 steps, batch 1

shockwave_dit_jupiter_srba_loss_h2_1 per frame (VAE+DiT, mean ± std over sims)
| Frame | RMSE | VRMSE | 1 – SSIM | H1 semi | H2 semi |
|---|---|---|---|---|---|
| 40 | 0.019 ± 0.014 | 0.108 ± 0.065 | 0.010 ± 0.009 | 5.87e-04 ± 6.13e-04 | 1.17e-03 ± 1.09e-03 |
| 50 | 0.026 ± 0.018 | 0.115 ± 0.042 | 0.016 ± 0.016 | 8.48e-04 ± 8.87e-04 | 1.54e-03 ± 1.42e-03 |
| 60 | 0.031 ± 0.022 | 0.139 ± 0.051 | 0.024 ± 0.023 | 1.05e-03 ± 1.01e-03 | 1.79e-03 ± 1.51e-03 |
| 70 | 0.036 ± 0.024 | 0.164 ± 0.062 | 0.030 ± 0.028 | 1.17e-03 ± 1.10e-03 | 1.89e-03 ± 1.58e-03 |
| 80 | 0.040 ± 0.026 | 0.216 ± 0.098 | 0.036 ± 0.032 | 1.27e-03 ± 1.15e-03 | 2.01e-03 ± 1.61e-03 |
| 90 | 0.043 ± 0.027 | 0.289 ± 0.156 | 0.038 ± 0.030 | 1.24e-03 ± 9.96e-04 | 1.88e-03 ± 1.31e-03 |
