| Model | RMSE | VRMSE | 1 – SSIM | MLW | H1 semi | H2 semi |
|---|---|---|---|---|---|---|
| shockwave_dit_jupiter_srba_loss_h1_0 | 0.033 ± 0.022 | 0.149 ± 0.050 | 0.026 ± 0.023 | 105.3 ± 22.9 | 1.01e-03 ± 9.19e-04 | 1.70e-03 ± 1.35e-03 |
| shockwave_dit_jupiter_srba_loss_h1_0: its VAE round trip | 0.015 ± 0.008 | 0.079 ± 0.018 | 0.012 ± 0.009 | 101.5 ± 24.7 | 5.82e-04 ± 4.81e-04 | 1.31e-03 ± 9.96e-04 |

shockwave_dit_jupiter_srba_loss_h1_0: frames [40, 50, 60, 70, 80, 90], clamp=True
shockwave_dit_jupiter_srba_loss_h1_0: generation per trajectory median 0.32 s (range 0.29-1.01), mean 0.33 s over 60 sims (20 s total); per generated frame 0.003 s (median / 96 frames); NVIDIA GH200 120GB, 5 steps, batch 1

shockwave_dit_jupiter_srba_loss_h1_0 per frame (VAE+DiT, mean ± std over sims)
| Frame | RMSE | VRMSE | 1 – SSIM | H1 semi | H2 semi |
|---|---|---|---|---|---|
| 40 | 0.019 ± 0.013 | 0.112 ± 0.058 | 0.011 ± 0.009 | 6.21e-04 ± 5.84e-04 | 1.27e-03 ± 1.06e-03 |
| 50 | 0.025 ± 0.018 | 0.117 ± 0.040 | 0.017 ± 0.016 | 8.52e-04 ± 8.49e-04 | 1.57e-03 ± 1.36e-03 |
| 60 | 0.031 ± 0.021 | 0.136 ± 0.050 | 0.024 ± 0.023 | 1.01e-03 ± 9.64e-04 | 1.73e-03 ± 1.43e-03 |
| 70 | 0.035 ± 0.024 | 0.160 ± 0.061 | 0.030 ± 0.028 | 1.12e-03 ± 1.08e-03 | 1.83e-03 ± 1.52e-03 |
| 80 | 0.039 ± 0.026 | 0.213 ± 0.095 | 0.036 ± 0.031 | 1.22e-03 ± 1.10e-03 | 1.95e-03 ± 1.52e-03 |
| 90 | 0.042 ± 0.027 | 0.285 ± 0.154 | 0.038 ± 0.030 | 1.21e-03 ± 9.79e-04 | 1.83e-03 ± 1.27e-03 |
