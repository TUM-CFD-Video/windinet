| Model | RMSE | VRMSE | 1 – SSIM | MLW | H1 semi | H2 semi |
|---|---|---|---|---|---|---|
| shockwave_dit_jupiter_srba_loss_mlw0p05 | 0.161 ± 0.056 | 0.846 ± 0.126 | 0.088 ± 0.041 | 165.5 ± 19.1 | 1.65e-03 ± 1.06e-03 | 1.89e-03 ± 1.26e-03 |
| shockwave_dit_jupiter_srba_loss_mlw0p05: its VAE round trip | 0.104 ± 0.026 | 0.429 ± 0.038 | 0.034 ± 0.017 | 141.4 ± 21.8 | 9.42e-04 ± 6.39e-04 | 1.62e-03 ± 1.14e-03 |

shockwave_dit_jupiter_srba_loss_mlw0p05: frames [40, 50, 60, 70, 80, 90], clamp=True
shockwave_dit_jupiter_srba_loss_mlw0p05: generation per trajectory median 0.30 s (range 0.30-1.25), mean 0.32 s over 60 sims (19 s total); per generated frame 0.003 s (median / 96 frames); NVIDIA GH200 120GB, 5 steps, batch 1

shockwave_dit_jupiter_srba_loss_mlw0p05 per frame (VAE+DiT, mean ± std over sims)
| Frame | RMSE | VRMSE | 1 – SSIM | H1 semi | H2 semi |
|---|---|---|---|---|---|
| 40 | 0.143 ± 0.047 | 0.867 ± 0.377 | 0.064 ± 0.033 | 1.48e-03 ± 1.06e-03 | 1.69e-03 ± 1.22e-03 |
| 50 | 0.158 ± 0.052 | 0.816 ± 0.104 | 0.079 ± 0.039 | 1.68e-03 ± 1.15e-03 | 1.87e-03 ± 1.35e-03 |
| 60 | 0.164 ± 0.056 | 0.842 ± 0.117 | 0.090 ± 0.043 | 1.73e-03 ± 1.10e-03 | 1.95e-03 ± 1.30e-03 |
| 70 | 0.165 ± 0.059 | 0.895 ± 0.172 | 0.098 ± 0.046 | 1.73e-03 ± 1.09e-03 | 1.97e-03 ± 1.33e-03 |
| 80 | 0.168 ± 0.063 | 1.164 ± 0.513 | 0.102 ± 0.048 | 1.76e-03 ± 1.09e-03 | 2.04e-03 ± 1.33e-03 |
| 90 | 0.164 ± 0.060 | 1.507 ± 0.958 | 0.094 ± 0.040 | 1.55e-03 ± 9.06e-04 | 1.82e-03 ± 1.12e-03 |
