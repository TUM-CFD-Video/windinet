| Model | RMSE | VRMSE | 1 – SSIM | MLW | H1 semi | H2 semi |
|---|---|---|---|---|---|---|
| shockwave_dit_jupiter_latnoise_1e2_s44 | 0.049 ± 0.036 | 0.630 ± 0.396 | 0.036 ± 0.036 | 62.8 ± 9.5 | 2.39e-04 ± 3.69e-04 | 1.61e-04 ± 2.74e-04 |
| shockwave_dit_jupiter_latnoise_1e2_s44: its VAE round trip | 0.006 ± 0.005 | 0.137 ± 0.800 | 0.005 ± 0.007 | 39.0 ± 13.3 | 6.30e-05 ± 1.13e-04 | 9.61e-05 ± 1.77e-04 |

shockwave_dit_jupiter_latnoise_1e2_s44: frames [40, 50, 60, 70, 80, 90], clamp=True
shockwave_dit_jupiter_latnoise_1e2_s44: generation per trajectory median 0.31 s (range 0.29-1.19), mean 0.32 s over 500 sims (160 s total); per generated frame 0.003 s (median / 96 frames); NVIDIA GH200 120GB, 5 steps, batch 1

shockwave_dit_jupiter_latnoise_1e2_s44 per frame (VAE+DiT, mean ± std over sims)
| Frame | RMSE | VRMSE | 1 – SSIM | H1 semi | H2 semi |
|---|---|---|---|---|---|
| 40 | 0.051 ± 0.035 | 0.565 ± 0.587 | 0.045 ± 0.042 | 3.45e-04 ± 5.09e-04 | 2.41e-04 ± 3.82e-04 |
| 50 | 0.049 ± 0.036 | 0.646 ± 1.224 | 0.041 ± 0.041 | 2.90e-04 ± 4.55e-04 | 1.97e-04 ± 3.39e-04 |
| 60 | 0.048 ± 0.038 | 0.676 ± 0.584 | 0.037 ± 0.040 | 2.52e-04 ± 4.16e-04 | 1.69e-04 ± 3.03e-04 |
| 70 | 0.048 ± 0.038 | 0.906 ± 4.067 | 0.034 ± 0.037 | 2.13e-04 ± 3.63e-04 | 1.42e-04 ± 2.67e-04 |
| 80 | 0.047 ± 0.038 | 1.043 ± 4.625 | 0.031 ± 0.034 | 1.78e-04 ± 3.00e-04 | 1.17e-04 ± 2.21e-04 |
| 90 | 0.048 ± 0.039 | 1.175 ± 5.148 | 0.028 ± 0.032 | 1.56e-04 ± 2.64e-04 | 1.02e-04 ± 1.96e-04 |
