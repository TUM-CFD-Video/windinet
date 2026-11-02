| Model | RMSE | VRMSE | 1 – SSIM | MLW | H1 semi | H2 semi |
|---|---|---|---|---|---|---|
| shockwave_dit_jupiter_latnoise_1e1_s44 | 0.052 ± 0.039 | 0.676 ± 0.441 | 0.037 ± 0.037 | 64.5 ± 9.3 | 2.38e-04 ± 3.65e-04 | 1.60e-04 ± 2.72e-04 |
| shockwave_dit_jupiter_latnoise_1e1_s44: its VAE round trip | 0.007 ± 0.005 | 0.137 ± 0.771 | 0.005 ± 0.007 | 39.7 ± 13.1 | 6.41e-05 ± 1.15e-04 | 9.75e-05 ± 1.78e-04 |

shockwave_dit_jupiter_latnoise_1e1_s44: frames [40, 50, 60, 70, 80, 90], clamp=True
shockwave_dit_jupiter_latnoise_1e1_s44: generation per trajectory median 0.32 s (range 0.31-0.98), mean 0.32 s over 500 sims (161 s total); per generated frame 0.003 s (median / 96 frames); NVIDIA GH200 120GB, 5 steps, batch 1

shockwave_dit_jupiter_latnoise_1e1_s44 per frame (VAE+DiT, mean ± std over sims)
| Frame | RMSE | VRMSE | 1 – SSIM | H1 semi | H2 semi |
|---|---|---|---|---|---|
| 40 | 0.053 ± 0.037 | 0.580 ± 0.437 | 0.046 ± 0.043 | 3.44e-04 ± 5.05e-04 | 2.39e-04 ± 3.78e-04 |
| 50 | 0.052 ± 0.038 | 0.737 ± 2.388 | 0.042 ± 0.041 | 2.89e-04 ± 4.50e-04 | 1.96e-04 ± 3.35e-04 |
| 60 | 0.052 ± 0.040 | 0.741 ± 0.858 | 0.038 ± 0.040 | 2.49e-04 ± 4.09e-04 | 1.68e-04 ± 3.02e-04 |
| 70 | 0.051 ± 0.040 | 1.164 ± 8.363 | 0.034 ± 0.038 | 2.11e-04 ± 3.56e-04 | 1.41e-04 ± 2.66e-04 |
| 80 | 0.051 ± 0.041 | 1.307 ± 8.661 | 0.031 ± 0.035 | 1.77e-04 ± 2.99e-04 | 1.17e-04 ± 2.21e-04 |
| 90 | 0.052 ± 0.042 | 1.474 ± 9.478 | 0.029 ± 0.033 | 1.56e-04 ± 2.64e-04 | 1.02e-04 ± 1.96e-04 |
