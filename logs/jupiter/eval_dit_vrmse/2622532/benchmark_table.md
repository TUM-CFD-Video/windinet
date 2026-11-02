| Model | RMSE | VRMSE | 1 – SSIM | MLW | H1 semi | H2 semi |
|---|---|---|---|---|---|---|
| shockwave_dit_jupiter_latnoise_1e3_s44 | 0.048 ± 0.035 | 0.625 ± 0.493 | 0.036 ± 0.036 | 62.3 ± 9.6 | 2.38e-04 ± 3.67e-04 | 1.62e-04 ± 2.74e-04 |
| shockwave_dit_jupiter_latnoise_1e3_s44: its VAE round trip | 0.006 ± 0.005 | 0.138 ± 0.811 | 0.005 ± 0.007 | 39.3 ± 13.4 | 6.32e-05 ± 1.14e-04 | 9.63e-05 ± 1.77e-04 |

shockwave_dit_jupiter_latnoise_1e3_s44: frames [40, 50, 60, 70, 80, 90], clamp=True
shockwave_dit_jupiter_latnoise_1e3_s44: generation per trajectory median 0.31 s (range 0.30-1.15), mean 0.31 s over 500 sims (154 s total); per generated frame 0.003 s (median / 96 frames); NVIDIA GH200 120GB, 5 steps, batch 1

shockwave_dit_jupiter_latnoise_1e3_s44 per frame (VAE+DiT, mean ± std over sims)
| Frame | RMSE | VRMSE | 1 – SSIM | H1 semi | H2 semi |
|---|---|---|---|---|---|
| 40 | 0.049 ± 0.034 | 0.561 ± 0.775 | 0.045 ± 0.041 | 3.42e-04 ± 5.02e-04 | 2.40e-04 ± 3.79e-04 |
| 50 | 0.048 ± 0.035 | 0.645 ± 1.377 | 0.040 ± 0.040 | 2.89e-04 ± 4.55e-04 | 1.97e-04 ± 3.37e-04 |
| 60 | 0.047 ± 0.036 | 0.671 ± 0.645 | 0.037 ± 0.039 | 2.51e-04 ± 4.15e-04 | 1.69e-04 ± 3.05e-04 |
| 70 | 0.046 ± 0.036 | 0.912 ± 4.385 | 0.033 ± 0.036 | 2.12e-04 ± 3.59e-04 | 1.42e-04 ± 2.68e-04 |
| 80 | 0.046 ± 0.036 | 1.045 ± 4.843 | 0.030 ± 0.034 | 1.78e-04 ± 2.99e-04 | 1.18e-04 ± 2.22e-04 |
| 90 | 0.046 ± 0.037 | 1.191 ± 5.707 | 0.028 ± 0.032 | 1.56e-04 ± 2.63e-04 | 1.03e-04 ± 1.97e-04 |
