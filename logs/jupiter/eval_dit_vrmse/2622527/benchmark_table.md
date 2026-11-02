| Model | RMSE | VRMSE | 1 – SSIM | MLW | H1 semi | H2 semi |
|---|---|---|---|---|---|---|
| shockwave_dit_jupiter_latnoise_1e4_s44 | 0.049 ± 0.036 | 0.634 ± 0.439 | 0.036 ± 0.036 | 62.6 ± 9.4 | 2.40e-04 ± 3.68e-04 | 1.62e-04 ± 2.74e-04 |
| shockwave_dit_jupiter_latnoise_1e4_s44: its VAE round trip | 0.006 ± 0.005 | 0.142 ± 0.876 | 0.005 ± 0.007 | 39.2 ± 13.5 | 6.31e-05 ± 1.14e-04 | 9.62e-05 ± 1.77e-04 |

shockwave_dit_jupiter_latnoise_1e4_s44: frames [40, 50, 60, 70, 80, 90], clamp=True
shockwave_dit_jupiter_latnoise_1e4_s44: generation per trajectory median 0.31 s (range 0.30-1.06), mean 0.31 s over 500 sims (154 s total); per generated frame 0.003 s (median / 96 frames); NVIDIA GH200 120GB, 5 steps, batch 1

shockwave_dit_jupiter_latnoise_1e4_s44 per frame (VAE+DiT, mean ± std over sims)
| Frame | RMSE | VRMSE | 1 – SSIM | H1 semi | H2 semi |
|---|---|---|---|---|---|
| 40 | 0.050 ± 0.035 | 0.564 ± 0.622 | 0.045 ± 0.042 | 3.46e-04 ± 5.05e-04 | 2.41e-04 ± 3.79e-04 |
| 50 | 0.049 ± 0.036 | 0.690 ± 2.160 | 0.041 ± 0.040 | 2.92e-04 ± 4.55e-04 | 1.98e-04 ± 3.38e-04 |
| 60 | 0.048 ± 0.037 | 0.693 ± 0.799 | 0.037 ± 0.039 | 2.53e-04 ± 4.15e-04 | 1.69e-04 ± 3.04e-04 |
| 70 | 0.047 ± 0.037 | 1.061 ± 7.449 | 0.034 ± 0.037 | 2.13e-04 ± 3.61e-04 | 1.43e-04 ± 2.68e-04 |
| 80 | 0.047 ± 0.037 | 1.192 ± 7.911 | 0.031 ± 0.034 | 1.80e-04 ± 3.01e-04 | 1.18e-04 ± 2.22e-04 |
| 90 | 0.047 ± 0.038 | 1.322 ± 8.688 | 0.028 ± 0.032 | 1.57e-04 ± 2.66e-04 | 1.02e-04 ± 1.96e-04 |
