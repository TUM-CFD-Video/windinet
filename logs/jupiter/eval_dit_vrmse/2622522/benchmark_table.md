| Model | RMSE | VRMSE | 1 – SSIM | MLW | H1 semi | H2 semi |
|---|---|---|---|---|---|---|
| shockwave_dit_jupiter_latnoise_1e5_s44 | 0.048 ± 0.035 | 0.630 ± 0.444 | 0.036 ± 0.036 | 62.6 ± 9.5 | 2.38e-04 ± 3.66e-04 | 1.61e-04 ± 2.73e-04 |
| shockwave_dit_jupiter_latnoise_1e5_s44: its VAE round trip | 0.006 ± 0.005 | 0.141 ± 0.857 | 0.005 ± 0.007 | 39.3 ± 13.5 | 6.33e-05 ± 1.14e-04 | 9.64e-05 ± 1.77e-04 |

shockwave_dit_jupiter_latnoise_1e5_s44: frames [40, 50, 60, 70, 80, 90], clamp=True
shockwave_dit_jupiter_latnoise_1e5_s44: generation per trajectory median 0.31 s (range 0.30-1.07), mean 0.31 s over 500 sims (157 s total); per generated frame 0.003 s (median / 96 frames); NVIDIA GH200 120GB, 5 steps, batch 1

shockwave_dit_jupiter_latnoise_1e5_s44 per frame (VAE+DiT, mean ± std over sims)
| Frame | RMSE | VRMSE | 1 – SSIM | H1 semi | H2 semi |
|---|---|---|---|---|---|
| 40 | 0.050 ± 0.034 | 0.565 ± 0.684 | 0.045 ± 0.041 | 3.43e-04 ± 5.05e-04 | 2.40e-04 ± 3.79e-04 |
| 50 | 0.048 ± 0.035 | 0.653 ± 1.462 | 0.040 ± 0.040 | 2.88e-04 ± 4.51e-04 | 1.96e-04 ± 3.36e-04 |
| 60 | 0.048 ± 0.037 | 0.677 ± 0.630 | 0.037 ± 0.039 | 2.51e-04 ± 4.15e-04 | 1.69e-04 ± 3.04e-04 |
| 70 | 0.047 ± 0.037 | 0.943 ± 4.925 | 0.033 ± 0.037 | 2.12e-04 ± 3.59e-04 | 1.42e-04 ± 2.67e-04 |
| 80 | 0.047 ± 0.037 | 1.077 ± 5.400 | 0.030 ± 0.034 | 1.78e-04 ± 2.99e-04 | 1.18e-04 ± 2.22e-04 |
| 90 | 0.047 ± 0.037 | 1.227 ± 6.191 | 0.028 ± 0.032 | 1.57e-04 ± 2.63e-04 | 1.02e-04 ± 1.97e-04 |
