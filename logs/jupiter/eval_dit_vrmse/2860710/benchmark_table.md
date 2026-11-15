
### benchmark

| Model | RMSE | VRMSE | 1 – SSIM | MLW | H1 rel | H2 rel |
|---|---|---|---|---|---|---|
| shockwave_dit_jupiter_clip33_aligned_s44_cond9 | 0.045 ± 0.033 | 0.564 ± 0.263 | 0.034 ± 0.035 | 61.0 ± 9.1 | 1.092 ± 0.125 | 1.208 ± 0.338 |
| shockwave_dit_jupiter_clip33_aligned_s44_cond9: its VAE round trip | 0.006 ± 0.005 | 0.102 ± 0.354 | 0.004 ± 0.006 | 36.6 ± 11.6 | 0.582 ± 0.251 | 1.002 ± 0.465 |

shockwave_dit_jupiter_clip33_aligned_s44_cond9: frames [40, 50, 60, 70, 80, 90], clamp=True

### benchmark_10

| Model | RMSE | VRMSE | 1 – SSIM | MLW | H1 rel | H2 rel |
|---|---|---|---|---|---|---|
| shockwave_dit_jupiter_clip33_aligned_s44_cond9 | 0.043 ± 0.030 | 0.385 ± 0.106 | 0.035 ± 0.032 | 63.5 ± 11.0 | 0.957 ± 0.063 | 1.069 ± 0.107 |
| shockwave_dit_jupiter_clip33_aligned_s44_cond9: its VAE round trip | 0.007 ± 0.006 | 0.071 ± 0.030 | 0.006 ± 0.007 | 42.5 ± 13.7 | 0.531 ± 0.072 | 0.842 ± 0.131 |

shockwave_dit_jupiter_clip33_aligned_s44_cond9: frames [10, 20, 30, 40, 50, 60, 70, 80, 90], clamp=True
shockwave_dit_jupiter_clip33_aligned_s44_cond9: generation per trajectory median 0.29 s (range 0.28-1.03), mean 0.29 s over 500 sims (147 s total); per generated frame 0.003 s (median / 96 frames); NVIDIA GH200 120GB, 5 steps, batch 1

shockwave_dit_jupiter_clip33_aligned_s44_cond9 per frame (VAE+DiT, mean ± std over sims)
| Frame | RMSE | VRMSE | 1 – SSIM | H1 rel | H2 rel |
|---|---|---|---|---|---|
| 10 | 0.029 ± 0.020 | 0.203 ± 0.064 | 0.024 ± 0.025 | 0.725 ± 0.112 | 0.964 ± 0.126 |
| 20 | 0.040 ± 0.028 | 0.348 ± 0.552 | 0.040 ± 0.037 | 0.975 ± 0.190 | 1.117 ± 0.194 |
| 30 | 0.042 ± 0.030 | 0.426 ± 0.790 | 0.043 ± 0.040 | 1.047 ± 0.179 | 1.154 ± 0.219 |
| 40 | 0.043 ± 0.031 | 0.456 ± 0.215 | 0.041 ± 0.040 | 1.069 ± 0.132 | 1.190 ± 0.300 |
| 50 | 0.044 ± 0.033 | 0.630 ± 2.427 | 0.038 ± 0.039 | 1.101 ± 0.249 | 1.259 ± 0.605 |
| 60 | 0.045 ± 0.034 | 0.625 ± 0.733 | 0.035 ± 0.038 | 1.129 ± 0.322 | 1.339 ± 0.907 |
| 70 | 0.045 ± 0.035 | 1.009 ± 7.656 | 0.032 ± 0.035 | 1.156 ± 0.446 | 1.427 ± 1.205 |
| 80 | 0.045 ± 0.035 | 1.147 ± 8.344 | 0.030 ± 0.033 | 1.225 ± 0.815 | 1.602 ± 1.706 |
| 90 | 0.045 ± 0.036 | 1.335 ± 9.278 | 0.027 ± 0.031 | 1.238 ± 0.602 | 1.732 ± 1.824 |
