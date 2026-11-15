
### benchmark

| Model | RMSE | VRMSE | 1 – SSIM | MLW | H1 rel | H2 rel |
|---|---|---|---|---|---|---|
| shockwave_dit_jupiter_clip33_any_s44_cond9 | 0.045 ± 0.033 | 0.573 ± 0.266 | 0.034 ± 0.035 | 61.1 ± 9.0 | 1.096 ± 0.133 | 1.209 ± 0.335 |
| shockwave_dit_jupiter_clip33_any_s44_cond9: its VAE round trip | 0.006 ± 0.005 | 0.099 ± 0.320 | 0.004 ± 0.006 | 36.5 ± 11.5 | 0.577 ± 0.235 | 0.996 ± 0.440 |

shockwave_dit_jupiter_clip33_any_s44_cond9: frames [40, 50, 60, 70, 80, 90], clamp=True

### benchmark_10

| Model | RMSE | VRMSE | 1 – SSIM | MLW | H1 rel | H2 rel |
|---|---|---|---|---|---|---|
| shockwave_dit_jupiter_clip33_any_s44_cond9 | 0.044 ± 0.030 | 0.388 ± 0.105 | 0.035 ± 0.032 | 63.6 ± 11.0 | 0.956 ± 0.062 | 1.067 ± 0.103 |
| shockwave_dit_jupiter_clip33_any_s44_cond9: its VAE round trip | 0.007 ± 0.006 | 0.070 ± 0.030 | 0.006 ± 0.007 | 42.5 ± 13.6 | 0.531 ± 0.070 | 0.843 ± 0.128 |

shockwave_dit_jupiter_clip33_any_s44_cond9: frames [10, 20, 30, 40, 50, 60, 70, 80, 90], clamp=True
shockwave_dit_jupiter_clip33_any_s44_cond9: generation per trajectory median 0.30 s (range 0.29-1.31), mean 0.30 s over 500 sims (152 s total); per generated frame 0.003 s (median / 96 frames); NVIDIA GH200 120GB, 5 steps, batch 1

shockwave_dit_jupiter_clip33_any_s44_cond9 per frame (VAE+DiT, mean ± std over sims)
| Frame | RMSE | VRMSE | 1 – SSIM | H1 rel | H2 rel |
|---|---|---|---|---|---|
| 10 | 0.029 ± 0.020 | 0.200 ± 0.063 | 0.024 ± 0.025 | 0.722 ± 0.106 | 0.960 ± 0.119 |
| 20 | 0.040 ± 0.028 | 0.354 ± 0.670 | 0.041 ± 0.037 | 0.976 ± 0.212 | 1.117 ± 0.227 |
| 30 | 0.042 ± 0.030 | 0.438 ± 0.945 | 0.043 ± 0.040 | 1.049 ± 0.202 | 1.155 ± 0.236 |
| 40 | 0.043 ± 0.031 | 0.461 ± 0.227 | 0.041 ± 0.040 | 1.072 ± 0.142 | 1.192 ± 0.311 |
| 50 | 0.044 ± 0.033 | 0.632 ± 2.283 | 0.038 ± 0.039 | 1.105 ± 0.242 | 1.259 ± 0.598 |
| 60 | 0.045 ± 0.034 | 0.633 ± 0.690 | 0.036 ± 0.038 | 1.137 ± 0.353 | 1.342 ± 0.909 |
| 70 | 0.045 ± 0.035 | 0.988 ± 6.975 | 0.033 ± 0.036 | 1.163 ± 0.448 | 1.427 ± 1.188 |
| 80 | 0.045 ± 0.035 | 1.137 ± 7.912 | 0.030 ± 0.034 | 1.230 ± 0.774 | 1.599 ± 1.676 |
| 90 | 0.046 ± 0.037 | 1.308 ± 8.834 | 0.028 ± 0.032 | 1.244 ± 0.599 | 1.727 ± 1.808 |
