
### benchmark

| Model | RMSE | VRMSE | 1 – SSIM | MLW | H1 rel | H2 rel |
|---|---|---|---|---|---|---|
| shockwave_dit_jupiter_srba_s2_flip_vrmse_h1_100 | 0.030 ± 0.021 | 0.132 ± 0.050 | 0.023 ± 0.021 | 99.2 ± 19.5 | 0.678 ± 0.120 | 0.904 ± 0.111 |
| shockwave_dit_jupiter_srba_s2_flip_vrmse_h1_100: its VAE round trip | 0.016 ± 0.009 | 0.075 ± 0.021 | 0.011 ± 0.010 | 94.5 ± 20.5 | 0.560 ± 0.076 | 0.813 ± 0.092 |

shockwave_dit_jupiter_srba_s2_flip_vrmse_h1_100: frames [40, 50, 60, 70, 80, 90], clamp=True

### benchmark_10

| Model | RMSE | VRMSE | 1 – SSIM | MLW | H1 rel | H2 rel |
|---|---|---|---|---|---|---|
| shockwave_dit_jupiter_srba_s2_flip_vrmse_h1_100 | 0.025 ± 0.018 | 0.118 ± 0.041 | 0.016 ± 0.015 | 121.2 ± 15.0 | 0.628 ± 0.109 | 0.853 ± 0.106 |
| shockwave_dit_jupiter_srba_s2_flip_vrmse_h1_100: its VAE round trip | 0.013 ± 0.007 | 0.068 ± 0.016 | 0.009 ± 0.007 | 116.3 ± 15.7 | 0.528 ± 0.072 | 0.777 ± 0.089 |

shockwave_dit_jupiter_srba_s2_flip_vrmse_h1_100: frames [10, 20, 30, 40, 50, 60, 70, 80, 90], clamp=True
shockwave_dit_jupiter_srba_s2_flip_vrmse_h1_100: generation per trajectory median 0.30 s (range 0.29-1.07), mean 0.32 s over 60 sims (19 s total); per generated frame 0.006 s (median / 48 frames); NVIDIA GH200 120GB, 5 steps, batch 1

shockwave_dit_jupiter_srba_s2_flip_vrmse_h1_100 per frame (VAE+DiT, mean ± std over sims)
| Frame | RMSE | VRMSE | 1 – SSIM | H1 rel | H2 rel |
|---|---|---|---|---|---|
| 10 | 0.005 ± 0.002 | 3.502 ± 2.565 | 0.002 ± 0.001 | 4.793 ± 3.390 | 8.522 ± 6.205 |
| 20 | 0.008 ± 0.004 | 1.157 ± 1.936 | 0.003 ± 0.002 | 1.722 ± 2.294 | 2.940 ± 4.086 |
| 30 | 0.012 ± 0.007 | 0.568 ± 1.360 | 0.005 ± 0.004 | 1.078 ± 1.571 | 1.789 ± 2.849 |
| 40 | 0.017 ± 0.012 | 0.100 ± 0.048 | 0.009 ± 0.009 | 0.576 ± 0.100 | 0.825 ± 0.111 |
| 50 | 0.023 ± 0.017 | 0.104 ± 0.036 | 0.014 ± 0.015 | 0.626 ± 0.112 | 0.873 ± 0.099 |
| 60 | 0.028 ± 0.020 | 0.120 ± 0.047 | 0.021 ± 0.021 | 0.667 ± 0.121 | 0.901 ± 0.115 |
| 70 | 0.032 ± 0.024 | 0.144 ± 0.060 | 0.027 ± 0.026 | 0.701 ± 0.131 | 0.930 ± 0.122 |
| 80 | 0.036 ± 0.025 | 0.188 ± 0.095 | 0.031 ± 0.030 | 0.715 ± 0.136 | 0.934 ± 0.134 |
| 90 | 0.039 ± 0.026 | 0.257 ± 0.154 | 0.034 ± 0.029 | 0.791 ± 0.168 | 1.026 ± 0.172 |
