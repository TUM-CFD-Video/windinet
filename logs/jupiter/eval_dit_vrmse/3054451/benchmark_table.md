
### benchmark

| Model | RMSE | VRMSE | 1 – SSIM | MLW | H1 rel | H2 rel |
|---|---|---|---|---|---|---|
| shockwave_dit_jupiter_srba_s2_flip_vrmse_h1_10 | 0.033 ± 0.032 | 0.146 ± 0.109 | 0.026 ± 0.027 | 108.7 ± 23.1 | 0.713 ± 0.121 | 0.942 ± 0.107 |
| shockwave_dit_jupiter_srba_s2_flip_vrmse_h1_10: its VAE round trip | 0.020 ± 0.011 | 0.092 ± 0.024 | 0.017 ± 0.014 | 106.4 ± 24.0 | 0.605 ± 0.073 | 0.846 ± 0.083 |

shockwave_dit_jupiter_srba_s2_flip_vrmse_h1_10: frames [40, 50, 60, 70, 80, 90], clamp=True

### benchmark_10

| Model | RMSE | VRMSE | 1 – SSIM | MLW | H1 rel | H2 rel |
|---|---|---|---|---|---|---|
| shockwave_dit_jupiter_srba_s2_flip_vrmse_h1_10 | 0.028 ± 0.027 | 0.131 ± 0.086 | 0.019 ± 0.019 | 132.9 ± 18.3 | 0.668 ± 0.111 | 0.896 ± 0.098 |
| shockwave_dit_jupiter_srba_s2_flip_vrmse_h1_10: its VAE round trip | 0.017 ± 0.009 | 0.084 ± 0.019 | 0.013 ± 0.010 | 131.6 ± 19.0 | 0.575 ± 0.068 | 0.814 ± 0.079 |

shockwave_dit_jupiter_srba_s2_flip_vrmse_h1_10: frames [10, 20, 30, 40, 50, 60, 70, 80, 90], clamp=True
shockwave_dit_jupiter_srba_s2_flip_vrmse_h1_10: generation per trajectory median 0.31 s (range 0.29-1.04), mean 0.31 s over 60 sims (19 s total); per generated frame 0.006 s (median / 48 frames); NVIDIA GH200 120GB, 5 steps, batch 1

shockwave_dit_jupiter_srba_s2_flip_vrmse_h1_10 per frame (VAE+DiT, mean ± std over sims)
| Frame | RMSE | VRMSE | 1 – SSIM | H1 rel | H2 rel |
|---|---|---|---|---|---|
| 10 | 0.007 ± 0.002 | 4.474 ± 3.520 | 0.003 ± 0.002 | 6.679 ± 5.300 | 12.106 ± 10.047 |
| 20 | 0.010 ± 0.007 | 1.517 ± 2.743 | 0.005 ± 0.004 | 2.061 ± 2.864 | 3.506 ± 5.145 |
| 30 | 0.015 ± 0.015 | 0.776 ± 1.935 | 0.008 ± 0.010 | 1.407 ± 2.458 | 2.367 ± 4.564 |
| 40 | 0.020 ± 0.022 | 0.119 ± 0.097 | 0.012 ± 0.016 | 0.618 ± 0.113 | 0.863 ± 0.104 |
| 50 | 0.025 ± 0.026 | 0.117 ± 0.084 | 0.017 ± 0.022 | 0.655 ± 0.116 | 0.898 ± 0.094 |
| 60 | 0.031 ± 0.031 | 0.135 ± 0.102 | 0.024 ± 0.027 | 0.701 ± 0.128 | 0.931 ± 0.115 |
| 70 | 0.035 ± 0.035 | 0.160 ± 0.132 | 0.030 ± 0.032 | 0.743 ± 0.132 | 0.973 ± 0.121 |
| 80 | 0.039 ± 0.038 | 0.220 ± 0.248 | 0.035 ± 0.035 | 0.755 ± 0.133 | 0.978 ± 0.135 |
| 90 | 0.041 ± 0.038 | 0.298 ± 0.361 | 0.036 ± 0.032 | 0.833 ± 0.174 | 1.082 ± 0.169 |
