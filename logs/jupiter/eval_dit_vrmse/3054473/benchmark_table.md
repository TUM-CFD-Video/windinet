
### benchmark

| Model | RMSE | VRMSE | 1 – SSIM | MLW | H1 rel | H2 rel |
|---|---|---|---|---|---|---|
| shockwave_dit_jupiter_srba_s2_flip_vrmse_h1_1000 | 0.032 ± 0.022 | 0.141 ± 0.051 | 0.023 ± 0.022 | 101.9 ± 21.2 | 0.699 ± 0.122 | 0.925 ± 0.121 |
| shockwave_dit_jupiter_srba_s2_flip_vrmse_h1_1000: its VAE round trip | 0.015 ± 0.009 | 0.075 ± 0.023 | 0.011 ± 0.009 | 101.6 ± 23.7 | 0.548 ± 0.086 | 0.801 ± 0.113 |

shockwave_dit_jupiter_srba_s2_flip_vrmse_h1_1000: frames [40, 50, 60, 70, 80, 90], clamp=True

### benchmark_10

| Model | RMSE | VRMSE | 1 – SSIM | MLW | H1 rel | H2 rel |
|---|---|---|---|---|---|---|
| shockwave_dit_jupiter_srba_s2_flip_vrmse_h1_1000 | 0.026 ± 0.018 | 0.126 ± 0.042 | 0.016 ± 0.015 | 123.9 ± 17.1 | 0.643 ± 0.110 | 0.866 ± 0.114 |
| shockwave_dit_jupiter_srba_s2_flip_vrmse_h1_1000: its VAE round trip | 0.013 ± 0.008 | 0.067 ± 0.018 | 0.008 ± 0.007 | 127.0 ± 18.7 | 0.512 ± 0.081 | 0.760 ± 0.108 |

shockwave_dit_jupiter_srba_s2_flip_vrmse_h1_1000: frames [10, 20, 30, 40, 50, 60, 70, 80, 90], clamp=True
shockwave_dit_jupiter_srba_s2_flip_vrmse_h1_1000: generation per trajectory median 0.28 s (range 0.28-1.23), mean 0.30 s over 60 sims (18 s total); per generated frame 0.006 s (median / 48 frames); NVIDIA GH200 120GB, 5 steps, batch 1

shockwave_dit_jupiter_srba_s2_flip_vrmse_h1_1000 per frame (VAE+DiT, mean ± std over sims)
| Frame | RMSE | VRMSE | 1 – SSIM | H1 rel | H2 rel |
|---|---|---|---|---|---|
| 10 | 0.005 ± 0.002 | 3.679 ± 2.705 | 0.001 ± 0.001 | 5.136 ± 3.685 | 9.395 ± 6.947 |
| 20 | 0.007 ± 0.004 | 1.257 ± 2.139 | 0.002 ± 0.002 | 1.760 ± 2.443 | 3.099 ± 4.465 |
| 30 | 0.011 ± 0.007 | 0.637 ± 1.572 | 0.005 ± 0.004 | 1.088 ± 1.614 | 1.857 ± 3.048 |
| 40 | 0.018 ± 0.013 | 0.104 ± 0.048 | 0.009 ± 0.009 | 0.576 ± 0.108 | 0.825 ± 0.121 |
| 50 | 0.024 ± 0.018 | 0.110 ± 0.038 | 0.015 ± 0.015 | 0.642 ± 0.115 | 0.890 ± 0.105 |
| 60 | 0.029 ± 0.021 | 0.127 ± 0.049 | 0.021 ± 0.021 | 0.681 ± 0.128 | 0.915 ± 0.128 |
| 70 | 0.034 ± 0.024 | 0.152 ± 0.062 | 0.027 ± 0.026 | 0.719 ± 0.134 | 0.948 ± 0.130 |
| 80 | 0.038 ± 0.026 | 0.202 ± 0.097 | 0.033 ± 0.030 | 0.743 ± 0.137 | 0.963 ± 0.145 |
| 90 | 0.041 ± 0.027 | 0.276 ± 0.161 | 0.035 ± 0.029 | 0.826 ± 0.161 | 1.058 ± 0.156 |
