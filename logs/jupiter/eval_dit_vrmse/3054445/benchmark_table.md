
### benchmark

| Model | RMSE | VRMSE | 1 – SSIM | MLW | H1 rel | H2 rel |
|---|---|---|---|---|---|---|
| shockwave_dit_jupiter_srba_s2_flip_vrmse | 0.030 ± 0.021 | 0.132 ± 0.049 | 0.023 ± 0.021 | 101.4 ± 20.4 | 0.673 ± 0.118 | 0.895 ± 0.107 |
| shockwave_dit_jupiter_srba_s2_flip_vrmse: its VAE round trip | 0.016 ± 0.009 | 0.078 ± 0.020 | 0.012 ± 0.010 | 100.1 ± 22.7 | 0.566 ± 0.071 | 0.816 ± 0.082 |

shockwave_dit_jupiter_srba_s2_flip_vrmse: frames [40, 50, 60, 70, 80, 90], clamp=True

### benchmark_10

| Model | RMSE | VRMSE | 1 – SSIM | MLW | H1 rel | H2 rel |
|---|---|---|---|---|---|---|
| shockwave_dit_jupiter_srba_s2_flip_vrmse | 0.025 ± 0.017 | 0.118 ± 0.040 | 0.016 ± 0.015 | 124.4 ± 16.2 | 0.626 ± 0.105 | 0.847 ± 0.099 |
| shockwave_dit_jupiter_srba_s2_flip_vrmse: its VAE round trip | 0.014 ± 0.008 | 0.071 ± 0.016 | 0.009 ± 0.008 | 124.8 ± 17.7 | 0.538 ± 0.067 | 0.784 ± 0.078 |

shockwave_dit_jupiter_srba_s2_flip_vrmse: frames [10, 20, 30, 40, 50, 60, 70, 80, 90], clamp=True
shockwave_dit_jupiter_srba_s2_flip_vrmse: generation per trajectory median 0.29 s (range 0.28-0.73), mean 0.30 s over 60 sims (18 s total); per generated frame 0.006 s (median / 48 frames); NVIDIA GH200 120GB, 5 steps, batch 1

shockwave_dit_jupiter_srba_s2_flip_vrmse per frame (VAE+DiT, mean ± std over sims)
| Frame | RMSE | VRMSE | 1 – SSIM | H1 rel | H2 rel |
|---|---|---|---|---|---|
| 10 | 0.006 ± 0.002 | 3.477 ± 2.641 | 0.002 ± 0.001 | 5.283 ± 3.959 | 9.701 ± 7.575 |
| 20 | 0.009 ± 0.004 | 1.132 ± 1.896 | 0.003 ± 0.002 | 1.818 ± 2.475 | 3.171 ± 4.586 |
| 30 | 0.012 ± 0.007 | 0.630 ± 1.572 | 0.005 ± 0.004 | 1.256 ± 2.132 | 2.165 ± 4.078 |
| 40 | 0.018 ± 0.012 | 0.103 ± 0.048 | 0.009 ± 0.009 | 0.578 ± 0.093 | 0.822 ± 0.101 |
| 50 | 0.023 ± 0.017 | 0.105 ± 0.036 | 0.015 ± 0.015 | 0.628 ± 0.108 | 0.870 ± 0.094 |
| 60 | 0.028 ± 0.020 | 0.119 ± 0.046 | 0.021 ± 0.021 | 0.660 ± 0.118 | 0.889 ± 0.109 |
| 70 | 0.032 ± 0.023 | 0.142 ± 0.059 | 0.027 ± 0.025 | 0.693 ± 0.127 | 0.916 ± 0.115 |
| 80 | 0.036 ± 0.025 | 0.188 ± 0.095 | 0.032 ± 0.029 | 0.710 ± 0.136 | 0.924 ± 0.130 |
| 90 | 0.039 ± 0.026 | 0.256 ± 0.154 | 0.034 ± 0.028 | 0.786 ± 0.170 | 1.020 ± 0.179 |
