
### benchmark

| Model | RMSE | VRMSE | 1 – SSIM | MLW | H1 rel | H2 rel |
|---|---|---|---|---|---|---|
| shockwave_dit_jupiter_srba_s2_flip_vrmse_h1_300 | 0.030 ± 0.022 | 0.134 ± 0.052 | 0.023 ± 0.022 | 100.4 ± 20.5 | 0.685 ± 0.128 | 0.916 ± 0.122 |
| shockwave_dit_jupiter_srba_s2_flip_vrmse_h1_300: its VAE round trip | 0.015 ± 0.009 | 0.074 ± 0.022 | 0.011 ± 0.010 | 97.4 ± 21.9 | 0.557 ± 0.081 | 0.814 ± 0.103 |

shockwave_dit_jupiter_srba_s2_flip_vrmse_h1_300: frames [40, 50, 60, 70, 80, 90], clamp=True

### benchmark_10

| Model | RMSE | VRMSE | 1 – SSIM | MLW | H1 rel | H2 rel |
|---|---|---|---|---|---|---|
| shockwave_dit_jupiter_srba_s2_flip_vrmse_h1_300 | 0.025 ± 0.018 | 0.119 ± 0.042 | 0.016 ± 0.015 | 120.8 ± 16.2 | 0.632 ± 0.115 | 0.860 ± 0.114 |
| shockwave_dit_jupiter_srba_s2_flip_vrmse_h1_300: its VAE round trip | 0.013 ± 0.007 | 0.067 ± 0.018 | 0.008 ± 0.007 | 119.8 ± 16.6 | 0.522 ± 0.077 | 0.774 ± 0.099 |

shockwave_dit_jupiter_srba_s2_flip_vrmse_h1_300: frames [10, 20, 30, 40, 50, 60, 70, 80, 90], clamp=True
shockwave_dit_jupiter_srba_s2_flip_vrmse_h1_300: generation per trajectory median 0.28 s (range 0.27-0.72), mean 0.28 s over 60 sims (17 s total); per generated frame 0.006 s (median / 48 frames); NVIDIA GH200 120GB, 5 steps, batch 1

shockwave_dit_jupiter_srba_s2_flip_vrmse_h1_300 per frame (VAE+DiT, mean ± std over sims)
| Frame | RMSE | VRMSE | 1 – SSIM | H1 rel | H2 rel |
|---|---|---|---|---|---|
| 10 | 0.005 ± 0.002 | 3.400 ± 2.512 | 0.001 ± 0.001 | 4.977 ± 3.568 | 9.025 ± 6.640 |
| 20 | 0.007 ± 0.004 | 1.114 ± 1.869 | 0.002 ± 0.002 | 1.720 ± 2.340 | 2.996 ± 4.230 |
| 30 | 0.011 ± 0.007 | 0.544 ± 1.300 | 0.005 ± 0.004 | 1.078 ± 1.589 | 1.824 ± 2.954 |
| 40 | 0.017 ± 0.013 | 0.099 ± 0.049 | 0.009 ± 0.009 | 0.572 ± 0.110 | 0.823 ± 0.120 |
| 50 | 0.023 ± 0.018 | 0.104 ± 0.040 | 0.015 ± 0.015 | 0.629 ± 0.123 | 0.880 ± 0.111 |
| 60 | 0.028 ± 0.021 | 0.120 ± 0.050 | 0.021 ± 0.022 | 0.667 ± 0.135 | 0.905 ± 0.130 |
| 70 | 0.033 ± 0.024 | 0.145 ± 0.062 | 0.027 ± 0.026 | 0.708 ± 0.140 | 0.940 ± 0.133 |
| 80 | 0.036 ± 0.026 | 0.192 ± 0.098 | 0.032 ± 0.030 | 0.727 ± 0.143 | 0.951 ± 0.145 |
| 90 | 0.039 ± 0.027 | 0.262 ± 0.157 | 0.034 ± 0.029 | 0.809 ± 0.165 | 1.048 ± 0.165 |
