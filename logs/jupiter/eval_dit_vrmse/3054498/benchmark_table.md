
### benchmark

| Model | RMSE | VRMSE | 1 – SSIM | MLW | H1 rel | H2 rel |
|---|---|---|---|---|---|---|
| shockwave_dit_jupiter_srba_s2_flip_sds_3e2 | 0.035 ± 0.021 | 0.163 ± 0.044 | 0.028 ± 0.022 | 111.3 ± 25.5 | 0.813 ± 0.098 | 1.057 ± 0.116 |
| shockwave_dit_jupiter_srba_s2_flip_sds_3e2: its VAE round trip | 0.015 ± 0.008 | 0.080 ± 0.020 | 0.012 ± 0.009 | 106.1 ± 26.1 | 0.615 ± 0.099 | 0.895 ± 0.149 |

shockwave_dit_jupiter_srba_s2_flip_sds_3e2: frames [40, 50, 60, 70, 80, 90], clamp=True

### benchmark_10

| Model | RMSE | VRMSE | 1 – SSIM | MLW | H1 rel | H2 rel |
|---|---|---|---|---|---|---|
| shockwave_dit_jupiter_srba_s2_flip_sds_3e2 | 0.030 ± 0.017 | 0.148 ± 0.036 | 0.020 ± 0.016 | 134.9 ± 20.4 | 0.753 ± 0.091 | 0.995 ± 0.113 |
| shockwave_dit_jupiter_srba_s2_flip_sds_3e2: its VAE round trip | 0.013 ± 0.007 | 0.072 ± 0.017 | 0.009 ± 0.007 | 132.1 ± 20.7 | 0.577 ± 0.093 | 0.851 ± 0.139 |

shockwave_dit_jupiter_srba_s2_flip_sds_3e2: frames [10, 20, 30, 40, 50, 60, 70, 80, 90], clamp=True
shockwave_dit_jupiter_srba_s2_flip_sds_3e2: generation per trajectory median 0.29 s (range 0.28-0.79), mean 0.30 s over 60 sims (18 s total); per generated frame 0.006 s (median / 48 frames); NVIDIA GH200 120GB, 5 steps, batch 1

shockwave_dit_jupiter_srba_s2_flip_sds_3e2 per frame (VAE+DiT, mean ± std over sims)
| Frame | RMSE | VRMSE | 1 – SSIM | H1 rel | H2 rel |
|---|---|---|---|---|---|
| 10 | 0.006 ± 0.002 | 5.301 ± 3.784 | 0.002 ± 0.001 | 6.041 ± 4.331 | 10.915 ± 8.067 |
| 20 | 0.009 ± 0.004 | 1.963 ± 3.406 | 0.004 ± 0.002 | 2.170 ± 2.961 | 3.735 ± 5.337 |
| 30 | 0.015 ± 0.009 | 0.918 ± 2.238 | 0.007 ± 0.005 | 1.350 ± 2.039 | 2.246 ± 3.778 |
| 40 | 0.022 ± 0.014 | 0.135 ± 0.078 | 0.012 ± 0.010 | 0.683 ± 0.093 | 0.950 ± 0.113 |
| 50 | 0.028 ± 0.018 | 0.132 ± 0.034 | 0.018 ± 0.017 | 0.732 ± 0.098 | 0.990 ± 0.107 |
| 60 | 0.034 ± 0.021 | 0.154 ± 0.045 | 0.027 ± 0.023 | 0.817 ± 0.134 | 1.068 ± 0.160 |
| 70 | 0.037 ± 0.023 | 0.173 ± 0.056 | 0.033 ± 0.027 | 0.855 ± 0.127 | 1.110 ± 0.136 |
| 80 | 0.040 ± 0.024 | 0.223 ± 0.084 | 0.038 ± 0.030 | 0.856 ± 0.116 | 1.099 ± 0.150 |
| 90 | 0.044 ± 0.025 | 0.299 ± 0.141 | 0.040 ± 0.029 | 0.913 ± 0.117 | 1.159 ± 0.141 |
