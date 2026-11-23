
### benchmark

| Model | RMSE | VRMSE | 1 – SSIM | MLW | H1 rel | H2 rel |
|---|---|---|---|---|---|---|
| shockwave_dit_jupiter_srba_s2_flip_sds_1e2 | 0.030 ± 0.021 | 0.137 ± 0.049 | 0.024 ± 0.022 | 107.8 ± 24.4 | 0.733 ± 0.124 | 0.990 ± 0.148 |
| shockwave_dit_jupiter_srba_s2_flip_sds_1e2: its VAE round trip | 0.015 ± 0.009 | 0.079 ± 0.021 | 0.012 ± 0.010 | 103.5 ± 25.4 | 0.613 ± 0.100 | 0.894 ± 0.150 |

shockwave_dit_jupiter_srba_s2_flip_sds_1e2: frames [40, 50, 60, 70, 80, 90], clamp=True

### benchmark_10

| Model | RMSE | VRMSE | 1 – SSIM | MLW | H1 rel | H2 rel |
|---|---|---|---|---|---|---|
| shockwave_dit_jupiter_srba_s2_flip_sds_1e2 | 0.025 ± 0.018 | 0.124 ± 0.040 | 0.017 ± 0.015 | 131.0 ± 19.8 | 0.680 ± 0.113 | 0.932 ± 0.139 |
| shockwave_dit_jupiter_srba_s2_flip_sds_1e2: its VAE round trip | 0.013 ± 0.007 | 0.071 ± 0.017 | 0.009 ± 0.007 | 128.8 ± 20.2 | 0.575 ± 0.094 | 0.849 ± 0.140 |

shockwave_dit_jupiter_srba_s2_flip_sds_1e2: frames [10, 20, 30, 40, 50, 60, 70, 80, 90], clamp=True
shockwave_dit_jupiter_srba_s2_flip_sds_1e2: generation per trajectory median 0.30 s (range 0.29-0.79), mean 0.31 s over 60 sims (18 s total); per generated frame 0.006 s (median / 48 frames); NVIDIA GH200 120GB, 5 steps, batch 1

shockwave_dit_jupiter_srba_s2_flip_sds_1e2 per frame (VAE+DiT, mean ± std over sims)
| Frame | RMSE | VRMSE | 1 – SSIM | H1 rel | H2 rel |
|---|---|---|---|---|---|
| 10 | 0.006 ± 0.002 | 4.546 ± 3.302 | 0.002 ± 0.001 | 5.983 ± 4.311 | 10.809 ± 8.022 |
| 20 | 0.008 ± 0.004 | 1.381 ± 2.316 | 0.003 ± 0.002 | 2.045 ± 2.809 | 3.555 ± 5.083 |
| 30 | 0.012 ± 0.008 | 0.718 ± 1.745 | 0.006 ± 0.004 | 1.281 ± 1.991 | 2.171 ± 3.712 |
| 40 | 0.018 ± 0.012 | 0.113 ± 0.062 | 0.010 ± 0.009 | 0.634 ± 0.104 | 0.905 ± 0.136 |
| 50 | 0.024 ± 0.017 | 0.112 ± 0.036 | 0.015 ± 0.015 | 0.687 ± 0.116 | 0.961 ± 0.130 |
| 60 | 0.028 ± 0.021 | 0.124 ± 0.045 | 0.022 ± 0.021 | 0.718 ± 0.126 | 0.982 ± 0.150 |
| 70 | 0.032 ± 0.023 | 0.147 ± 0.057 | 0.028 ± 0.026 | 0.757 ± 0.131 | 1.018 ± 0.159 |
| 80 | 0.036 ± 0.026 | 0.192 ± 0.094 | 0.033 ± 0.030 | 0.768 ± 0.141 | 1.019 ± 0.178 |
| 90 | 0.039 ± 0.027 | 0.263 ± 0.156 | 0.035 ± 0.030 | 0.849 ± 0.160 | 1.115 ± 0.176 |
