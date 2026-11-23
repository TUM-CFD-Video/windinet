
### benchmark

| Model | RMSE | VRMSE | 1 – SSIM | MLW | H1 rel | H2 rel |
|---|---|---|---|---|---|---|
| shockwave_dit_jupiter_srba_s2_flip_sds_3e3 | 0.030 ± 0.022 | 0.135 ± 0.050 | 0.023 ± 0.022 | 104.0 ± 22.6 | 0.724 ± 0.124 | 0.978 ± 0.146 |
| shockwave_dit_jupiter_srba_s2_flip_sds_3e3: its VAE round trip | 0.015 ± 0.009 | 0.079 ± 0.021 | 0.012 ± 0.010 | 101.9 ± 24.0 | 0.611 ± 0.100 | 0.892 ± 0.150 |

shockwave_dit_jupiter_srba_s2_flip_sds_3e3: frames [40, 50, 60, 70, 80, 90], clamp=True

### benchmark_10

| Model | RMSE | VRMSE | 1 – SSIM | MLW | H1 rel | H2 rel |
|---|---|---|---|---|---|---|
| shockwave_dit_jupiter_srba_s2_flip_sds_3e3 | 0.025 ± 0.018 | 0.121 ± 0.041 | 0.017 ± 0.015 | 126.4 ± 17.9 | 0.671 ± 0.113 | 0.921 ± 0.137 |
| shockwave_dit_jupiter_srba_s2_flip_sds_3e3: its VAE round trip | 0.013 ± 0.007 | 0.071 ± 0.018 | 0.009 ± 0.007 | 126.2 ± 19.0 | 0.572 ± 0.095 | 0.846 ± 0.140 |

shockwave_dit_jupiter_srba_s2_flip_sds_3e3: frames [10, 20, 30, 40, 50, 60, 70, 80, 90], clamp=True
shockwave_dit_jupiter_srba_s2_flip_sds_3e3: generation per trajectory median 0.29 s (range 0.28-0.73), mean 0.30 s over 60 sims (18 s total); per generated frame 0.006 s (median / 48 frames); NVIDIA GH200 120GB, 5 steps, batch 1

shockwave_dit_jupiter_srba_s2_flip_sds_3e3 per frame (VAE+DiT, mean ± std over sims)
| Frame | RMSE | VRMSE | 1 – SSIM | H1 rel | H2 rel |
|---|---|---|---|---|---|
| 10 | 0.006 ± 0.002 | 4.505 ± 3.298 | 0.002 ± 0.001 | 5.918 ± 4.231 | 10.729 ± 7.902 |
| 20 | 0.008 ± 0.004 | 1.361 ± 2.281 | 0.003 ± 0.002 | 2.000 ± 2.748 | 3.490 ± 4.987 |
| 30 | 0.012 ± 0.007 | 0.643 ± 1.534 | 0.005 ± 0.004 | 1.238 ± 1.905 | 2.111 ± 3.577 |
| 40 | 0.017 ± 0.012 | 0.104 ± 0.056 | 0.009 ± 0.009 | 0.615 ± 0.103 | 0.887 ± 0.136 |
| 50 | 0.023 ± 0.017 | 0.106 ± 0.037 | 0.015 ± 0.015 | 0.672 ± 0.117 | 0.947 ± 0.130 |
| 60 | 0.027 ± 0.021 | 0.121 ± 0.047 | 0.021 ± 0.021 | 0.706 ± 0.125 | 0.968 ± 0.146 |
| 70 | 0.032 ± 0.024 | 0.145 ± 0.059 | 0.027 ± 0.026 | 0.745 ± 0.131 | 1.002 ± 0.152 |
| 80 | 0.036 ± 0.026 | 0.191 ± 0.094 | 0.032 ± 0.030 | 0.763 ± 0.145 | 1.011 ± 0.179 |
| 90 | 0.039 ± 0.027 | 0.263 ± 0.152 | 0.035 ± 0.030 | 0.851 ± 0.152 | 1.116 ± 0.169 |
