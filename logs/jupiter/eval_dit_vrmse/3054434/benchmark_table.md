
### benchmark

| Model | RMSE | VRMSE | 1 – SSIM | MLW | H1 rel | H2 rel |
|---|---|---|---|---|---|---|
| shockwave_dit_jupiter_srba_s2_ref | 0.037 ± 0.025 | 0.164 ± 0.056 | 0.029 ± 0.025 | 107.0 ± 22.3 | 0.793 ± 0.122 | 1.032 ± 0.130 |
| shockwave_dit_jupiter_srba_s2_ref: its VAE round trip | 0.017 ± 0.010 | 0.085 ± 0.024 | 0.013 ± 0.011 | 102.2 ± 23.6 | 0.626 ± 0.105 | 0.905 ± 0.152 |

shockwave_dit_jupiter_srba_s2_ref: frames [40, 50, 60, 70, 80, 90], clamp=True

### benchmark_10

| Model | RMSE | VRMSE | 1 – SSIM | MLW | H1 rel | H2 rel |
|---|---|---|---|---|---|---|
| shockwave_dit_jupiter_srba_s2_ref | 0.031 ± 0.020 | 0.147 ± 0.046 | 0.020 ± 0.018 | 130.7 ± 17.6 | 0.729 ± 0.111 | 0.966 ± 0.124 |
| shockwave_dit_jupiter_srba_s2_ref: its VAE round trip | 0.014 ± 0.008 | 0.076 ± 0.020 | 0.009 ± 0.008 | 126.6 ± 18.6 | 0.585 ± 0.098 | 0.857 ± 0.142 |

shockwave_dit_jupiter_srba_s2_ref: frames [10, 20, 30, 40, 50, 60, 70, 80, 90], clamp=True
shockwave_dit_jupiter_srba_s2_ref: generation per trajectory median 0.29 s (range 0.29-1.05), mean 0.31 s over 60 sims (18 s total); per generated frame 0.006 s (median / 48 frames); NVIDIA GH200 120GB, 5 steps, batch 1

shockwave_dit_jupiter_srba_s2_ref per frame (VAE+DiT, mean ± std over sims)
| Frame | RMSE | VRMSE | 1 – SSIM | H1 rel | H2 rel |
|---|---|---|---|---|---|
| 10 | 0.005 ± 0.002 | 4.182 ± 3.044 | 0.002 ± 0.001 | 5.873 ± 4.180 | 10.649 ± 7.803 |
| 20 | 0.008 ± 0.004 | 1.371 ± 2.312 | 0.003 ± 0.002 | 1.987 ± 2.709 | 3.459 ± 4.919 |
| 30 | 0.013 ± 0.009 | 0.726 ± 1.743 | 0.006 ± 0.005 | 1.250 ± 1.869 | 2.110 ± 3.513 |
| 40 | 0.021 ± 0.015 | 0.122 ± 0.066 | 0.011 ± 0.011 | 0.653 ± 0.117 | 0.918 ± 0.139 |
| 50 | 0.028 ± 0.020 | 0.129 ± 0.044 | 0.019 ± 0.018 | 0.725 ± 0.125 | 0.985 ± 0.123 |
| 60 | 0.034 ± 0.024 | 0.149 ± 0.055 | 0.027 ± 0.025 | 0.775 ± 0.129 | 1.020 ± 0.131 |
| 70 | 0.039 ± 0.027 | 0.179 ± 0.068 | 0.034 ± 0.030 | 0.825 ± 0.132 | 1.066 ± 0.137 |
| 80 | 0.044 ± 0.029 | 0.235 ± 0.109 | 0.039 ± 0.035 | 0.847 ± 0.133 | 1.082 ± 0.149 |
| 90 | 0.047 ± 0.030 | 0.311 ± 0.170 | 0.041 ± 0.032 | 0.918 ± 0.140 | 1.167 ± 0.151 |
