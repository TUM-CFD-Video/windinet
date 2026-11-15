
### benchmark

| Model | RMSE | VRMSE | 1 – SSIM | MLW | H1 rel | H2 rel |
|---|---|---|---|---|---|---|
| shockwave_dit_jupiter_energy_1e2_s44_cond9 | 0.045 ± 0.033 | 0.585 ± 0.448 | 0.034 ± 0.035 | 62.1 ± 9.7 | 1.106 ± 0.228 | 1.249 ± 0.456 |
| shockwave_dit_jupiter_energy_1e2_s44_cond9: its VAE round trip | 0.006 ± 0.005 | 0.130 ± 0.695 | 0.005 ± 0.007 | 39.2 ± 13.2 | 0.666 ± 0.444 | 1.124 ± 0.764 |

shockwave_dit_jupiter_energy_1e2_s44_cond9: frames [40, 50, 60, 70, 80, 90], clamp=True

### benchmark_10

| Model | RMSE | VRMSE | 1 – SSIM | MLW | H1 rel | H2 rel |
|---|---|---|---|---|---|---|
| shockwave_dit_jupiter_energy_1e2_s44_cond9 | 0.044 ± 0.030 | 0.391 ± 0.104 | 0.035 ± 0.032 | 64.8 ± 11.6 | 0.963 ± 0.053 | 1.082 ± 0.130 |
| shockwave_dit_jupiter_energy_1e2_s44_cond9: its VAE round trip | 0.008 ± 0.006 | 0.081 ± 0.033 | 0.007 ± 0.008 | 45.2 ± 14.8 | 0.599 ± 0.069 | 0.918 ± 0.154 |

shockwave_dit_jupiter_energy_1e2_s44_cond9: frames [10, 20, 30, 40, 50, 60, 70, 80, 90], clamp=True
shockwave_dit_jupiter_energy_1e2_s44_cond9: generation per trajectory median 0.30 s (range 0.29-1.24), mean 0.31 s over 500 sims (157 s total); per generated frame 0.003 s (median / 96 frames); NVIDIA GH200 120GB, 5 steps, batch 1

shockwave_dit_jupiter_energy_1e2_s44_cond9 per frame (VAE+DiT, mean ± std over sims)
| Frame | RMSE | VRMSE | 1 – SSIM | H1 rel | H2 rel |
|---|---|---|---|---|---|
| 10 | 0.031 ± 0.021 | 0.224 ± 0.066 | 0.027 ± 0.026 | 0.766 ± 0.092 | 0.999 ± 0.123 |
| 20 | 0.040 ± 0.027 | 0.354 ± 0.595 | 0.041 ± 0.037 | 0.971 ± 0.249 | 1.120 ± 0.296 |
| 30 | 0.042 ± 0.029 | 0.430 ± 0.725 | 0.042 ± 0.040 | 1.040 ± 0.246 | 1.162 ± 0.324 |
| 40 | 0.043 ± 0.031 | 0.480 ± 0.538 | 0.041 ± 0.039 | 1.072 ± 0.270 | 1.215 ± 0.460 |
| 50 | 0.044 ± 0.033 | 0.657 ± 2.689 | 0.038 ± 0.039 | 1.119 ± 0.395 | 1.309 ± 0.820 |
| 60 | 0.045 ± 0.034 | 0.647 ± 0.884 | 0.036 ± 0.038 | 1.159 ± 0.450 | 1.418 ± 1.199 |
| 70 | 0.045 ± 0.035 | 1.115 ± 9.523 | 0.033 ± 0.036 | 1.199 ± 0.659 | 1.532 ± 1.622 |
| 80 | 0.045 ± 0.035 | 1.258 ± 10.247 | 0.030 ± 0.034 | 1.283 ± 0.985 | 1.750 ± 2.242 |
| 90 | 0.046 ± 0.037 | 1.446 ± 10.931 | 0.028 ± 0.032 | 1.315 ± 0.904 | 1.929 ± 2.463 |
