
### benchmark

| Model | RMSE | VRMSE | 1 – SSIM | MLW | H1 rel | H2 rel |
|---|---|---|---|---|---|---|
| shockwave_dit_jupiter_energy_3e2_s44_cond9 | 0.045 ± 0.034 | 0.587 ± 0.533 | 0.034 ± 0.035 | 62.5 ± 10.2 | 1.105 ± 0.284 | 1.261 ± 0.537 |
| shockwave_dit_jupiter_energy_3e2_s44_cond9: its VAE round trip | 0.007 ± 0.006 | 0.161 ± 1.022 | 0.006 ± 0.008 | 40.9 ± 14.2 | 0.711 ± 0.584 | 1.174 ± 0.960 |

shockwave_dit_jupiter_energy_3e2_s44_cond9: frames [40, 50, 60, 70, 80, 90], clamp=True

### benchmark_10

| Model | RMSE | VRMSE | 1 – SSIM | MLW | H1 rel | H2 rel |
|---|---|---|---|---|---|---|
| shockwave_dit_jupiter_energy_3e2_s44_cond9 | 0.044 ± 0.030 | 0.391 ± 0.103 | 0.035 ± 0.032 | 65.5 ± 12.1 | 0.960 ± 0.051 | 1.082 ± 0.139 |
| shockwave_dit_jupiter_energy_3e2_s44_cond9: its VAE round trip | 0.010 ± 0.007 | 0.091 ± 0.037 | 0.008 ± 0.009 | 47.0 ± 15.6 | 0.632 ± 0.073 | 0.949 ± 0.161 |

shockwave_dit_jupiter_energy_3e2_s44_cond9: frames [10, 20, 30, 40, 50, 60, 70, 80, 90], clamp=True
shockwave_dit_jupiter_energy_3e2_s44_cond9: generation per trajectory median 0.31 s (range 0.31-1.42), mean 0.32 s over 500 sims (159 s total); per generated frame 0.003 s (median / 96 frames); NVIDIA GH200 120GB, 5 steps, batch 1

shockwave_dit_jupiter_energy_3e2_s44_cond9 per frame (VAE+DiT, mean ± std over sims)
| Frame | RMSE | VRMSE | 1 – SSIM | H1 rel | H2 rel |
|---|---|---|---|---|---|
| 10 | 0.033 ± 0.021 | 0.233 ± 0.067 | 0.028 ± 0.027 | 0.776 ± 0.086 | 1.006 ± 0.126 |
| 20 | 0.040 ± 0.027 | 0.358 ± 0.672 | 0.040 ± 0.036 | 0.966 ± 0.309 | 1.117 ± 0.369 |
| 30 | 0.042 ± 0.029 | 0.438 ± 0.835 | 0.042 ± 0.039 | 1.038 ± 0.312 | 1.167 ± 0.418 |
| 40 | 0.043 ± 0.031 | 0.486 ± 0.694 | 0.040 ± 0.039 | 1.071 ± 0.342 | 1.225 ± 0.569 |
| 50 | 0.044 ± 0.033 | 0.627 ± 1.980 | 0.038 ± 0.039 | 1.120 ± 0.469 | 1.324 ± 0.935 |
| 60 | 0.045 ± 0.035 | 0.645 ± 0.812 | 0.035 ± 0.038 | 1.165 ± 0.548 | 1.446 ± 1.389 |
| 70 | 0.045 ± 0.035 | 0.997 ± 6.794 | 0.033 ± 0.036 | 1.209 ± 0.816 | 1.568 ± 1.901 |
| 80 | 0.045 ± 0.036 | 1.133 ± 7.218 | 0.030 ± 0.034 | 1.303 ± 1.133 | 1.809 ± 2.602 |
| 90 | 0.046 ± 0.037 | 1.322 ± 8.126 | 0.028 ± 0.032 | 1.339 ± 1.102 | 2.002 ± 2.873 |
