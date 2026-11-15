
### benchmark

| Model | RMSE | VRMSE | 1 – SSIM | MLW | H1 rel | H2 rel |
|---|---|---|---|---|---|---|
| shockwave_dit_jupiter_energy_1e1_s44_cond9 | 0.045 ± 0.033 | 0.586 ± 0.578 | 0.034 ± 0.034 | 62.7 ± 10.4 | 1.110 ± 0.324 | 1.284 ± 0.601 |
| shockwave_dit_jupiter_energy_1e1_s44_cond9: its VAE round trip | 0.008 ± 0.006 | 0.152 ± 0.831 | 0.007 ± 0.009 | 41.5 ± 14.2 | 0.730 ± 0.555 | 1.213 ± 0.918 |

shockwave_dit_jupiter_energy_1e1_s44_cond9: frames [40, 50, 60, 70, 80, 90], clamp=True

### benchmark_10

| Model | RMSE | VRMSE | 1 – SSIM | MLW | H1 rel | H2 rel |
|---|---|---|---|---|---|---|
| shockwave_dit_jupiter_energy_1e1_s44_cond9 | 0.043 ± 0.030 | 0.387 ± 0.103 | 0.035 ± 0.032 | 65.7 ± 12.4 | 0.959 ± 0.053 | 1.091 ± 0.163 |
| shockwave_dit_jupiter_energy_1e1_s44_cond9: its VAE round trip | 0.010 ± 0.008 | 0.093 ± 0.038 | 0.009 ± 0.010 | 47.8 ± 15.7 | 0.653 ± 0.076 | 0.985 ± 0.184 |

shockwave_dit_jupiter_energy_1e1_s44_cond9: frames [10, 20, 30, 40, 50, 60, 70, 80, 90], clamp=True
shockwave_dit_jupiter_energy_1e1_s44_cond9: generation per trajectory median 0.31 s (range 0.30-1.18), mean 0.32 s over 500 sims (159 s total); per generated frame 0.003 s (median / 96 frames); NVIDIA GH200 120GB, 5 steps, batch 1

shockwave_dit_jupiter_energy_1e1_s44_cond9 per frame (VAE+DiT, mean ± std over sims)
| Frame | RMSE | VRMSE | 1 – SSIM | H1 rel | H2 rel |
|---|---|---|---|---|---|
| 10 | 0.032 ± 0.021 | 0.225 ± 0.067 | 0.028 ± 0.027 | 0.775 ± 0.089 | 1.017 ± 0.144 |
| 20 | 0.040 ± 0.027 | 0.356 ± 0.747 | 0.040 ± 0.037 | 0.965 ± 0.290 | 1.125 ± 0.355 |
| 30 | 0.042 ± 0.029 | 0.452 ± 1.184 | 0.042 ± 0.040 | 1.037 ± 0.353 | 1.177 ± 0.461 |
| 40 | 0.043 ± 0.031 | 0.481 ± 0.626 | 0.040 ± 0.039 | 1.073 ± 0.379 | 1.242 ± 0.632 |
| 50 | 0.044 ± 0.032 | 0.632 ± 2.124 | 0.038 ± 0.038 | 1.127 ± 0.540 | 1.354 ± 1.082 |
| 60 | 0.044 ± 0.034 | 0.644 ± 0.864 | 0.035 ± 0.038 | 1.179 ± 0.636 | 1.494 ± 1.620 |
| 70 | 0.045 ± 0.034 | 1.011 ± 7.147 | 0.032 ± 0.035 | 1.228 ± 0.950 | 1.629 ± 2.202 |
| 80 | 0.045 ± 0.035 | 1.140 ± 7.542 | 0.030 ± 0.033 | 1.332 ± 1.268 | 1.906 ± 3.043 |
| 90 | 0.045 ± 0.035 | 1.301 ± 8.265 | 0.028 ± 0.031 | 1.387 ± 1.326 | 2.136 ± 3.389 |
