
### benchmark

| Model | RMSE | VRMSE | 1 – SSIM | MLW | H1 rel | H2 rel |
|---|---|---|---|---|---|---|
| shockwave_dit_jupiter_baseline_s44_lr1e4_cond9 | 0.023 ± 0.000 | 0.417 ± 0.077 | 0.014 ± 0.001 | 59.6 ± 1.9 | 1.063 ± 0.066 | 1.233 ± 0.141 |
| shockwave_dit_jupiter_baseline_s44_lr1e4_cond9: its VAE round trip | 0.003 ± 0.001 | 0.077 ± 0.001 | 0.001 ± 0.001 | 40.0 ± 0.8 | 0.642 ± 0.028 | 1.121 ± 0.167 |

shockwave_dit_jupiter_baseline_s44_lr1e4_cond9: frames [40, 50, 60, 70, 80, 90], clamp=True

### benchmark_10

| Model | RMSE | VRMSE | 1 – SSIM | MLW | H1 rel | H2 rel |
|---|---|---|---|---|---|---|
| shockwave_dit_jupiter_baseline_s44_lr1e4_cond9 | 0.023 ± 0.001 | 0.323 ± 0.016 | 0.014 ± 0.002 | 65.5 ± 0.1 | 0.915 ± 0.021 | 1.058 ± 0.013 |
| shockwave_dit_jupiter_baseline_s44_lr1e4_cond9: its VAE round trip | 0.004 ± 0.001 | 0.072 ± 0.009 | 0.002 ± 0.001 | 48.8 ± 1.0 | 0.582 ± 0.041 | 0.915 ± 0.032 |

shockwave_dit_jupiter_baseline_s44_lr1e4_cond9: frames [10, 20, 30, 40, 50, 60, 70, 80, 90], clamp=True
shockwave_dit_jupiter_baseline_s44_lr1e4_cond9: generation per trajectory median 1.00 s (range 0.29-1.70), mean 1.00 s over 2 sims (2 s total); per generated frame 0.010 s (median / 96 frames); NVIDIA GH200 120GB, 5 steps, batch 1

shockwave_dit_jupiter_baseline_s44_lr1e4_cond9 per frame (VAE+DiT, mean ± std over sims)
| Frame | RMSE | VRMSE | 1 – SSIM | H1 rel | H2 rel |
|---|---|---|---|---|---|
| 10 | 0.021 ± 0.003 | 0.237 ± 0.027 | 0.011 ± 0.003 | 0.732 ± 0.045 | 0.978 ± 0.041 |
| 20 | 0.026 ± 0.006 | 0.325 ± 0.008 | 0.016 ± 0.004 | 0.906 ± 0.026 | 1.079 ± 0.014 |
| 30 | 0.021 ± 0.006 | 0.330 ± 0.005 | 0.017 ± 0.008 | 0.964 ± 0.053 | 1.084 ± 0.012 |
| 40 | 0.020 ± 0.006 | 0.375 ± 0.017 | 0.015 ± 0.007 | 0.981 ± 0.001 | 1.141 ± 0.072 |
| 50 | 0.021 ± 0.005 | 0.413 ± 0.031 | 0.014 ± 0.005 | 1.026 ± 0.044 | 1.200 ± 0.107 |
| 60 | 0.021 ± 0.001 | 0.413 ± 0.058 | 0.013 ± 0.001 | 1.069 ± 0.058 | 1.296 ± 0.123 |
| 70 | 0.022 ± 0.002 | 0.421 ± 0.102 | 0.014 ± 0.002 | 1.099 ± 0.077 | 1.368 ± 0.113 |
| 80 | 0.024 ± 0.004 | 0.446 ± 0.128 | 0.014 ± 0.003 | 1.213 ± 0.062 | 1.741 ± 0.318 |
| 90 | 0.026 ± 0.005 | 0.489 ± 0.135 | 0.014 ± 0.004 | 1.257 ± 0.054 | 1.911 ± 0.355 |
