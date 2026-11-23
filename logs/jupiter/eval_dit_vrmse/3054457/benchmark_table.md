
### benchmark

| Model | RMSE | VRMSE | 1 – SSIM | MLW | H1 rel | H2 rel |
|---|---|---|---|---|---|---|
| shockwave_dit_jupiter_srba_s2_flip_vrmse_h1_30 | 0.030 ± 0.021 | 0.132 ± 0.049 | 0.023 ± 0.021 | 100.2 ± 20.1 | 0.674 ± 0.118 | 0.897 ± 0.109 |
| shockwave_dit_jupiter_srba_s2_flip_vrmse_h1_30: its VAE round trip | 0.016 ± 0.009 | 0.076 ± 0.020 | 0.012 ± 0.010 | 98.0 ± 21.8 | 0.563 ± 0.072 | 0.814 ± 0.085 |

shockwave_dit_jupiter_srba_s2_flip_vrmse_h1_30: frames [40, 50, 60, 70, 80, 90], clamp=True

### benchmark_10

| Model | RMSE | VRMSE | 1 – SSIM | MLW | H1 rel | H2 rel |
|---|---|---|---|---|---|---|
| shockwave_dit_jupiter_srba_s2_flip_vrmse_h1_30 | 0.025 ± 0.017 | 0.119 ± 0.040 | 0.016 ± 0.015 | 121.6 ± 15.5 | 0.627 ± 0.106 | 0.849 ± 0.101 |
| shockwave_dit_jupiter_srba_s2_flip_vrmse_h1_30: its VAE round trip | 0.014 ± 0.007 | 0.070 ± 0.016 | 0.009 ± 0.007 | 121.4 ± 16.7 | 0.534 ± 0.068 | 0.781 ± 0.082 |

shockwave_dit_jupiter_srba_s2_flip_vrmse_h1_30: frames [10, 20, 30, 40, 50, 60, 70, 80, 90], clamp=True
shockwave_dit_jupiter_srba_s2_flip_vrmse_h1_30: generation per trajectory median 0.31 s (range 0.30-0.91), mean 0.32 s over 60 sims (19 s total); per generated frame 0.006 s (median / 48 frames); NVIDIA GH200 120GB, 5 steps, batch 1

shockwave_dit_jupiter_srba_s2_flip_vrmse_h1_30 per frame (VAE+DiT, mean ± std over sims)
| Frame | RMSE | VRMSE | 1 – SSIM | H1 rel | H2 rel |
|---|---|---|---|---|---|
| 10 | 0.006 ± 0.002 | 3.451 ± 2.536 | 0.002 ± 0.001 | 4.906 ± 3.541 | 8.828 ± 6.604 |
| 20 | 0.008 ± 0.004 | 1.153 ± 1.947 | 0.003 ± 0.002 | 1.742 ± 2.325 | 3.007 ± 4.233 |
| 30 | 0.012 ± 0.007 | 0.612 ± 1.519 | 0.005 ± 0.004 | 1.152 ± 1.790 | 1.950 ± 3.363 |
| 40 | 0.017 ± 0.012 | 0.101 ± 0.048 | 0.009 ± 0.009 | 0.578 ± 0.096 | 0.823 ± 0.104 |
| 50 | 0.023 ± 0.016 | 0.105 ± 0.036 | 0.015 ± 0.015 | 0.629 ± 0.110 | 0.874 ± 0.095 |
| 60 | 0.027 ± 0.020 | 0.120 ± 0.046 | 0.020 ± 0.021 | 0.660 ± 0.120 | 0.890 ± 0.112 |
| 70 | 0.032 ± 0.023 | 0.143 ± 0.059 | 0.026 ± 0.025 | 0.693 ± 0.128 | 0.917 ± 0.118 |
| 80 | 0.035 ± 0.025 | 0.188 ± 0.094 | 0.031 ± 0.029 | 0.711 ± 0.134 | 0.926 ± 0.131 |
| 90 | 0.039 ± 0.026 | 0.256 ± 0.152 | 0.033 ± 0.028 | 0.789 ± 0.166 | 1.023 ± 0.175 |
