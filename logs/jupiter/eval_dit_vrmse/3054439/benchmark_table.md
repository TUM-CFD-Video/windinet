
### benchmark

| Model | RMSE | VRMSE | 1 – SSIM | MLW | H1 rel | H2 rel |
|---|---|---|---|---|---|---|
| shockwave_dit_jupiter_srba_s2_flip | 0.031 ± 0.021 | 0.140 ± 0.049 | 0.023 ± 0.022 | 106.1 ± 23.0 | 0.730 ± 0.122 | 0.980 ± 0.143 |
| shockwave_dit_jupiter_srba_s2_flip: its VAE round trip | 0.015 ± 0.009 | 0.079 ± 0.021 | 0.012 ± 0.010 | 101.9 ± 23.7 | 0.610 ± 0.100 | 0.889 ± 0.149 |

shockwave_dit_jupiter_srba_s2_flip: frames [40, 50, 60, 70, 80, 90], clamp=True

### benchmark_10

| Model | RMSE | VRMSE | 1 – SSIM | MLW | H1 rel | H2 rel |
|---|---|---|---|---|---|---|
| shockwave_dit_jupiter_srba_s2_flip | 0.026 ± 0.018 | 0.125 ± 0.040 | 0.017 ± 0.015 | 130.3 ± 18.0 | 0.676 ± 0.112 | 0.922 ± 0.135 |
| shockwave_dit_jupiter_srba_s2_flip: its VAE round trip | 0.013 ± 0.007 | 0.071 ± 0.018 | 0.009 ± 0.007 | 126.7 ± 18.7 | 0.571 ± 0.094 | 0.844 ± 0.139 |

shockwave_dit_jupiter_srba_s2_flip: frames [10, 20, 30, 40, 50, 60, 70, 80, 90], clamp=True
shockwave_dit_jupiter_srba_s2_flip: generation per trajectory median 0.32 s (range 0.31-1.08), mean 0.34 s over 60 sims (20 s total); per generated frame 0.007 s (median / 48 frames); NVIDIA GH200 120GB, 5 steps, batch 1

shockwave_dit_jupiter_srba_s2_flip per frame (VAE+DiT, mean ± std over sims)
| Frame | RMSE | VRMSE | 1 – SSIM | H1 rel | H2 rel |
|---|---|---|---|---|---|
| 10 | 0.005 ± 0.002 | 4.192 ± 2.978 | 0.002 ± 0.001 | 5.770 ± 4.112 | 10.493 ± 7.692 |
| 20 | 0.008 ± 0.004 | 1.414 ± 2.368 | 0.003 ± 0.002 | 1.947 ± 2.655 | 3.399 ± 4.832 |
| 30 | 0.012 ± 0.007 | 0.704 ± 1.707 | 0.005 ± 0.004 | 1.213 ± 1.815 | 2.059 ± 3.412 |
| 40 | 0.018 ± 0.013 | 0.107 ± 0.059 | 0.010 ± 0.009 | 0.620 ± 0.106 | 0.890 ± 0.135 |
| 50 | 0.024 ± 0.017 | 0.110 ± 0.037 | 0.015 ± 0.015 | 0.677 ± 0.113 | 0.948 ± 0.124 |
| 60 | 0.028 ± 0.021 | 0.125 ± 0.047 | 0.021 ± 0.021 | 0.709 ± 0.125 | 0.966 ± 0.145 |
| 70 | 0.033 ± 0.023 | 0.150 ± 0.059 | 0.027 ± 0.026 | 0.747 ± 0.131 | 0.999 ± 0.152 |
| 80 | 0.037 ± 0.025 | 0.199 ± 0.095 | 0.033 ± 0.030 | 0.773 ± 0.143 | 1.016 ± 0.176 |
| 90 | 0.040 ± 0.026 | 0.275 ± 0.158 | 0.035 ± 0.029 | 0.859 ± 0.153 | 1.120 ± 0.169 |
