# Wan DiT: sampling settings

res256/pairs/unrolled/unrolled, protocol v2 (500 test sims, from frame 10, per-sim seeds), paired 95 % CI against
its own 2-step deterministic test (* = significant). Runs under `results/wan/dit/res256/pairs/unrolled/unrolled/`.
η = share of fresh noise per step (`sampling.stochastic`): 0 = UniPC, deterministic.

| run | steps | η | samples | VRMSE | Δ | bench | LDE | front shift |
|---|---|---|---|---|---|---|---|---|
| (the run) | 2 | 0 | 1 | 0.2322 | – | 0.214 | 0.389 | 7.05 px |
| v2_5s | 5 | 0 | 1 | 0.2336 | +0.0014* | 0.216 | 0.379 | 7.01 px |
| bench_steps10 | 10 | 0 | 1 | 0.2425 | +0.0103* | 0.224 | 0.343 | 7.09 px |
| bench_steps20 | 20 | 0 | 1 | 0.2561 | +0.0239* | 0.237 | 0.316 | 7.34 px |
| bench_steps50 | 50 | 0 | 1 | 0.2708 | +0.0386* | 0.251 | 0.309 | 7.67 px |
| bench_steps2_eta0.5 | 2 | 0.5 | 1 | 0.2320 | −0.0002* | 0.214 | 0.389 | 7.05 px |
| bench_steps2_eta1 | 2 | 1 | 1 | 0.2312 | −0.0009* | 0.214 | 0.390 | 7.07 px |
| bench_steps5_eta0.5 | 5 | 0.5 | 1 | 0.2301 | −0.0021* | 0.213 | 0.388 | 6.99 px |
| bench_steps5_eta1 | 5 | 1 | 1 | **0.2284** | −0.0038* | **0.211** | 0.385 | 6.95 px |
| bench_steps5_eta1_ens4 | 5 | 1 | 4 (mean) | 0.2240 | – | – | 0.404 | 6.87 px |

- More deterministic steps sharpen the prediction (LDE 0.389 → 0.309) but put the detail in the wrong place:
  VRMSE rises. The unrolled training covers chains of 2–5 steps only.
- Stochastic sampling at 5 steps, η 1 is the best single prediction and the evaluation default
  (`configs/wan/dit_eval.yaml`).
- The 4-sample mean is lower again, but its spread is a quarter of its error (spread-skill 0.25): an
  under-dispersed ensemble.
