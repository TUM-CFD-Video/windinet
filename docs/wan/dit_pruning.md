# Wan DiT: pruned students

Goal: at most half the parameters of the teacher `res256/pairs/unrolled/unrolled2` (v2 test VRMSE 0.2307 at 2 steps)
at almost its quality. Plan and sources: [pruning_research.md](pruning_research.md). Runs under
`results/wan/dit/res256/pairs/pruned/`; config `configs/wan/dit_prune.yaml`; code `windinet/wan/dit/{prune,distill}.py`.

## Summary

- Best student: `res256/pairs/pruned/inner_c2_f3776/long`, 705.2 M = 49.7 % of the teacher: cross-attention heads
  12 → 2 and FFN 8960 → 3776 in all 30 blocks, self-attention and the stream kept.
- 2-step test 0.2431 vs teacher 0.2307 (paired +0.012); target ≤ 0.242 missed by 0.001. 5 steps η 1: 0.2410 vs 0.2277.
- Depth (14 blocks) and stream width (760) cuts recover far worse than block-internal width at equal updates.
- The least-squares refit of the layer after each width cut is what makes the I cut cheap (0.138 → 0.033 rel MSE).
- Compute: 20.1 node-h.

## Method

- Student = teacher with parts cut, no re-init; patchify, gamma tokens and head kept.
- Importance from our own data: train sims at drawn sigmas, half of them the sampler's own chain states (as in
  unrolled training), the teacher's conditioning.
- Score of a head or FFN channel: the output energy it adds, E[x²] ‖W[:, c]‖²; of a stream channel: the same summed
  over every layer that reads the stream.
- Compensation at cut time: every layer whose inputs were cut is refit by least squares on the calibration activations
  (weights and bias); a kept block followed by dropped ones adds their mean residual as a trainable bias.
- Recovery loss: flow loss on the ground truth (anchor) + MSE to the teacher's velocity at the same inputs, the
  unrolled ones being the student's own chain states (+ relative MSE of block outputs where set).
- Closeness to the teacher: velocity cosine and relative MSE on a fixed batch (8 val sims × σ 1, 0.9, 0.75, 0.5).

## Parameters

Teacher: 1419.2 M (30 blocks of 46.4 M, 26.0 M outside the blocks). Limit: 709.6 M.

| arm | cut | params | share |
|---|---|---|---|
| A | depth 30 → 14 (15 would be 50.9 %) | 676.2 M | 47.6 % |
| W | stream 1536 → 760, heads and FFN kept | 702.4 M | 49.5 % |
| I (first plan) | self heads 12 → 8, cross heads 12 → 4, FFN 8960 → 4288 | 705.1 M | 49.7 % |
| I (run) | cross heads 12 → 2, FFN 8960 → 3776, self heads kept | 705.2 M | 49.7 % |

The stream axis is linear in the parameters while heads and FFN keep their width, so 1088 channels would still be
70.8 %.

## Right after the cut

Velocity relative MSE / cosine against the teacher on the fixed val batch, before training (fp32 grams).
The `cut-only` rows come from a scratch script, `/scratch/project_465003416/lc_work/scratch_tests/research_dit/`
`cut_splits.py` with its output `cut_splits.json`, which is not in the repo.

| run | cut | params | plain | refit |
|---|---|---|---|---|
| cut-only | I: 8 / 4 / 4288 (self heads / cross heads / FFN) | 705.1 M | 0.241 / 0.875 | 0.044 / 0.977 |
| cut-only | I: 8 / 2 / 4800 | 705.1 M | 0.232 / 0.879 | 0.046 / 0.976 |
| cut-only | I: 6 / 6 / 4288 | 705.1 M | 0.388 / 0.784 | 0.076 / 0.961 |
| cut-only | I: 6 / 2 / 5312 | 705.1 M | 0.402 / 0.779 | 0.079 / 0.959 |
| cut-only | I: 4 / 4 / 5312 | 705.1 M | 0.842 / 0.507 | 0.111 / 0.943 |
| cut-only | I: 12 / 2 / 3776 | 705.2 M | 0.138 / 0.929 | **0.033 / 0.983** |
| cut-only | W: stream 760 | 702.4 M | 0.881 / 0.350 | 0.472 / 0.725 |
| res256/pairs/pruned/depth14_spread | A: 14 blocks, evenly spaced | 676.2 M | 0.661 / 0.597 | 0.374 / 0.792 |
| res256/pairs/pruned/depth14_scan | A: 14 blocks by drop scan ([0–6, 8, 12, 20, 26–29]) | 676.1 M | 0.218 / 0.888 | 0.202 / 0.894 |
| res256/pairs/pruned/inner_c2_f3776 | I: 12 / 2 / 3776 | 705.2 M | 0.129 / 0.934 | 0.034 / 0.982 |

- Self-attention heads cost the most per parameter; cross-attention heads (4 gamma tokens) the least.
- The least-squares refit cuts the error of every width cut 2–5×; the mean-residual bias helps depth less.
- With the grams summed in bf16 (first cut-only job) every refit failed (cos ≈ 0); the grams are now fp32.

## Pilots

1500 updates (lr 1e-5, warm-up 100, cosine), teacher's unrolled recipe + distillation. Val VRMSE on 128 val sims at
2 steps (teacher: 0.242); cos = velocity cosine to the teacher on the fixed batch.

| run | arm | init | 500 | 1000 | 1500 | cos at last val | s / update |
|---|---|---|---|---|---|---|---|
| res256/pairs/pruned/depth14_spread | A, every other | 1.561 | 0.517 | 0.454 | stopped | 0.977 | 3.1 |
| res256/pairs/pruned/depth14_scan | A, drop scan | 1.368 | 0.490 | stopped | – | 0.977 | 3.1 |
| res256/pairs/pruned/stream760 | W | 1.532 | stopped at 270 | – | – | – | 4.4 |
| res256/pairs/pruned/inner_c2_f3776 | I | 0.440 | 0.293 | 0.274 | 0.271 | 0.996 | 4.5 |

- I starts right after the cut closer to the teacher than either A after 1000 updates.
- W did not learn: its velocity distillation loss stayed at about 1.0 for 260 updates.
- The drop-scan block choice beats every other block by 0.027 at step 500; both stay far behind I.
Gradual against one-shot on I, at equal total updates (val VRMSE, 128 sims):

| run | path | after cut | 500 | 1000 | 1500 |
|---|---|---|---|---|---|
| res256/pairs/pruned/inner_c2_f3776 | one-shot to 2 / 3776 | 0.440 | 0.293 | 0.274 | 0.271 |
| res256/pairs/pruned/gradual_c6_f6000 (+ /c2_f3776) | 6 / 6000 (1005 M) for 500, then 2 / 3776 | 0.321 | 0.258 | 0.284 | 0.272 |

- The gradual path cuts the damage of the final cut (0.374 vs 0.440 right after it) but ends level with one-shot.

- Test of the I pilot (500 sims, 2 steps): 0.2532, cfd_bench 0.231; paired Δ to the teacher +0.0224 [+0.0202, +0.0247].

## Long run of I

The I pilot continued with its AdamW moments: 6000 updates at peak lr 1e-5, warm-up 100, cosine.

| run | updates | val 1000 | 2000 | 3000 | 4000 | 5000 | 6000 | cos at 6000 |
|---|---|---|---|---|---|---|---|---|
| res256/pairs/pruned/inner_c2_f3776/long | 1500 + 6000 | 0.279 | 0.270 | 0.267 | 0.260 | 0.257 | 0.257 | 0.9973 |
| res256/pairs/pruned/inner_c2_f3776/long/cont | + 2000 of 4000 at 3e-6 (stopped) | 0.259 | 0.257 | – | – | – | – | 0.9974 |

## Test (500 sims, v2)

| run | params | 2 steps | paired Δ, 2 steps | 5 steps η 1 | paired Δ, 5 steps | bench (2 steps) | LDE | front shift |
|---|---|---|---|---|---|---|---|---|
| res256/pairs/unrolled/unrolled2 (teacher) | 1419.2 M | 0.2307 | – | 0.2277 | – | 0.213 | 0.380 | 7.03 px |
| res256/pairs/pruned/inner_c2_f3776 | 705.2 M | 0.2532 | +0.0224 [+0.0202, +0.0247] | – | – | 0.231 | 0.445 | 7.67 px |
| res256/pairs/pruned/inner_c2_f3776/long | 705.2 M | 0.2431 | +0.0123 [+0.0107, +0.0140] | 0.2410 | +0.0133 [+0.0117, +0.0148] | 0.223 | 0.421 | 7.44 px |

5-step numbers: the `bench_steps5_eta1` folder of each run. Paired Δ: student − teacher at the same setting.

- The long run closes 0.010 of the pilot's gap; the 2-step target (≤ 0.242) is missed by 0.001.
- At the eval default (5 steps, η 1) the student reaches 0.2410.

## Inference time

One sim, one GCD, mean of 5 rollouts after a warm-up.
The times come from a scratch script, `/scratch/project_465003416/lc_work/scratch_tests/research_dit/time_rollout.py`,
which is not in the repo.

| run | 2 steps: DiT | + VAE decode | 5 steps η 1: DiT | + VAE decode |
|---|---|---|---|---|
| res256/pairs/unrolled/unrolled2 | 0.437 s | 3.63 s | 1.088 s | 4.29 s |
| res256/pairs/pruned/inner_c2_f3776 | 0.341 s | 3.55 s | 0.847 s | 4.06 s |

- Half the parameters make the DiT only 1.28× faster: the cut is in the FFN and the cross-attention, self-attention
  over ≈ 3300 tokens stays.
- The VAE decode dominates a rollout (3.2 s), so the whole rollout is 2–5 % faster.

## Reviewer verdicts

One fresh reviewer per GPU job before submission.

| job | verdict | reason |
|---|---|---|
| 23498015 smoke, every cut at once | change | 20 min breaks the dev-g limit and a 16 % model cannot size the pilots: split into a coverage smoke (4 updates) and a sizing smoke (arm A) |
| 23498221 smoke, arm A sizing | change | s/update and calibration time were not logged: added to the progress and prune lines first |
| 23498362 pilot A, depth 14 spread | go | every-other is the strong baseline and drops blocks the scan rates costly, so it contrasts with the scan choice |
| pilot I, heads 8 / cross 4 / FFN 4288 | change | one of six equal-size splits, none measured and the refit unchecked on I: first a cut-only job over the splits |
| cut-only comparison of six I splits and W | go | cheapest way to choose the I split and to check the damped refit on width cuts; rule: pilot the best refit split, the more balanced one within 0.01 rel MSE, no refit where it is worse than plain |
| cut-only rerun with fp32 grams | go | the first run's grams were summed in bf16 under autocast, which broke every width refit and the importance ranking |
| pilot I, heads 12 / cross 2 / FFN 3776 | change | write prune.heads=12 explicitly: the cut-only number was measured with it |
| stop 23498362 pilot A spread at step 1100 | stopped | val 0.517 at step 500 vs 0.293 for pilot I, 0.454 at 1000; dit_1000.pt kept |
| pilot W, stream 760 | change | worst start of all arms: decide at step 500 only (same 1500-update schedule, cancelled after val 500), no 500-sim test |
| pilot A scan | change | step 500 answers scan vs every-other: cancel after val 500 unless close to I |
| stop 23499597 pilot W at step ~270 | stopped | kd_velocity flat at ~1.0 and flow 0.7 over 260 updates (A: 0.13, I: 0.02 at step 200); no val reached |
| stop 23500120 pilot A scan after val 500 | stopped (as planned) | val 0.490 vs 0.517 every-other and 0.293 for I at step 500; dit_500.pt kept |
| long run of I, 6000 more updates | change | target unlikely (pilot gained 0.003 from 1000 to 1500), still worth it: stop after val 3000 if above 0.268 (target val ≈ 0.259 from I's test/val ratio 0.935) |
| teacher 5-step η 1 eval + rollout timing | go | unrolled2 has no 5-step η 1 number and the slot is free; timing also at 5 steps |
| gradual A 30 → 22 → 14 | change | A is out (0.45 vs 0.27): test gradual vs one-shot on I instead (cross 12 → 6, FFN 6000, 500 updates; then the final cut, to 1500 in total) |
| gradual I stage 2 | go | the only direct gradual-vs-one-shot test at equal updates, ~1.4 node-h; stage 1 already led at 500 (0.258 vs 0.293) |
| long run of I at val 3000 | continued | 0.267 ≤ 0.268 (stop rule); 0.279 / 0.270 / 0.267 at 1000 / 2000 / 3000 |
| final student 5-step η 1 eval | go | the eval default needs the student's own number for a paired CI |
| stretch ~40 % cut-only (from the final student) | change | only if the 2-step test ≤ 0.242; drop 8/2/3264; pilot stop rules: val 1000 > 0.285, long run val 3000 > I's 0.267 + 0.01 |
| continue I long run, 4000 at 3e-6 (teacher unrolled2 recipe) | change | test moved ~0.37× val for the teacher: stop at val 2000 unless ≤ 0.2555; skip the test unless end val ≤ 0.2545 |
| stop 23509172 continuation at val 2000 | stopped (rule) | val 0.259 / 0.257 at 1000 / 2000, not ≤ 0.2555: the student has plateaued at val 0.257; dit_1000.pt kept |
