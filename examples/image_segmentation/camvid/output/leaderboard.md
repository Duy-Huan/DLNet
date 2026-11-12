# Leaderboard (classification, primary metric = miou, higher is better)

**Warnings**

- top1 was trained only 4 epochs: its rank is a rough early estimate. Increase --budget, or reduce --zoo / --per-family.
- Only 1 of 100 candidates could be trained within the budget; raise --budget, or lower --zoo / --per-family, or use --seeds N.
- Only 1 of the 3 requested models could be exported (too few trained candidates).

| rank | model | roles | group | epochs | runs | params | MFLOPs | MB | lr | wd | batch | val (mean+/-noise) | test |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | top1_fcn | top1 | 0 | 4 | 3 | 20,707 | 89.6 | 4529 | 0.003 | 0.01 | 64 | 0.09823 +/- 0.02515 | mIoU 0.0956, pixel acc 0.44205, kappa 0.23055 (+/- 0.00652) |

Each number is a mean over `runs` (seeds); +/- is the larger of the std over those runs and the sampling error of the validation set (the only noise estimate when runs = 1). Models are ranked within the same `epochs` (more epochs first). Finalists were ranked on 1 split(s).

## Reference baselines (same split(s))

| model | val | test |
|---|---|---|
| pixel_random_forest_5x5 | 0.26623 | mIoU 0.26843, pixel acc 0.70207, kappa 0.61361 |
| pixel_logistic_regression_5x5 | 0.15113 | mIoU 0.14597, pixel acc 0.50583, kappa 0.33983 |

## Verdict

**No significant win over the best reference (pixel_random_forest_5x5).** Val margin -0.168 vs noise 0.02515; test margin -0.17283 vs noise 0.02476. Consider the reference baseline first; the deep-learning candidates below are still valid to try.

## Objective: accuracy

Pareto front of (validation, parameters) among the ranked finalists:

| family | val | params | FLOPs |
|---|---|---|---|
| fcn | 0.09823 | 20,707 | 89,617,920 |

## Budget: 10m (seconds)

Wall clock 427.5 s, 16.0 epochs trained, overrun -28.8%.

| stage | seconds | epochs |
|---|---|---|
| load | 12.48 | 0 |
| reference | 43.2 | 0 |
| generate | 17.32 | 0 |
| proxies | 65.2 | 0 |
| seed_selection | 0.0 | 0 |
| search | 151.1 | 8 |
| evolution | 0.0 | 0 |
| runs | 138.1 | 8 |
| hparams | 0.0 | 0 |
| export | 0.0 | 0 |
