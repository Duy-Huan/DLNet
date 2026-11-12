# Leaderboard (classification, primary metric = miou, higher is better)

**Warnings**

- top1 was trained only 4 epochs: its rank is a rough early estimate. Increase --budget, or reduce --zoo / --per-family.
- Only 1 of 100 candidates could be trained within the budget; raise --budget, or lower --zoo / --per-family, or use --seeds N.
- Only 1 of the 3 requested models could be exported (too few trained candidates).

| rank | model | roles | group | epochs | runs | params | MFLOPs | MB | lr | wd | batch | val (mean+/-noise) | test |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | top1_deeplabv3plus | top1 | 0 | 4 | 3 | 9,950 | 180.4 | 4161 | 0.001 | 0.0001 | 32 | 0.29558 +/- 0.18628 | mIoU 0.29843, pixel acc 0.59685, kappa 0.0 (+/- 0.13697) |

Each number is a mean over `runs` (seeds); +/- is the larger of the std over those runs and the sampling error of the validation set (the only noise estimate when runs = 1). Models are ranked within the same `epochs` (more epochs first). Finalists were ranked on 1 split(s).

## Reference baselines (same split(s))

| model | val | test |
|---|---|---|
| pixel_random_forest_5x5 | 0.65553 | mIoU 0.68483, pixel acc 0.87228, kappa 0.60245 |
| pixel_logistic_regression_5x5 | 0.62033 | mIoU 0.61854, pixel acc 0.84475, kappa 0.48926 |

## Verdict

**No significant win over the best reference (pixel_random_forest_5x5).** Val margin -0.35995 vs noise 0.18628; test margin -0.38641 vs noise 0.1868. Consider the reference baseline first; the deep-learning candidates below are still valid to try.

## Objective: accuracy

Pareto front of (validation, parameters) among the ranked finalists:

| family | val | params | FLOPs |
|---|---|---|---|
| deeplabv3plus | 0.29558 | 9,950 | 180,422,528 |

## Budget: 10m (seconds)

Wall clock 537.8 s, 12.0 epochs trained, overrun -10.4%.

| stage | seconds | epochs |
|---|---|---|
| load | 6.52 | 0 |
| reference | 5.19 | 0 |
| generate | 429.85 | 0 |
| proxies | 72.41 | 0 |
| seed_selection | 0.01 | 0 |
| search | 8.54 | 4 |
| evolution | 0.0 | 0 |
| runs | 15.19 | 8 |
| hparams | 0.0 | 0 |
| export | 0.0 | 0 |
