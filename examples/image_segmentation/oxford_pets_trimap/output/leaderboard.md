# Leaderboard (classification, primary metric = miou, higher is better)

**Warnings**

- Some models were trained only once: their +/- is the sampling error of the validation set, not run-to-run noise. A larger --budget repeats the finalists.
- Only 2 of 100 candidates could be trained within the budget; raise --budget, or lower --zoo / --per-family, or use --seeds N.
- Only 2 of the 3 requested models could be exported (too few trained candidates).

| rank | model | roles | group | epochs | runs | params | MFLOPs | MB | lr | wd | batch | val (mean+/-noise) | test |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | top1_deconvnet | top1 | 0 | 30 | 2 | 22,323 | 62.2 | 373 | 0.003 | 0.0001 | 32 | 0.57849 +/- 0.0287 | mIoU 0.57689, pixel acc 0.81449, kappa 0.65766 (+/- 0.00615) |
| 2 | top2_deeplabv3 | best-small, most-different | 1 | 30 | 1 | 15,987 | 34.4 | 391 | 0.003 | 0.0001 | 32 | 0.53383 +/- 0.029 | mIoU 0.5355, pixel acc 0.80221, kappa 0.62939 (+/- 0.0) |

Each number is a mean over `runs` (seeds); +/- is the larger of the std over those runs and the sampling error of the validation set (the only noise estimate when runs = 1). Models are ranked within the same `epochs` (more epochs first). Finalists were ranked on 1 split(s).

## Reference baselines (same split(s))

| model | val | test |
|---|---|---|
| pixel_random_forest_5x5 | 0.356 | mIoU 0.35966, pixel acc 0.64552, kappa 0.30636 |
| pixel_logistic_regression_5x5 | 0.19894 | mIoU 0.19673, pixel acc 0.5565, kappa -0.02242 |

## Verdict

top1 beats the best reference (pixel_random_forest_5x5): val margin 0.22249 (> noise 0.0287) and test margin 0.21723 (> noise 0.02872).

## Objective: accuracy

Pareto front of (validation, parameters) among the ranked finalists:

| family | val | params | FLOPs |
|---|---|---|---|
| deeplabv3 | 0.53383 | 15,987 | 34,357,760 |
| deconvnet | 0.57849 | 22,323 | 62,227,840 |

## Budget: 10m (seconds)

Wall clock 590.6 s, 90.0 epochs trained, overrun -1.6%.

| stage | seconds | epochs |
|---|---|---|
| load | 18.89 | 0 |
| reference | 57.46 | 0 |
| generate | 76.59 | 0 |
| proxies | 64.03 | 0 |
| seed_selection | 0.0 | 0 |
| search | 149.46 | 34 |
| evolution | 0.0 | 0 |
| equalize | 112.73 | 26 |
| runs | 111.35 | 30 |
| hparams | 0.0 | 0 |
| export | 0.0 | 0 |
