# Leaderboard (classification, primary metric = accuracy, higher is better)

**Warnings**

- the target holds 37 distinct whole numbers and was read as classification; if it is a regression target, run again with --task regression
- top1 was trained only 4 epochs: its rank is a rough early estimate. Increase --budget, or reduce --zoo / --per-family.
- Only 2 of 92 candidates could be trained within the budget; raise --budget, or lower --zoo / --per-family, or use --seeds N.
- Only 2 of the 3 requested models could be exported (too few trained candidates).

| rank | model | roles | group | epochs | runs | params | MFLOPs | MB | lr | wd | batch | val (mean+/-noise) | test |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | top1_zoo_gfnet_block_patchify_pos | top1, most-stable | 0 | 4 | 3 | 482,549 | 24.7 | 200 | 0.003 | 0.01 | 64 | 0.10766 +/- 0.01139 | acc 0.1009, bal 0.1009, kappa 0.07593 (+/- 0.00665) |
| 2 | top2_resnet | best-small, most-different | 1 | 4 | 3 | 45,261 | 54.2 | 215 | 0.001 | 0.01 | 32 | 0.05676 +/- 0.01103 | acc 0.06216, bal 0.06216, kappa 0.03611 (+/- 0.00441) |

Each number is a mean over `runs` (seeds); +/- is the larger of the std over those runs and the sampling error of the validation set (the only noise estimate when runs = 1). Models are ranked within the same `epochs` (more epochs first). Finalists were ranked on 1 split(s).

## Reference baselines (same split(s))

| model | val | test |
|---|---|---|
| random_forest | 0.12432 | acc 0.09189, bal 0.09189, kappa 0.06667 |
| hist_gradient_boosting | 0.11081 | acc 0.09865, bal 0.09865, kappa 0.07361 |
| logistic_regression | 0.10135 | acc 0.08514, bal 0.08514, kappa 0.05972 |

## Verdict

**No significant win over the best reference (random_forest).** Val margin -0.01667 vs noise 0.01139; test margin 0.00225 vs noise 0.01107. Consider the reference baseline first; the deep-learning candidates below are still valid to try.

## Objective: accuracy

Pareto front of (validation, parameters) among the ranked finalists:

| family | val | params | FLOPs |
|---|---|---|---|
| resnet | 0.05676 | 45,261 | 54,200,640 |
| zoo_gfnet_block_patchify_pos | 0.10766 | 482,549 | 24,668,800 |

## Budget: 10m (seconds)

Wall clock 587.7 s, 42.0 epochs trained, overrun -2.0%.

| stage | seconds | epochs |
|---|---|---|
| load | 30.19 | 0 |
| reference | 29.75 | 0 |
| generate | 15.82 | 0 |
| proxies | 72.85 | 0 |
| seed_selection | 2.12 | 0 |
| search | 152.16 | 16 |
| evolution | 0.0 | 0 |
| runs | 114.56 | 8 |
| hparams | 170.18 | 18 |
| export | 0.0 | 0 |
