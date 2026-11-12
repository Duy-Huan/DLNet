# Leaderboard (classification, primary metric = accuracy, higher is better)

**Warnings**

- Some models were trained only once: their +/- is the sampling error of the validation set, not run-to-run noise. A larger --budget repeats the finalists.
- Only 4 of 92 candidates could be trained within the budget; raise --budget, or lower --zoo / --per-family, or use --seeds N.

| rank | model | roles | group | epochs | runs | params | MFLOPs | MB | lr | wd | batch | val (mean+/-noise) | test |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | top1_vgg | top1 | 0 | 30 | 1 | 26,682 | 41.3 | 145 | 0.001 | 0.0001 | 32 | 0.605 +/- 0.01996 | acc 0.595, bal 0.595, kappa 0.55 (+/- 0.0) |
| 2 | top2_zoo_gfnet_block_patchify_pos | most-different | 1 | 30 | 1 | 480,794 | 24.7 | 200 | 0.003 | 0.01 | 64 | 0.54167 +/- 0.02034 | acc 0.54167, bal 0.54167, kappa 0.49074 (+/- 0.0) |
| 3 | top3_resnet |  | 2 | 30 | 1 | 44,370 | 54.2 | 215 | 0.001 | 0.01 | 32 | 0.49333 +/- 0.02041 | acc 0.48, bal 0.48, kappa 0.42222 (+/- 0.0) |

Each number is a mean over `runs` (seeds); +/- is the larger of the std over those runs and the sampling error of the validation set (the only noise estimate when runs = 1). Models are ranked within the same `epochs` (more epochs first). Finalists were ranked on 1 split(s).

## Reference baselines (same split(s))

| model | val | test |
|---|---|---|
| random_forest | 0.41833 | acc 0.385, bal 0.385, kappa 0.31667 |
| hist_gradient_boosting | 0.41333 | acc 0.36, bal 0.36, kappa 0.28889 |
| logistic_regression | 0.365 | acc 0.32667, bal 0.32667, kappa 0.25185 |

## Verdict

top1 beats the best reference (random_forest): val margin 0.18667 (> noise 0.01996) and test margin 0.21 (> noise 0.02004).

## Objective: accuracy

Pareto front of (validation, parameters) among the ranked finalists:

| family | val | params | FLOPs |
|---|---|---|---|
| vgg | 0.605 | 26,682 | 41,288,960 |

## Zero-cost proxies (information only, nothing was pruned)

Spearman rank correlation with the validation score after 4 epochs (rung 1):

| proxy | Spearman | candidates |
|---|---|---|
| naswot | None | 1 |
| synflow | None | 1 |
| grad_norm | None | 1 |
| snip | None | 1 |
| zico | None | 1 |
| params | 0.632 | 4 |
| seed_prior | -0.775 | 4 |

Clearly better than the parameter count: ['seed_prior'].

## Budget: 10m (seconds)

Wall clock 601.3 s, 120.0 epochs trained, overrun 0.2%.

| stage | seconds | epochs |
|---|---|---|
| load | 23.97 | 0 |
| reference | 15.86 | 0 |
| generate | 10.8 | 0 |
| proxies | 77.97 | 0 |
| seed_selection | 1.53 | 0 |
| search | 193.98 | 54 |
| evolution | 0.0 | 0 |
| equalize | 196.05 | 44 |
| runs | 25.7 | 8 |
| hparams | 55.36 | 14 |
| export | 0.0 | 0 |
