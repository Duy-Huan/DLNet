# Leaderboard (classification, primary metric = accuracy, higher is better)

**Warnings**

- Only 4 of 92 candidates could be trained within the budget; raise --budget, or lower --zoo / --per-family, or use --seeds N.

| rank | model | roles | group | epochs | runs | params | MFLOPs | MB | lr | wd | batch | val (mean+/-noise) | test |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | top1_cnn | top1, most-stable | 0 | 30 | 3 | 72,954 | 19.8 | 92 | 0.001 | 0.01 | 64 | 0.61639 +/- 0.01404 | acc 0.62333, bal 0.62333, kappa 0.58148 (+/- 0.01016) |
| 2 | top2_zoo_ffn_patchify_pos | most-different | 1 | 30 | 3 | 26,890 | 4.1 | 14 | 0.001 | 0.01 | 32 | 0.48417 +/- 0.01443 | acc 0.47139, bal 0.47139, kappa 0.41265 (+/- 0.01059) |
| 3 | top3_zoo_fnet_block_patchify | best-small | 2 | 30 | 3 | 20,618 | 2.5 | 11 | 0.003 | 0.01 | 32 | 0.38722 +/- 0.01406 | acc 0.4125, bal 0.4125, kappa 0.34722 (+/- 0.00802) |

Each number is a mean over `runs` (seeds); +/- is the larger of the std over those runs and the sampling error of the validation set (the only noise estimate when runs = 1). Models are ranked within the same `epochs` (more epochs first). Finalists were ranked on 1 split(s).

## Reference baselines (same split(s))

| model | val | test |
|---|---|---|
| hist_gradient_boosting | 0.40333 | acc 0.4, bal 0.4, kappa 0.33333 |
| random_forest | 0.39917 | acc 0.41333, bal 0.41333, kappa 0.34815 |
| logistic_regression | 0.35083 | acc 0.38, bal 0.38, kappa 0.31111 |

## Verdict

top1 beats the best reference (hist_gradient_boosting): val margin 0.21306 (> noise 0.01404) and test margin 0.21 (> noise 0.01399).

## Objective: accuracy

Pareto front of (validation, parameters) among the ranked finalists:

| family | val | params | FLOPs |
|---|---|---|---|
| zoo_fnet_block_patchify | 0.38722 | 20,618 | 2,491,648 |
| zoo_ffn_patchify_pos | 0.48417 | 26,890 | 4,065,792 |
| cnn | 0.61639 | 72,954 | 19,760,384 |

## Zero-cost proxies (information only, nothing was pruned)

Spearman rank correlation with the validation score after 4 epochs (rung 1):

| proxy | Spearman | candidates |
|---|---|---|
| naswot | None | 2 |
| synflow | None | 2 |
| grad_norm | None | 2 |
| snip | None | 2 |
| zico | None | 2 |
| params | -0.4 | 4 |
| seed_prior | 0.316 | 4 |

Clearly better than the parameter count: none.

## Budget: 10m (seconds)

Wall clock 600.7 s, 308.0 epochs trained, overrun 0.1%.

| stage | seconds | epochs |
|---|---|---|
| load | 12.91 | 0 |
| reference | 17.07 | 0 |
| generate | 7.42 | 0 |
| proxies | 62.98 | 0 |
| seed_selection | 1.23 | 0 |
| search | 203.58 | 124 |
| evolution | 0.0 | 0 |
| equalize | 0.0 | 0 |
| runs | 295.47 | 184 |
| hparams | 0.0 | 0 |
| export | 0.0 | 0 |
