# Leaderboard (classification, primary metric = accuracy, higher is better)

**Warnings**

- Some models were trained only once: their +/- is the sampling error of the validation set, not run-to-run noise. A larger --budget repeats the finalists.
- Only 4 of 92 candidates could be trained within the budget; raise --budget, or lower --zoo / --per-family, or use --seeds N.

| rank | model | roles | group | epochs | runs | params | MFLOPs | MB | lr | wd | batch | val (mean+/-noise) | test |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | top1_cnn | top1 | 0 | 30 | 1 | 72,666 | 14.7 | 71 | 0.001 | 0.01 | 64 | 0.98667 +/- 0.00331 | acc 0.98833, bal 0.98833, kappa 0.98704 (+/- 0.0) |
| 2 | top2_resnet | most-different | 0 | 30 | 1 | 707,074 | 161.0 | 339 | 0.003 | 0.01 | 64 | 0.98333 +/- 0.0037 | acc 0.98667, bal 0.98667, kappa 0.98519 (+/- 0.0) |
| 3 | top3_zoo_gfnet_block_patchify_pos |  | 1 | 30 | 1 | 369,386 | 5.9 | 61 | 0.003 | 0.01 | 64 | 0.97 +/- 0.00492 | acc 0.96667, bal 0.96667, kappa 0.96296 (+/- 0.0) |

Each number is a mean over `runs` (seeds); +/- is the larger of the std over those runs and the sampling error of the validation set (the only noise estimate when runs = 1). Models are ranked within the same `epochs` (more epochs first). Finalists were ranked on 1 split(s).

## Reference baselines (same split(s))

| model | val | test |
|---|---|---|
| hist_gradient_boosting | 0.9325 | acc 0.92833, bal 0.92833, kappa 0.92037 |
| random_forest | 0.91667 | acc 0.92333, bal 0.92333, kappa 0.91481 |
| logistic_regression | 0.885 | acc 0.91333, bal 0.91333, kappa 0.9037 |

## Verdict

top1 beats the best reference (hist_gradient_boosting): val margin 0.05417 (> noise 0.00331) and test margin 0.06 (> noise 0.0031).

## Objective: accuracy

Pareto front of (validation, parameters) among the ranked finalists:

| family | val | params | FLOPs |
|---|---|---|---|
| zoo_fnet_block_patchify | 0.55083 | 18,570 | 1,707,264 |
| cnn | 0.98667 | 72,666 | 14,677,760 |

## Zero-cost proxies (information only, nothing was pruned)

Spearman rank correlation with the validation score after 4 epochs (rung 1):

| proxy | Spearman | candidates |
|---|---|---|
| naswot | None | 2 |
| synflow | None | 2 |
| grad_norm | None | 2 |
| snip | None | 2 |
| zico | None | 2 |
| params | 0.8 | 4 |
| seed_prior | 0.6 | 4 |

Clearly better than the parameter count: none.

## Budget: 10m (seconds)

Wall clock 605.9 s, 150.0 epochs trained, overrun 1.0%.

| stage | seconds | epochs |
|---|---|---|
| load | 3.03 | 0 |
| reference | 7.85 | 0 |
| generate | 6.72 | 0 |
| proxies | 61.9 | 0 |
| seed_selection | 1.1 | 0 |
| search | 206.99 | 124 |
| evolution | 0.0 | 0 |
| equalize | 318.19 | 26 |
| runs | 0.0 | 0 |
| hparams | 0.0 | 0 |
| export | 0.0 | 0 |
