# Leaderboard (classification, primary metric = accuracy, higher is better)

**Warnings**

- Some models were trained only once: their +/- is the sampling error of the validation set, not run-to-run noise. A larger --budget repeats the finalists.
- The Top-K mixes models trained [4, 30] epochs (the budget ended before they could be trained to the same length); compare scores only within the same epochs.
- Only 3 of 92 candidates could be trained within the budget; raise --budget, or lower --zoo / --per-family, or use --seeds N.

| rank | model | roles | group | epochs | runs | params | MFLOPs | MB | lr | wd | batch | val (mean+/-noise) | test |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | top1_zoo_unet_res_block | top1 | 0 | 30 | 2 | 171,562 | 36.8 | 160 | 0.003 | 0.0001 | 64 | 0.85292 +/- 0.01022 | acc 0.86792, bal 0.86792, kappa 0.85324 (+/- 0.00375) |
| 2 | top2_zoo_gfnet_block_patchify_pos | most-different | 1 | 30 | 2 | 369,386 | 5.9 | 61 | 0.003 | 0.01 | 64 | 0.82875 +/- 0.01088 | acc 0.82417, bal 0.82417, kappa 0.80463 (+/- 0.00167) |
| 3 | top3_resnet |  | 2 | 4 | 1 | 707,074 | 161.0 | 339 | 0.003 | 0.01 | 64 | 0.64167 +/- 0.01384 | acc 0.64333, bal 0.64333, kappa 0.6037 (+/- 0.0) |

Each number is a mean over `runs` (seeds); +/- is the larger of the std over those runs and the sampling error of the validation set (the only noise estimate when runs = 1). Models are ranked within the same `epochs` (more epochs first). Finalists were ranked on 1 split(s).

## Reference baselines (same split(s))

| model | val | test |
|---|---|---|
| hist_gradient_boosting | 0.82917 | acc 0.83167, bal 0.83167, kappa 0.81296 |
| logistic_regression | 0.82167 | acc 0.82, bal 0.82, kappa 0.8 |
| random_forest | 0.8125 | acc 0.825, bal 0.825, kappa 0.80556 |

## Verdict

top1 beats the best reference (hist_gradient_boosting): val margin 0.02375 (> noise 0.01022) and test margin 0.03625 (> noise 0.00977).

## Objective: accuracy

Pareto front of (validation, parameters) among the ranked finalists:

| family | val | params | FLOPs |
|---|---|---|---|
| zoo_unet_res_block | 0.85292 | 171,562 | 36,780,288 |

## Zero-cost proxies (information only, nothing was pruned)

Spearman rank correlation with the validation score after 4 epochs (rung 1):

| proxy | Spearman | candidates |
|---|---|---|
| naswot | None | 2 |
| synflow | -1.0 | 3 |
| grad_norm | -0.5 | 3 |
| snip | -0.5 | 3 |
| zico | -1.0 | 3 |
| params | -1.0 | 3 |
| vote | -0.5 | 3 |
| seed_prior | 0.5 | 3 |

Clearly better than the parameter count: none.

## Budget: 10m (seconds)

Wall clock 565.1 s, 136.0 epochs trained, overrun -5.8%.

| stage | seconds | epochs |
|---|---|---|
| load | 3.01 | 0 |
| reference | 9.78 | 0 |
| generate | 6.92 | 0 |
| proxies | 57.95 | 0 |
| seed_selection | 1.34 | 0 |
| search | 211.82 | 58 |
| evolution | 0.0 | 0 |
| equalize | 201.81 | 48 |
| runs | 72.46 | 30 |
| hparams | 0.0 | 0 |
| export | 0.0 | 0 |
