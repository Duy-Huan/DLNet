# Leaderboard (classification, primary metric = accuracy, higher is better)

**Warnings**

- --split pixel with --patch 9: validation/test pixels sit inside the patches of training pixels, so scores are optimistic (spatial leakage). Use --split block for an honest estimate of performance on new areas.
- Only 18 of 188 candidates could be trained within the budget; raise --budget, or lower --zoo / --per-family, or use --seeds N.

| rank | model | roles | group | epochs | runs | params | MFLOPs | MB | lr | wd | batch | val (mean+/-noise) | test |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | top1_spec_fft2d | top1, most-stable | 0 | 30 | 3 | 65,257 | 10.5 | 14 | 0.003 | 0.0001 | 32 | 0.99378 +/- 0.0034 | acc 0.99286, bal 0.98521, kappa 0.99053 (+/- 0.00188) |
| 2 | top2_zoo_afno_block_moe | best-small, most-different | 0 | 30 | 3 | 36,427 | 5.7 | 72 | 0.003 | 0.001 | 64 | 0.993 +/- 0.0034 | acc 0.99209, bal 0.98353, kappa 0.9895 (+/- 0.00148) |
| 3 | top3_ffc2d |  | 0 | 30 | 2 | 77,833 | 12.3 | 23 | 0.001 | 0.001 | 32 | 0.99183 +/- 0.0034 | acc 0.9925, bal 0.98443, kappa 0.99006 (+/- 0.0008) |

Each number is a mean over `runs` (seeds); +/- is the larger of the std over those runs and the sampling error of the validation set (the only noise estimate when runs = 1). Models are ranked within the same `epochs` (more epochs first). Finalists were ranked on 1 split(s).

## Reference baselines (same split(s))

| model | val | test |
|---|---|---|
| random_forest_spectral_spatial | 0.95099 | acc 0.95538, bal 0.93011, kappa 0.94021 |
| svm_rbf_spectral_spatial | 0.92882 | acc 0.93921, bal 0.89689, kappa 0.91874 |
| svm_rbf_spectral | 0.87048 | acc 0.85719, bal 0.79143, kappa 0.8076 |
| random_forest_spectral | 0.81797 | acc 0.82195, bal 0.731, kappa 0.75424 |

## Verdict

top1 beats the best reference (random_forest_spectral_spatial): val margin 0.04278 (> noise 0.0034) and test margin 0.03747 (> noise 0.00049).

## Objective: accuracy

Pareto front of (validation, parameters) among the ranked finalists:

| family | val | params | FLOPs |
|---|---|---|---|
| zoo_gfnet_block | 0.98833 | 20,361 | 2,727,360 |
| zoo_afno_block_moe | 0.993 | 36,427 | 5,658,176 |
| spec_fft2d | 0.99378 | 65,257 | 10,451,520 |

## Zero-cost proxies (information only, nothing was pruned)

Spearman rank correlation with the validation score after 4 epochs (rung 1):

| proxy | Spearman | candidates |
|---|---|---|
| naswot | -0.181 | 14 |
| synflow | 0.096 | 14 |
| grad_norm | 0.471 | 15 |
| snip | 0.534 | 15 |
| zico | 0.397 | 15 |
| params | 0.614 | 18 |
| seed_prior | None | 18 |

Clearly better than the parameter count: none.

## Budget: 10m (seconds)

Wall clock 621.2 s, 596.0 epochs trained, overrun 3.5%.

| stage | seconds | epochs |
|---|---|---|
| load | 4.77 | 0 |
| reference | 16.17 | 0 |
| generate | 20.49 | 0 |
| proxies | 72.55 | 0 |
| seed_selection | 0.01 | 0 |
| search | 207.24 | 318 |
| evolution | 230.96 | 248 |
| runs | 68.88 | 30 |
| hparams | 0.01 | 0 |
| export | 0.0 | 0 |
