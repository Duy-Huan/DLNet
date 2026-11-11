# Leaderboard (classification, primary metric = accuracy, higher is better)

**Warnings**

- --split pixel with --patch 9: validation/test pixels sit inside the patches of training pixels, so scores are optimistic (spatial leakage). Use --split block for an honest estimate of performance on new areas.
- Some models were trained only once: their +/- is the sampling error of the validation set, not run-to-run noise. A larger --budget repeats the finalists.
- Only 14 of 188 candidates could be trained within the budget; raise --budget, or lower --zoo / --per-family, or use --seeds N.

| rank | model | roles | group | epochs | runs | params | MFLOPs | MB | lr | wd | batch | val (mean+/-noise) | test |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | top1_afno2d | top1 | 0 | 30 | 1 | 89,744 | 14.2 | 100 | 0.001 | 0.0001 | 64 | 0.99723 +/- 0.00302 | acc 0.99613, bal 0.9971, kappa 0.99569 (+/- 0.0) |
| 2 | top2_ffc2d | best-small, most-different | 0 | 30 | 1 | 78,288 | 12.3 | 23 | 0.001 | 0.001 | 32 | 0.99631 +/- 0.00302 | acc 0.99261, bal 0.98842, kappa 0.99177 (+/- 0.0) |
| 3 | top3_cnn2d |  | 0 | 30 | 3 | 100,624 | 14.8 | 14 | 0.003 | 0.0001 | 32 | 0.99508 +/- 0.00302 | acc 0.99264, bal 0.9957, kappa 0.9918 (+/- 0.00215) |

Each number is a mean over `runs` (seeds); +/- is the larger of the std over those runs and the sampling error of the validation set (the only noise estimate when runs = 1). Models are ranked within the same `epochs` (more epochs first). Finalists were ranked on 1 split(s).

## Reference baselines (same split(s))

| model | val | test |
|---|---|---|
| random_forest_spectral_spatial | 0.97692 | acc 0.97544, bal 0.98423, kappa 0.97263 |
| svm_rbf_spectral_spatial | 0.94922 | acc 0.94361, bal 0.96214, kappa 0.93714 |
| random_forest_spectral | 0.89566 | acc 0.89429, bal 0.9306, kappa 0.88178 |
| svm_rbf_spectral | 0.88273 | acc 0.88286, bal 0.90426, kappa 0.86883 |

## Verdict

top1 beats the best reference (random_forest_spectral_spatial): val margin 0.02031 (> noise 0.00302) and test margin 0.02069 (> noise 0.00044).

## Objective: accuracy

Pareto front of (validation, parameters) among the ranked finalists:

| family | val | params | FLOPs |
|---|---|---|---|
| gfnet2d | 0.99261 | 31,952 | 4,054,912 |
| resnet2d | 0.99446 | 46,352 | 7,372,672 |
| ffc2d | 0.99631 | 78,288 | 12,275,456 |
| afno2d | 0.99723 | 89,744 | 14,155,520 |

## Zero-cost proxies (information only, nothing was pruned)

Spearman rank correlation with the validation score after 4 epochs (rung 1):

| proxy | Spearman | candidates |
|---|---|---|
| naswot | -0.273 | 12 |
| synflow | 0.182 | 11 |
| grad_norm | 0.727 | 12 |
| snip | 0.643 | 12 |
| zico | 0.294 | 12 |
| params | 0.666 | 14 |
| seed_prior | None | 14 |

Clearly better than the parameter count: none.

## Budget: 10m (seconds)

Wall clock 592.4 s, 272.0 epochs trained, overrun -1.3%.

| stage | seconds | epochs |
|---|---|---|
| load | 5.01 | 0 |
| reference | 23.19 | 0 |
| generate | 21.21 | 0 |
| proxies | 63.88 | 0 |
| seed_selection | 0.0 | 0 |
| search | 283.96 | 158 |
| evolution | 0.0 | 0 |
| equalize | 143.2 | 54 |
| runs | 51.9 | 60 |
| hparams | 0.01 | 0 |
| export | 0.0 | 0 |
