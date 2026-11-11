# Leaderboard (classification, primary metric = accuracy, higher is better)

**Warnings**

- Only 20 of 188 candidates could be trained within the budget; raise --budget, or lower --zoo / --per-family, or use --seeds N.

| rank | model | roles | group | epochs | runs | params | MFLOPs | MB | lr | wd | batch | val (mean+/-noise) | test |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | top1_zoo_afno_block | top1 | 0 | 30 | 2 | 37,104 | 5.8 | 72 | 0.003 | 0.001 | 64 | 0.4668 +/- 0.04301 | acc 0.51312, bal 0.47212, kappa 0.4329 (+/- 0.01475) |
| 2 | top2_afno2d | most-stable, most-different | 0 | 30 | 3 | 36,592 | 5.7 | 72 | 0.003 | 0.001 | 64 | 0.41484 +/- 0.02124 | acc 0.45082, bal 0.44754, kappa 0.38036 (+/- 0.09342) |
| 3 | top3_zoo_gfnet_block | best-small | 0 | 30 | 2 | 32,912 | 4.2 | 26 | 0.003 | 0.001 | 32 | 0.41367 +/- 0.03077 | acc 0.46287, bal 0.48544, kappa 0.37242 (+/- 0.0057) |

Each number is a mean over `runs` (seeds); +/- is the larger of the std over those runs and the sampling error of the validation set (the only noise estimate when runs = 1). Models are ranked within the same `epochs` (more epochs first). Finalists were ranked on 3 split(s), so the std includes split-to-split noise.

## Reference baselines (same split(s))

| model | val | test |
|---|---|---|
| random_forest_spectral_spatial | 0.28901 | acc 0.43313, bal 0.43957, kappa 0.35684 |
| random_forest_spectral | 0.25997 | acc 0.39947, bal 0.38832, kappa 0.32155 |
| svm_rbf_spectral | 0.24702 | acc 0.39247, bal 0.38639, kappa 0.3137 |
| svm_rbf_spectral_spatial | 0.24122 | acc 0.4113, bal 0.40136, kappa 0.33479 |

## Verdict

top1 beats the best reference (random_forest_spectral_spatial): val margin 0.17779 (> noise 0.04301) and test margin 0.07999 (> noise 0.00832).

## Objective: accuracy

Pareto front of (validation, parameters) among the ranked finalists:

| family | val | params | FLOPs |
|---|---|---|---|
| cnn3d | 0.36617 | 4,296 | 5,253,408 |
| zoo_afno_block | 0.4533 | 27,472 | 4,239,232 |
| zoo_afno_block | 0.4668 | 37,104 | 5,750,656 |

## Zero-cost proxies (information only, nothing was pruned)

Spearman rank correlation with the validation score after 4 epochs (rung 1):

| proxy | Spearman | candidates |
|---|---|---|
| naswot | -0.294 | 14 |
| synflow | -0.346 | 14 |
| grad_norm | -0.148 | 15 |
| snip | 0.038 | 15 |
| zico | 0.578 | 15 |
| params | 0.199 | 20 |
| seed_prior | None | 20 |

Clearly better than the parameter count: ['synflow', 'zico'].

## Budget: 10m (seconds)

Wall clock 601.8 s, 660.0 epochs trained, overrun 0.3%.

| stage | seconds | epochs |
|---|---|---|
| load | 2.41 | 0 |
| reference | 12.09 | 0 |
| generate | 22.9 | 0 |
| proxies | 64.47 | 0 |
| seed_selection | 0.0 | 0 |
| search | 204.49 | 296 |
| evolution | 173.71 | 244 |
| runs | 121.67 | 120 |
| hparams | 0.0 | 0 |
| export | 0.0 | 0 |
