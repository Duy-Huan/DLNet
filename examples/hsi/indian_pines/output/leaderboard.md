# Leaderboard (classification, primary metric = accuracy, higher is better)

**Warnings**

- --split pixel with --patch 9: validation/test pixels sit inside the patches of training pixels, so scores are optimistic (spatial leakage). Use --split block for an honest estimate of performance on new areas.
- Only 16 of 188 candidates could be trained within the budget; raise --budget, or lower --zoo / --per-family, or use --seeds N.

| rank | model | roles | group | epochs | runs | params | MFLOPs | MB | lr | wd | batch | val (mean+/-noise) | test |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | top1_zoo_gfnet_block_spectral_gf | top1 | 0 | 30 | 3 | 57,584 | 8.1 | 28 | 0.001 | 0.001 | 32 | 0.98504 +/- 0.00379 | acc 0.98609, bal 0.98352, kappa 0.98414 (+/- 0.00138) |
| 2 | top2_zoo_gfnet_block_spectral_gf_vgg_stage | most-stable, most-different | 0 | 30 | 2 | 170,672 | 25.4 | 62 | 0.001 | 0.001 | 32 | 0.98488 +/- 0.00381 | acc 0.9867, bal 0.98635, kappa 0.98484 (+/- 0.00012) |
| 3 | top3_spec_fft2d |  | 0 | 30 | 3 | 65,488 | 10.5 | 14 | 0.003 | 0.0001 | 32 | 0.98276 +/- 0.00407 | acc 0.98349, bal 0.929, kappa 0.98117 (+/- 0.00209) |

Each number is a mean over `runs` (seeds); +/- is the larger of the std over those runs and the sampling error of the validation set (the only noise estimate when runs = 1). Models are ranked within the same `epochs` (more epochs first). Finalists were ranked on 1 split(s).

## Reference baselines (same split(s))

| model | val | test |
|---|---|---|
| random_forest_spectral_spatial | 0.96 | acc 0.9534, bal 0.89898, kappa 0.9467 |
| svm_rbf_spectral_spatial | 0.84488 | acc 0.83728, bal 0.71824, kappa 0.81369 |
| random_forest_spectral | 0.69854 | acc 0.70188, bal 0.5439, kappa 0.65137 |
| svm_rbf_spectral | 0.68488 | acc 0.68309, bal 0.57623, kappa 0.63614 |

## Verdict

top1 beats the best reference (random_forest_spectral_spatial): val margin 0.02504 (> noise 0.00379) and test margin 0.03269 (> noise 0.00129).

## Objective: accuracy

Pareto front of (validation, parameters) among the ranked finalists:

| family | val | params | FLOPs |
|---|---|---|---|
| gfnet2d | 0.98049 | 31,952 | 4,054,912 |
| resnet2d | 0.98114 | 46,352 | 7,372,672 |
| zoo_gfnet_block_spectral_gf | 0.98504 | 57,584 | 8,109,824 |

## Zero-cost proxies (information only, nothing was pruned)

Spearman rank correlation with the validation score after 4 epochs (rung 1):

| proxy | Spearman | candidates |
|---|---|---|
| naswot | -0.124 | 14 |
| synflow | 0.055 | 13 |
| grad_norm | 0.673 | 14 |
| snip | 0.673 | 14 |
| zico | 0.194 | 14 |
| params | 0.786 | 16 |
| seed_prior | None | 16 |

Clearly better than the parameter count: none.

## Budget: 10m (seconds)

Wall clock 592.5 s, 708.0 epochs trained, overrun -1.3%.

| stage | seconds | epochs |
|---|---|---|
| load | 1.44 | 0 |
| reference | 6.75 | 0 |
| generate | 20.91 | 0 |
| proxies | 63.19 | 0 |
| seed_selection | 0.0 | 0 |
| search | 217.96 | 310 |
| evolution | 177.85 | 248 |
| runs | 104.24 | 150 |
| hparams | 0.0 | 0 |
| export | 0.0 | 0 |
