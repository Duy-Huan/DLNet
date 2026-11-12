# Leaderboard (classification, primary metric = accuracy, higher is better)

**Warnings**

- --split pixel with --patch 13: validation/test pixels sit inside the patches of training pixels, so scores are optimistic (spatial leakage). Use --split block for an honest estimate of performance on new areas.
- Only 9 of 180 candidates could be trained within the budget; raise --budget, or lower --zoo / --per-family, or use --seeds N.

| rank | model | roles | group | epochs | runs | params | MFLOPs | MB | lr | wd | batch | val (mean+/-noise) | test |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | top1_tucker_cnn2d | top1 | 0 | 30 | 3 | 40,080 | 10.4 | 40 | 0.001 | 0.0001 | 64 | 0.98472 +/- 0.00383 | acc 0.98284, bal 0.96244, kappa 0.98043 (+/- 0.00352) |
| 2 | top2_zoo_fused_mbconv_pool_flatten | most-different | 0 | 30 | 2 | 103,504 | 31.7 | 32 | 0.003 | 0.0001 | 32 | 0.98341 +/- 0.00399 | acc 0.98396, bal 0.97094, kappa 0.98171 (+/- 0.00055) |
| 3 | top3_zoo_conv_bn_act_gap | best-small, most-stable | 0 | 30 | 2 | 27,792 | 9.2 | 23 | 0.003 | 0.001 | 64 | 0.98146 +/- 0.00421 | acc 0.97573, bal 0.85696, kappa 0.97232 (+/- 0.00146) |

Each number is a mean over `runs` (seeds); +/- is the larger of the std over those runs and the sampling error of the validation set (the only noise estimate when runs = 1). Models are ranked within the same `epochs` (more epochs first). Finalists were ranked on 1 split(s).

## Reference baselines (same split(s))

| model | val | test |
|---|---|---|
| random_forest_spectral_spatial | 0.97073 | acc 0.96719, bal 0.95454, kappa 0.96255 |
| svm_rbf_spectral_spatial | 0.8439 | acc 0.83996, bal 0.72133, kappa 0.8168 |
| random_forest_spectral | 0.69854 | acc 0.70188, bal 0.5439, kappa 0.65137 |
| svm_rbf_spectral | 0.68488 | acc 0.68309, bal 0.57623, kappa 0.63614 |

## Verdict

top1 beats the best reference (random_forest_spectral_spatial): val margin 0.01398 (> noise 0.00383) and test margin 0.01565 (> noise 0.00143).

## Objective: accuracy

Pareto front of (validation, parameters) among the ranked finalists:

| family | val | params | FLOPs |
|---|---|---|---|
| dwsep2d | 0.96976 | 14,992 | 4,695,168 |
| zoo_conv_bn_act_gap | 0.98146 | 27,792 | 9,151,360 |
| tucker_cnn2d | 0.98472 | 40,080 | 10,358,528 |

## Zero-cost proxies (information only, nothing was pruned)

Spearman rank correlation with the validation score after 4 epochs (rung 1):

| proxy | Spearman | candidates |
|---|---|---|
| naswot | 0.029 | 6 |
| synflow | 0.6 | 6 |
| grad_norm | 0.886 | 6 |
| snip | 0.943 | 6 |
| zico | 0.486 | 6 |
| params | 0.867 | 9 |
| seed_prior | None | 9 |

Clearly better than the parameter count: none.

## Budget: 6m (seconds)

Wall clock 361.3 s, 406.0 epochs trained, overrun 0.4%.

| stage | seconds | epochs |
|---|---|---|
| load | 1.34 | 0 |
| reference | 6.88 | 0 |
| generate | 31.3 | 0 |
| proxies | 36.1 | 0 |
| seed_selection | 0.0 | 0 |
| search | 121.24 | 192 |
| evolution | 115.4 | 124 |
| equalize | 0.0 | 0 |
| runs | 48.95 | 90 |
| hparams | 0.0 | 0 |
| export | 0.0 | 0 |

## Paper baselines

Same data, splits, 3 run(s) and 30 epochs as the top-1 baseline (`top1_tucker_cnn2d`: val 0.9847 +- 0.0028, test 0.9828, 40,080 params). Fidelity: A = the authors' class, B = verified against the authors' code, C = written from the paper, not verified. "Paper reports" is copied from the paper: another split, preprocessing and budget, not comparable with the measured columns.

| Network | Fidelity | Params | Paper reports | Common protocol: val accuracy | test | Authors' recipe: test |
|---|---|---|---|---|---|---|
| SSFTT | B | 153,224 | - | 0.9730 +- 0.0051 | 0.9772 | 0.9810 (100 of 100 epochs) |
