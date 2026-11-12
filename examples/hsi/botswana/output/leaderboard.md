# Leaderboard (classification, primary metric = accuracy, higher is better)

**Warnings**

- --split pixel with --patch 9: validation/test pixels sit inside the patches of training pixels, so scores are optimistic (spatial leakage). Use --split block for an honest estimate of performance on new areas.
- Only 37 of 177 candidates could be trained within the budget; raise --budget, or lower --zoo / --per-family, or use --seeds N.

| rank | model | roles | group | epochs | runs | params | MFLOPs | MB | lr | wd | batch | val (mean+/-noise) | test |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | top1_spec_fft2d | top1, most-stable | 0 | 30 | 3 | 28,302 | 4.5 | 6 | 0.001 | 0.0001 | 32 | 0.99897 +/- 0.00552 | acc 0.99461, bal 0.99365, kappa 0.99416 (+/- 0.00083) |
| 2 | top2_zoo_ffn_gated_attn_block_gfnet_block | most-different | 0 | 30 | 3 | 119,950 | 19.9 | 66 | 0.003 | 0.01 | 32 | 0.99897 +/- 0.00552 | acc 0.9941, bal 0.99428, kappa 0.99361 (+/- 0.00018) |
| 3 | top3_zoo_fca_nin_block |  | 0 | 30 | 3 | 48,622 | 7.7 | 13 | 0.003 | 0.0001 | 32 | 0.99692 +/- 0.00552 | acc 0.99371, bal 0.9938, kappa 0.99319 (+/- 0.00192) |

Each number is a mean over `runs` (seeds); +/- is the larger of the std over those runs and the sampling error of the validation set (the only noise estimate when runs = 1). Models are ranked within the same `epochs` (more epochs first). Finalists were ranked on 1 split(s).

## Reference baselines (same split(s))

| model | val | test |
|---|---|---|
| random_forest_spectral_spatial | 0.99077 | acc 0.98922, bal 0.98398, kappa 0.98832 |
| svm_rbf_spectral_spatial | 0.96615 | acc 0.95266, bal 0.94446, kappa 0.94868 |
| random_forest_spectral | 0.90154 | acc 0.90377, bal 0.89933, kappa 0.89571 |
| svm_rbf_spectral | 0.85538 | acc 0.85104, bal 0.83888, kappa 0.83843 |

## Verdict

top1 beats the best reference (random_forest_spectral_spatial): val margin 0.00821 (> noise 0.00552) and test margin 0.00539 (> noise 0.00195).

## Objective: accuracy

Pareto front of (validation, parameters) among the ranked finalists:

| family | val | params | FLOPs |
|---|---|---|---|
| gfnet2d | 0.99487 | 20,526 | 2,727,680 |
| spec_fft2d | 0.99897 | 28,302 | 4,479,872 |

## Zero-cost proxies (information only, nothing was pruned)

Spearman rank correlation with the validation score after 4 epochs (rung 1):

| proxy | Spearman | candidates |
|---|---|---|
| naswot | 0.021 | 19 |
| synflow | 0.223 | 20 |
| grad_norm | 0.516 | 22 |
| snip | 0.401 | 22 |
| zico | 0.135 | 22 |
| params | 0.552 | 37 |
| seed_prior | -0.219 | 37 |

Clearly better than the parameter count: none.

## Budget: 10m (seconds)

Wall clock 540.5 s, 2529.9 epochs trained, overrun -9.9%.

| stage | seconds | epochs |
|---|---|---|
| load | 9.33 | 0 |
| reference | 1.45 | 0 |
| generate | 16.92 | 0 |
| proxies | 61.6 | 0 |
| seed_selection | 0.0 | 0 |
| search | 210.23 | 628 |
| evolution | 72.71 | 548 |
| runs | 0.0 | 0 |
| hparams | 168.23 | 1354 |
| export | 0.0 | 0 |
