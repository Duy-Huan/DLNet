# Leaderboard (classification, primary metric = accuracy, higher is better)

**Warnings**

- --split pixel with --patch 9: validation/test pixels sit inside the patches of training pixels, so scores are optimistic (spatial leakage). Use --split block for an honest estimate of performance on new areas.
- Some models were trained only once: their +/- is the sampling error of the validation set, not run-to-run noise. A larger --budget repeats the finalists.
- Only 30 of 188 candidates could be trained within the budget; raise --budget, or lower --zoo / --per-family, or use --seeds N.

| rank | model | roles | group | epochs | runs | params | MFLOPs | MB | lr | wd | batch | val (mean+/-noise) | test |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | top1_mambahsi | top1 | 0 | 30 | 1 | 24,145 | 3.3 | 55 | 0.003 | 0.0001 | 64 | 0.96346 +/- 0.00823 | acc 0.95133, bal 0.93755, kappa 0.94582 (+/- 0.0) |
| 2 | top2_zoo_afno_block | most-different | 0 | 30 | 3 | 53,901 | 8.5 | 27 | 0.003 | 0.001 | 32 | 0.95962 +/- 0.00863 | acc 0.9634, bal 0.93352, kappa 0.95925 (+/- 0.01092) |
| 3 | top3_zoo_mbconv_res2net_block | best-small | 0 | 30 | 3 | 10,651 | 1.4 | 31 | 0.003 | 0.01 | 64 | 0.95321 +/- 0.00926 | acc 0.94574, bal 0.91935, kappa 0.93955 (+/- 0.0042) |

Each number is a mean over `runs` (seeds); +/- is the larger of the std over those runs and the sampling error of the validation set (the only noise estimate when runs = 1). Models are ranked within the same `epochs` (more epochs first). Finalists were ranked on 1 split(s).

## Reference baselines (same split(s))

| model | val | test |
|---|---|---|
| random_forest_spectral_spatial | 0.96731 | acc 0.97243, bal 0.94471, kappa 0.9693 |
| random_forest_spectral | 0.9 | acc 0.90026, bal 0.82892, kappa 0.88879 |
| svm_rbf_spectral | 0.89038 | acc 0.87149, bal 0.7865, kappa 0.85666 |
| svm_rbf_spectral_spatial | 0.66923 | acc 0.67682, bal 0.48085, kappa 0.63236 |

## Verdict

**No significant win over the best reference (random_forest_spectral_spatial).** Val margin -0.00385 vs noise 0.00823; test margin -0.0211 vs noise 0.00333. Consider the reference baseline first; the deep-learning candidates below are still valid to try.

## Objective: accuracy

Pareto front of (validation, parameters) among the ranked finalists:

| family | val | params | FLOPs |
|---|---|---|---|
| zoo_mbconv_res2net_block | 0.95321 | 10,651 | 1,426,358 |
| mambahsi | 0.96346 | 24,145 | 3,320,768 |

## Zero-cost proxies (information only, nothing was pruned)

Spearman rank correlation with the validation score after 4 epochs (rung 1):

| proxy | Spearman | candidates |
|---|---|---|
| naswot | 0.088 | 16 |
| synflow | -0.184 | 16 |
| grad_norm | 0.143 | 18 |
| snip | 0.04 | 18 |
| zico | 0.194 | 18 |
| params | -0.404 | 30 |
| seed_prior | 0.229 | 30 |

Clearly better than the parameter count: none.

## Budget: 10m (seconds)

Wall clock 592.8 s, 662.0 epochs trained, overrun -1.2%.

| stage | seconds | epochs |
|---|---|---|
| load | 10.0 | 0 |
| reference | 2.8 | 0 |
| generate | 21.77 | 0 |
| proxies | 60.07 | 0 |
| seed_selection | 0.01 | 0 |
| search | 228.94 | 290 |
| evolution | 165.11 | 264 |
| equalize | 75.88 | 18 |
| runs | 28.15 | 90 |
| hparams | 0.0 | 0 |
| export | 0.0 | 0 |
