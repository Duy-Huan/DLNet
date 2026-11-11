# top3: zoo_mbconv_res2net_block  ['best-small']

- Config: `{'family': 'zoo_mbconv_res2net_block', 'spec': {'op': 'seq', 'body': [{'op': 'conv_bn_act', 'out': 16, 'k': 3, 's': 1, 'groups': 1, 'dilation': 2, 'act': 'identity', 'norm': 'frn', 'pad': 'same'}, {'op': 'seq', 'body': [{'op': 'mbconv', 'out': 16, 'expand': 6, 'k': 3, 's': 1, 'se': 0, 'act': 'hswish'}]}, {'op': 'seq', 'body': [{'op': 'res2net_block', 'out': 32, 's': 2, 'scale': 4, 'base_width': 26}]}]}, 'input': {'kind': 'map', 'dims': [30, 9, 9]}, 'out_dim': 13, 'head': 'cls', 'dropout': 0.0, 'origin': 'random', 'lr': 0.003, 'wd': 0.01, 'batch': 64, 'flops': 1426358, 'act_mb': 0.238204, 'vram_mb': 30.660528, 'seed_prior': 0.0, 'seed_rank': 66, 'seed_score': 154.0, 'seed_traits': ['light', 'local_conv', 'multi_scale']}`
- Origin: random
- Params: 10,651
- Cost model: 1,426,358 forward FLOPs per sample (matrix multiply and convolution only), about 31 MB training memory at batch 64 (estimate, not measured on a GPU)
- Validation (accuracy): 0.9532 +/- 0.0093 (3 runs on 1 split(s), 30 epochs)
- Test (mean over runs): 0.9457 (bal 0.9194, kappa 0.9395), primary metric +/- 0.0042
- Hyperparameters: lr 0.003, weight decay 0.01, batch 64, dropout 0.0, AdamW, cosine, 100 epochs (full recipe), clip 5.0 (searched; not estimated)
- Search time spent on this candidate: 19.1s
- Indistinguishable-group id: 0 (same id = difference within run-to-run noise)

Short search budget only. Re-train with the full recipe (see `python train.py -h`); neighbouring ranks may reorder.

Use: `python train.py --data ...` writes model.pt and preprocess.json; then `python predict.py --data new_file` predicts in the original label units (`--export model.pt2` saves a deployable graph). Training recipe: modern {'augment': True, 'label_smoothing': 0.1}.
