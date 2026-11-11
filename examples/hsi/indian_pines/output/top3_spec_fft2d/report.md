# top3: spec_fft2d  []

- Config: `{'family': 'spec_fft2d', 'patch': 9, 'width': 32, 'depth': 3, 'dropout': 0.2, 'lr': 0.003, 'wd': 0.0001, 'batch': 32, 'spec': {'op': 'seq', 'body': [{'op': 'spectral_rfft'}, {'op': 'seq', 'body': [{'op': 'conv_bn_act', 'out': 32, 'k': 3, 's': 1, 'act': 'relu', 'norm': 'batch'}, {'op': 'res', 'body': {'op': 'seq', 'body': [{'op': 'conv_bn_act', 'out': 32, 'k': 3, 's': 1, 'act': 'relu', 'norm': 'batch'}, {'op': 'conv_bn_act', 'out': 32, 'k': 3, 's': 1, 'act': 'identity', 'norm': 'batch'}]}}, {'op': 'res', 'body': {'op': 'seq', 'body': [{'op': 'conv_bn_act', 'out': 32, 'k': 3, 's': 1, 'act': 'relu', 'norm': 'batch'}, {'op': 'conv_bn_act', 'out': 32, 'k': 3, 's': 1, 'act': 'identity', 'norm': 'batch'}]}}, {'op': 'res', 'body': {'op': 'seq', 'body': [{'op': 'conv_bn_act', 'out': 32, 'k': 3, 's': 1, 'act': 'relu', 'norm': 'batch'}, {'op': 'conv_bn_act', 'out': 32, 'k': 3, 's': 1, 'act': 'identity', 'norm': 'batch'}]}}, {'op': 'gap', 'mode': 'avg'}]}]}, 'input': {'kind': 'map', 'dims': [30, 9, 9]}, 'head': 'cls', 'out_dim': 16, 'flops': 10451968, 'act_mb': 0.19744, 'vram_mb': 13.683968, 'seed_prior': 0.0, 'seed_rank': 74, 'seed_score': 134.0, 'seed_traits': ['local_conv']}`
- Origin: legacy family
- Params: 65,488
- Cost model: 10,451,968 forward FLOPs per sample (matrix multiply and convolution only), about 14 MB training memory at batch 32 (estimate, not measured on a GPU)
- Validation (accuracy): 0.9828 +/- 0.0041 (3 runs on 1 split(s), 30 epochs)
- Test (mean over runs): 0.9835 (bal 0.9290, kappa 0.9812), primary metric +/- 0.0021
- Hyperparameters: lr 0.003, weight decay 0.0001, batch 32, dropout 0.2, AdamW, cosine, 100 epochs (full recipe), clip 5.0 (searched; not estimated)
- Search time spent on this candidate: 48.5s
- Indistinguishable-group id: 0 (same id = difference within run-to-run noise)

Short search budget only. Re-train with the full recipe (see `python train.py -h`); neighbouring ranks may reorder.

Use: `python train.py --data ...` writes model.pt and preprocess.json; then `python predict.py --data new_file` predicts in the original label units (`--export model.pt2` saves a deployable graph). Training recipe: modern {'augment': True, 'label_smoothing': 0.1}.
