# top3: cnn2d  []

- Config: `{'family': 'cnn2d', 'patch': 9, 'width': 64, 'depth': 3, 'dropout': 0.4, 'lr': 0.003, 'wd': 0.0001, 'batch': 32, 'spec': {'op': 'seq', 'body': [{'op': 'conv_bn_act', 'out': 64, 'k': 3, 's': 1, 'act': 'relu', 'norm': 'batch'}, {'op': 'conv_bn_act', 'out': 64, 'k': 3, 's': 1, 'act': 'relu', 'norm': 'batch'}, {'op': 'conv_bn_act', 'out': 64, 'k': 3, 's': 1, 'act': 'relu', 'norm': 'batch'}, {'op': 'pool_flatten', 'size': 3}]}, 'input': {'kind': 'map', 'dims': [30, 9, 9]}, 'head': 'cls', 'out_dim': 16, 'flops': 14761728, 'act_mb': 0.1936, 'vram_mb': 14.000384, 'seed_prior': 0.0, 'seed_rank': 8, 'seed_score': 133.5, 'seed_traits': ['local_conv', 'position_aware']}`
- Origin: legacy family
- Params: 100,624
- Cost model: 14,761,728 forward FLOPs per sample (matrix multiply and convolution only), about 14 MB training memory at batch 32 (estimate, not measured on a GPU)
- Validation (accuracy): 0.9951 +/- 0.0030 (3 runs on 1 split(s), 30 epochs)
- Test (mean over runs): 0.9926 (bal 0.9957, kappa 0.9918), primary metric +/- 0.0022
- Hyperparameters: lr 0.003, weight decay 0.0001, batch 32, dropout 0.4, AdamW, cosine, 100 epochs (full recipe), clip 5.0 (searched; not estimated)
- Search time spent on this candidate: 63.8s
- Indistinguishable-group id: 0 (same id = difference within run-to-run noise)

Short search budget only. Re-train with the full recipe (see `python train.py -h`); neighbouring ranks may reorder.

Use: `python train.py --data ...` writes model.pt and preprocess.json; then `python predict.py --data new_file` predicts in the original label units (`--export model.pt2` saves a deployable graph). Training recipe: modern {'augment': True, 'label_smoothing': 0.1}.
