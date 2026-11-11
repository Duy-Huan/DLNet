# top2: ffc2d  ['best-small', 'most-different']

- Config: `{'family': 'ffc2d', 'patch': 9, 'width': 64, 'depth': 1, 'dropout': 0.0, 'lr': 0.001, 'wd': 0.001, 'batch': 32, 'spec': {'op': 'seq', 'body': [{'op': 'conv_bn_act', 'out': 64, 'k': 3, 's': 1, 'act': 'relu', 'norm': 'batch'}, {'op': 'ffc_block', 'ratio': 0.5, 'lfu': 0}, {'op': 'gap', 'mode': 'avg'}]}, 'input': {'kind': 'map', 'dims': [30, 9, 9]}, 'head': 'cls', 'out_dim': 16, 'flops': 12275456, 'act_mb': 0.336064, 'vram_mb': 22.760704, 'seed_prior': 0.0, 'seed_rank': 64, 'seed_score': 133.5, 'seed_traits': ['local_conv']}`
- Origin: legacy family
- Params: 78,288
- Cost model: 12,275,456 forward FLOPs per sample (matrix multiply and convolution only), about 23 MB training memory at batch 32 (estimate, not measured on a GPU)
- Validation (accuracy): 0.9963 +/- 0.0030 (1 runs on 1 split(s), 30 epochs)
- Test (mean over runs): 0.9926 (bal 0.9884, kappa 0.9918), primary metric +/- 0.0000
- Hyperparameters: lr 0.001, weight decay 0.001, batch 32, dropout 0.0, AdamW, cosine, 100 epochs (full recipe), clip 5.0 (searched; not estimated)
- Search time spent on this candidate: 53.5s
- Indistinguishable-group id: 0 (same id = difference within run-to-run noise)

Short search budget only. Re-train with the full recipe (see `python train.py -h`); neighbouring ranks may reorder.

Use: `python train.py --data ...` writes model.pt and preprocess.json; then `python predict.py --data new_file` predicts in the original label units (`--export model.pt2` saves a deployable graph). Training recipe: modern {'augment': True, 'label_smoothing': 0.1}.
