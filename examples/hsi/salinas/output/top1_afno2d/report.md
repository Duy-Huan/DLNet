# top1: afno2d  ['top1']

- Config: `{'family': 'afno2d', 'patch': 9, 'width': 64, 'depth': 2, 'dropout': 0.0, 'lr': 0.001, 'wd': 0.0001, 'batch': 64, 'spec': {'op': 'seq', 'body': [{'op': 'conv_bn_act', 'out': 64, 'k': 3, 's': 1, 'act': 'relu', 'norm': 'batch'}, {'op': 'afno_block', 'num_blocks': 8, 'mlp_ratio': 4, 'double_skip': 0, 'dropout': 0.0}, {'op': 'afno_block', 'num_blocks': 8, 'mlp_ratio': 4, 'double_skip': 0, 'dropout': 0.0}, {'op': 'gap', 'mode': 'avg'}]}, 'input': {'kind': 'map', 'dims': [30, 9, 9]}, 'head': 'cls', 'out_dim': 16, 'flops': 14155520, 'act_mb': 0.768064, 'vram_mb': 99.748096, 'seed_prior': 0.0, 'seed_rank': 57, 'seed_score': 133.5, 'seed_traits': ['local_conv']}`
- Origin: legacy family
- Params: 89,744
- Cost model: 14,155,520 forward FLOPs per sample (matrix multiply and convolution only), about 100 MB training memory at batch 64 (estimate, not measured on a GPU)
- Validation (accuracy): 0.9972 +/- 0.0030 (1 runs on 1 split(s), 30 epochs)
- Test (mean over runs): 0.9961 (bal 0.9971, kappa 0.9957), primary metric +/- 0.0000
- Hyperparameters: lr 0.001, weight decay 0.0001, batch 64, dropout 0.0, AdamW, cosine, 100 epochs (full recipe), clip 5.0 (searched; not estimated)
- Search time spent on this candidate: 78.3s
- Indistinguishable-group id: 0 (same id = difference within run-to-run noise)

Short search budget only. Re-train with the full recipe (see `python train.py -h`); neighbouring ranks may reorder.

Use: `python train.py --data ...` writes model.pt and preprocess.json; then `python predict.py --data new_file` predicts in the original label units (`--export model.pt2` saves a deployable graph). Training recipe: modern {'augment': True, 'label_smoothing': 0.1}.
