# top1: cnn  ['top1', 'most-stable']

- Config: `{'family': 'cnn', 'width': 16, 'depth': 3, 'dropout': 0.2, 'lr': 0.001, 'wd': 0.01, 'batch': 64, 'spec': {'op': 'seq', 'body': [{'op': 'conv_bn_act', 'out': 16, 'k': 3, 's': 1, 'act': 'relu', 'norm': 'batch'}, {'op': 'conv_bn_act', 'out': 16, 'k': 3, 's': 1, 'act': 'relu', 'norm': 'batch'}, {'op': 'pool', 'mode': 'max', 'k': 2, 's': 2}, {'op': 'conv_bn_act', 'out': 32, 'k': 3, 's': 1, 'act': 'relu', 'norm': 'batch'}, {'op': 'conv_bn_act', 'out': 32, 'k': 3, 's': 1, 'act': 'relu', 'norm': 'batch'}, {'op': 'pool', 'mode': 'max', 'k': 2, 's': 2}, {'op': 'conv_bn_act', 'out': 64, 'k': 3, 's': 1, 'act': 'relu', 'norm': 'batch'}, {'op': 'conv_bn_act', 'out': 64, 'k': 3, 's': 1, 'act': 'relu', 'norm': 'batch'}]}, 'input': {'kind': 'map', 'dims': [3, 32, 32]}, 'head': 'cls', 'out_dim': 10, 'flops': 19760384, 'act_mb': 0.713, 'vram_mb': 92.431264, 'seed_prior': 0.0, 'seed_rank': 8, 'seed_score': 83.5, 'seed_traits': ['local_conv']}`
- Origin: legacy family
- Params: 72,954
- Cost model: 19,760,384 forward FLOPs per sample (matrix multiply and convolution only), about 92 MB training memory at batch 64 (estimate, not measured on a GPU)
- Validation (accuracy): 0.6164 +/- 0.0140 (3 runs on 1 split(s), 30 epochs)
- Test (mean over runs): 0.6233 (bal 0.6233, kappa 0.5815), primary metric +/- 0.0102
- Hyperparameters: lr 0.001, weight decay 0.01, batch 64, dropout 0.2, AdamW, cosine, 100 epochs (full recipe), clip 5.0 (searched; not estimated)
- Model file: `model.py` is plain PyTorch (no DLNet import): the network's own source, identical to the searched network
- Search time spent on this candidate: 204.0s
- Indistinguishable-group id: 0 (same id = difference within run-to-run noise)

Short search budget only. Re-train with the full recipe (see `python train.py -h`); neighbouring ranks may reorder.

Use: `python train.py --data ...` writes model.pt and preprocess.json; then `python predict.py --data new_file` predicts in the original label units (`--export model.pt2` saves a deployable graph). Training recipe: modern {'augment': True, 'label_smoothing': 0.1}.
