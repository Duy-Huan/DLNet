# top3: resnet  []

- Config: `{'family': 'resnet', 'width': 8, 'depth': 2, 'dropout': 0.2, 'attention': 'none', 'lr': 0.001, 'wd': 0.01, 'batch': 32, 'spec': {'op': 'seq', 'body': [{'op': 'seq', 'body': [{'op': 'conv_bn_act', 'out': 8, 'k': 3, 's': 1, 'act': 'relu', 'norm': 'batch'}, {'op': 'basic_block', 'out': 8, 's': 1, 'act': 'relu', 'norm': 'batch'}, {'op': 'basic_block', 'out': 8, 's': 1, 'act': 'relu', 'norm': 'batch'}]}, {'op': 'seq', 'body': [{'op': 'basic_block', 'out': 16, 's': 2, 'act': 'relu', 'norm': 'batch'}, {'op': 'basic_block', 'out': 16, 's': 1, 'act': 'relu', 'norm': 'batch'}]}, {'op': 'seq', 'body': [{'op': 'basic_block', 'out': 32, 's': 2, 'act': 'relu', 'norm': 'batch'}, {'op': 'basic_block', 'out': 32, 's': 1, 'act': 'relu', 'norm': 'batch'}]}]}, 'input': {'kind': 'map', 'dims': [3, 64, 64]}, 'head': 'cls', 'out_dim': 10, 'flops': 54198912, 'act_mb': 3.342504, 'vram_mb': 214.630176, 'seed_prior': 1.0, 'seed_rank': 2, 'seed_score': 89.5, 'seed_traits': ['local_conv']}`
- Origin: legacy family
- Params: 44,370
- Cost model: 54,198,912 forward FLOPs per sample (matrix multiply and convolution only), about 215 MB training memory at batch 32 (estimate, not measured on a GPU)
- Validation (accuracy): 0.4933 +/- 0.0204 (1 runs on 1 split(s), 30 epochs)
- Test (mean over runs): 0.4800 (bal 0.4800, kappa 0.4222), primary metric +/- 0.0000
- Hyperparameters: lr 0.001, weight decay 0.01, batch 32, dropout 0.2, AdamW, cosine, 100 epochs (full recipe), clip 5.0 (searched; not estimated)
- Model file: `model.py` is plain PyTorch (no DLNet import): the network's own source, identical to the searched network
- Search time spent on this candidate: 164.6s
- Indistinguishable-group id: 2 (same id = difference within run-to-run noise)

Short search budget only. Re-train with the full recipe (see `python train.py -h`); neighbouring ranks may reorder.

Use: `python train.py --data ...` writes model.pt and preprocess.json; then `python predict.py --data new_file` predicts in the original label units (`--export model.pt2` saves a deployable graph). Training recipe: modern {'augment': True, 'label_smoothing': 0.1}.
