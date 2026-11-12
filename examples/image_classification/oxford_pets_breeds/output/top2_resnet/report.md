# top2: resnet  ['best-small', 'most-different']

- Config: `{'family': 'resnet', 'width': 8, 'depth': 2, 'dropout': 0.2, 'attention': 'none', 'lr': 0.001, 'wd': 0.01, 'batch': 32, 'spec': {'op': 'seq', 'body': [{'op': 'seq', 'body': [{'op': 'conv_bn_act', 'out': 8, 'k': 3, 's': 1, 'act': 'relu', 'norm': 'batch'}, {'op': 'basic_block', 'out': 8, 's': 1, 'act': 'relu', 'norm': 'batch'}, {'op': 'basic_block', 'out': 8, 's': 1, 'act': 'relu', 'norm': 'batch'}]}, {'op': 'seq', 'body': [{'op': 'basic_block', 'out': 16, 's': 2, 'act': 'relu', 'norm': 'batch'}, {'op': 'basic_block', 'out': 16, 's': 1, 'act': 'relu', 'norm': 'batch'}]}, {'op': 'seq', 'body': [{'op': 'basic_block', 'out': 32, 's': 2, 'act': 'relu', 'norm': 'batch'}, {'op': 'basic_block', 'out': 32, 's': 1, 'act': 'relu', 'norm': 'batch'}]}]}, 'input': {'kind': 'map', 'dims': [3, 64, 64]}, 'head': 'cls', 'out_dim': 37, 'flops': 54200640, 'act_mb': 3.342612, 'vram_mb': 214.651344, 'seed_prior': 0.0, 'seed_rank': 3, 'seed_score': 88.5, 'seed_traits': ['local_conv']}`
- Origin: legacy family
- Params: 45,261
- Cost model: 54,200,640 forward FLOPs per sample (matrix multiply and convolution only), about 215 MB training memory at batch 32 (estimate, not measured on a GPU)
- Validation (accuracy): 0.0568 +/- 0.0110 (3 runs on 1 split(s), 4 epochs)
- Test (mean over runs): 0.0622 (bal 0.0622, kappa 0.0361), primary metric +/- 0.0044
- Hyperparameters: lr 0.001, weight decay 0.01, batch 32, dropout 0.2, AdamW, cosine, 100 epochs (full recipe), clip 5.0 (searched; not estimated)
- Model file: `model.py` is plain PyTorch (no DLNet import): the network's own source, identical to the searched network
- Search time spent on this candidate: 161.7s
- Indistinguishable-group id: 1 (same id = difference within run-to-run noise)

Short search budget only. Re-train with the full recipe (see `python train.py -h`); neighbouring ranks may reorder.

Use: `python train.py --data ...` writes model.pt and preprocess.json; then `python predict.py --data new_file` predicts in the original label units (`--export model.pt2` saves a deployable graph). Training recipe: modern {'augment': True, 'label_smoothing': 0.1}.

Block diagram: `architecture.svg`. Training notebook (Jupyter / Colab): `../baselines.ipynb`.
