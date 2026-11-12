# top3: resnet  []

- Config: `{'family': 'resnet', 'width': 32, 'depth': 2, 'dropout': 0.0, 'attention': 'se', 'lr': 0.003, 'wd': 0.01, 'batch': 64, 'spec': {'op': 'seq', 'body': [{'op': 'seq', 'body': [{'op': 'conv_bn_act', 'out': 32, 'k': 3, 's': 1, 'act': 'relu', 'norm': 'batch'}, {'op': 'basic_block', 'out': 32, 's': 1, 'act': 'relu', 'norm': 'batch'}, {'op': 'basic_block', 'out': 32, 's': 1, 'act': 'relu', 'norm': 'batch'}, {'op': 'se', 'ratio': 0.25}]}, {'op': 'seq', 'body': [{'op': 'basic_block', 'out': 64, 's': 2, 'act': 'relu', 'norm': 'batch'}, {'op': 'basic_block', 'out': 64, 's': 1, 'act': 'relu', 'norm': 'batch'}, {'op': 'se', 'ratio': 0.25}]}, {'op': 'seq', 'body': [{'op': 'basic_block', 'out': 128, 's': 2, 'act': 'relu', 'norm': 'batch'}, {'op': 'basic_block', 'out': 128, 's': 1, 'act': 'relu', 'norm': 'batch'}, {'op': 'se', 'ratio': 0.25}]}]}, 'input': {'kind': 'map', 'dims': [1, 28, 28]}, 'head': 'cls', 'out_dim': 10, 'flops': 161038848, 'act_mb': 2.561768, 'vram_mb': 339.219488, 'seed_prior': -0.147, 'seed_rank': 4, 'seed_score': 86.5, 'seed_traits': ['channel_attention', 'local_conv']}`
- Origin: legacy family
- Params: 707,074
- Cost model: 161,038,848 forward FLOPs per sample (matrix multiply and convolution only), about 339 MB training memory at batch 64 (estimate, not measured on a GPU)
- Validation (accuracy): 0.6417 +/- 0.0138 (1 runs on 1 split(s), 4 epochs)
- Test (mean over runs): 0.6433 (bal 0.6433, kappa 0.6037), primary metric +/- 0.0000
- Hyperparameters: lr 0.003, weight decay 0.01, batch 64, dropout 0.0, AdamW, cosine, 100 epochs (full recipe), clip 5.0 (searched; not estimated)
- Model file: `model.py` is plain PyTorch (no DLNet import): the network's own source, identical to the searched network
- Search time spent on this candidate: 44.7s
- Indistinguishable-group id: 2 (same id = difference within run-to-run noise)

Short search budget only. Re-train with the full recipe (see `python train.py -h`); neighbouring ranks may reorder.

Use: `python train.py --data ...` writes model.pt and preprocess.json; then `python predict.py --data new_file` predicts in the original label units (`--export model.pt2` saves a deployable graph). Training recipe: modern {'augment': True, 'label_smoothing': 0.1}.

Block diagram: `architecture.svg`. Training notebook (Jupyter / Colab): `../baselines.ipynb`.
