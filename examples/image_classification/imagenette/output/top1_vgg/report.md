# top1: vgg  ['top1']

- Config: `{'family': 'vgg', 'width': 16, 'depth': 1, 'dropout': 0.0, 'attention': 'none', 'lr': 0.001, 'wd': 0.0001, 'batch': 32, 'spec': {'op': 'seq', 'body': [{'op': 'seq', 'body': [{'op': 'conv_bn_act', 'out': 16, 'k': 3, 's': 1, 'act': 'relu', 'norm': 'batch'}, {'op': 'conv_bn_act', 'out': 16, 'k': 3, 's': 1, 'act': 'relu', 'norm': 'batch'}]}, {'op': 'seq', 'body': [{'op': 'pool', 'mode': 'max', 'k': 2, 's': 2}, {'op': 'conv_bn_act', 'out': 32, 'k': 3, 's': 1, 'act': 'relu', 'norm': 'batch'}]}, {'op': 'seq', 'body': [{'op': 'pool', 'mode': 'max', 'k': 2, 's': 2}, {'op': 'conv_bn_act', 'out': 64, 'k': 3, 's': 1, 'act': 'relu', 'norm': 'batch'}]}]}, 'input': {'kind': 'map', 'dims': [3, 64, 64]}, 'head': 'cls', 'out_dim': 10, 'flops': 41288960, 'act_mb': 2.261288, 'vram_mb': 145.149344, 'seed_prior': 1.0, 'seed_rank': 11, 'seed_score': 81.0, 'seed_traits': ['local_conv']}`
- Origin: legacy family
- Params: 26,682
- Cost model: 41,288,960 forward FLOPs per sample (matrix multiply and convolution only), about 145 MB training memory at batch 32 (estimate, not measured on a GPU)
- Validation (accuracy): 0.6050 +/- 0.0200 (1 runs on 1 split(s), 30 epochs)
- Test (mean over runs): 0.5950 (bal 0.5950, kappa 0.5500), primary metric +/- 0.0000
- Hyperparameters: lr 0.001, weight decay 0.0001, batch 32, dropout 0.0, AdamW, cosine, 100 epochs (full recipe), clip 5.0 (searched; not estimated)
- Model file: `model.py` is plain PyTorch (no DLNet import): the network's own source, identical to the searched network
- Search time spent on this candidate: 87.6s
- Indistinguishable-group id: 0 (same id = difference within run-to-run noise)

Short search budget only. Re-train with the full recipe (see `python train.py -h`); neighbouring ranks may reorder.

Use: `python train.py --data ...` writes model.pt and preprocess.json; then `python predict.py --data new_file` predicts in the original label units (`--export model.pt2` saves a deployable graph). Training recipe: modern {'augment': True, 'label_smoothing': 0.1}.

Block diagram: `architecture.svg`. Training notebook (Jupyter / Colab): `../baselines.ipynb`.
