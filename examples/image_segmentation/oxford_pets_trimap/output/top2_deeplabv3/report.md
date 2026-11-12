# top2: deeplabv3  ['best-small', 'most-different']

- Config: `{'family': 'deeplabv3', 'width': 8, 'depth': 1, 'dropout': 0.1, 'encoder': 'densenet', 'attention': 'none', 'lr': 0.003, 'wd': 0.0001, 'batch': 32, 'spec': {'op': 'encdec', 'stages': [{'op': 'seq', 'body': [{'op': 'conv_bn_act', 'out': 8, 'k': 3, 's': 1, 'act': 'relu', 'norm': 'batch'}, {'op': 'conv_bn_act', 'out': 4, 'k': 1, 's': 1, 'act': 'relu', 'norm': 'batch'}, {'op': 'dense_block', 'layers': 1, 'growth': 4, 'bn_size': 2, 'dropout': 0.0}]}, {'op': 'seq', 'body': [{'op': 'conv_bn_act', 'out': 8, 'k': 1, 's': 1, 'act': 'relu', 'norm': 'batch'}, {'op': 'pool', 'mode': 'avg', 'k': 2, 's': 2}, {'op': 'dense_block', 'layers': 1, 'growth': 8, 'bn_size': 2, 'dropout': 0.0}]}, {'op': 'seq', 'body': [{'op': 'conv_bn_act', 'out': 16, 'k': 1, 's': 1, 'act': 'relu', 'norm': 'batch'}, {'op': 'pool', 'mode': 'avg', 'k': 2, 's': 2}, {'op': 'dense_block', 'layers': 1, 'growth': 16, 'bn_size': 2, 'dropout': 0.0}]}], 'dec': 'deeplabv3', 'width': 8}, 'input': {'kind': 'map', 'dims': [3, 96, 96]}, 'head': 'dense', 'out_dim': 3, 'flops': 34357760, 'act_mb': 6.101216, 'vram_mb': 390.733616, 'seed_prior': 0.0, 'seed_rank': 30, 'seed_score': 68.5, 'seed_traits': ['context', 'local_conv', 'multi_scale']}`
- Origin: legacy family
- Params: 15,987
- Cost model: 34,357,760 forward FLOPs per sample (matrix multiply and convolution only), about 391 MB training memory at batch 32 (estimate, not measured on a GPU)
- Validation (miou): 0.5338 +/- 0.0290 (1 runs on 1 split(s), 30 epochs)
- Test (mean over runs): mIoU 0.5355 (pixel acc 0.8022, kappa 0.6294), primary metric +/- 0.0000
- Hyperparameters: lr 0.003, weight decay 0.0001, batch 32, dropout 0.1, AdamW, cosine, 100 epochs (full recipe), clip 5.0 (searched; not estimated)
- Model file: `model.py` is plain PyTorch (no DLNet import): the network's own source, identical to the searched network
- Search time spent on this candidate: 126.8s
- Indistinguishable-group id: 1 (same id = difference within run-to-run noise)

Short search budget only. Re-train with the full recipe (see `python train.py -h`); neighbouring ranks may reorder.

Use: `python train.py --data ...` writes model.pt and preprocess.json; then `python predict.py --data new_file` predicts in the original label units (`--export model.pt2` saves a deployable graph). Training recipe: modern {'augment': True, 'label_smoothing': 0.1}.

Block diagram: `architecture.svg`. Training notebook (Jupyter / Colab): `../baselines.ipynb`.
