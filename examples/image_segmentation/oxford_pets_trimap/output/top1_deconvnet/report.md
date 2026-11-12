# top1: deconvnet  ['top1']

- Config: `{'family': 'deconvnet', 'width': 8, 'depth': 2, 'dropout': 0.1, 'encoder': 'vgg', 'attention': 'se', 'lr': 0.003, 'wd': 0.0001, 'batch': 32, 'spec': {'op': 'encdec', 'stages': [{'op': 'seq', 'body': [{'op': 'conv_bn_act', 'out': 8, 'k': 3, 's': 1, 'act': 'relu', 'norm': 'batch'}, {'op': 'conv_bn_act', 'out': 8, 'k': 3, 's': 1, 'act': 'relu', 'norm': 'batch'}, {'op': 'conv_bn_act', 'out': 8, 'k': 3, 's': 1, 'act': 'relu', 'norm': 'batch'}, {'op': 'se', 'ratio': 0.25}]}, {'op': 'seq', 'body': [{'op': 'pool', 'mode': 'max', 'k': 2, 's': 2}, {'op': 'conv_bn_act', 'out': 16, 'k': 3, 's': 1, 'act': 'relu', 'norm': 'batch'}, {'op': 'conv_bn_act', 'out': 16, 'k': 3, 's': 1, 'act': 'relu', 'norm': 'batch'}, {'op': 'se', 'ratio': 0.25}]}, {'op': 'seq', 'body': [{'op': 'pool', 'mode': 'max', 'k': 2, 's': 2}, {'op': 'conv_bn_act', 'out': 32, 'k': 3, 's': 1, 'act': 'relu', 'norm': 'batch'}, {'op': 'conv_bn_act', 'out': 32, 'k': 3, 's': 1, 'act': 'relu', 'norm': 'batch'}, {'op': 'se', 'ratio': 0.25}]}], 'dec': 'deconv', 'width': 8}, 'input': {'kind': 'map', 'dims': [3, 96, 96]}, 'head': 'dense', 'out_dim': 3, 'flops': 62227840, 'act_mb': 5.825088, 'vram_mb': 373.1628, 'seed_prior': 0.0, 'seed_rank': 15, 'seed_score': 68.5, 'seed_traits': ['channel_attention', 'local_conv']}`
- Origin: legacy family
- Params: 22,323
- Cost model: 62,227,840 forward FLOPs per sample (matrix multiply and convolution only), about 373 MB training memory at batch 32 (estimate, not measured on a GPU)
- Validation (miou): 0.5785 +/- 0.0287 (2 runs on 1 split(s), 30 epochs)
- Test (mean over runs): mIoU 0.5769 (pixel acc 0.8145, kappa 0.6577), primary metric +/- 0.0061
- Hyperparameters: lr 0.003, weight decay 0.0001, batch 32, dropout 0.1, AdamW, cosine, 100 epochs (full recipe), clip 5.0 (searched; not estimated)
- Model file: `model.py` is plain PyTorch (no DLNet import): the network's own source, identical to the searched network
- Search time spent on this candidate: 212.2s
- Indistinguishable-group id: 0 (same id = difference within run-to-run noise)

Short search budget only. Re-train with the full recipe (see `python train.py -h`); neighbouring ranks may reorder.

Use: `python train.py --data ...` writes model.pt and preprocess.json; then `python predict.py --data new_file` predicts in the original label units (`--export model.pt2` saves a deployable graph). Training recipe: modern {'augment': True, 'label_smoothing': 0.1}.

Block diagram: `architecture.svg`. Training notebook (Jupyter / Colab): `../baselines.ipynb`.
