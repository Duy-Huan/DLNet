# top1: fcn  ['top1']

- Config: `{'family': 'fcn', 'width': 8, 'depth': 2, 'dropout': 0.1, 'encoder': 'mobilenetv2', 'attention': 'none', 'lr': 0.003, 'wd': 0.01, 'batch': 64, 'spec': {'op': 'encdec', 'stages': [{'op': 'seq', 'body': [{'op': 'conv_bn_act', 'out': 8, 'k': 3, 's': 1, 'act': 'relu', 'norm': 'batch'}, {'op': 'mbconv', 'out': 8, 'expand': 4, 'k': 3, 's': 1, 'se': 0, 'act': 'relu6'}, {'op': 'mbconv', 'out': 8, 'expand': 4, 'k': 3, 's': 1, 'se': 0, 'act': 'relu6'}]}, {'op': 'seq', 'body': [{'op': 'mbconv', 'out': 16, 'expand': 4, 'k': 3, 's': 2, 'se': 0, 'act': 'relu6'}, {'op': 'mbconv', 'out': 16, 'expand': 4, 'k': 3, 's': 1, 'se': 0, 'act': 'relu6'}]}, {'op': 'seq', 'body': [{'op': 'mbconv', 'out': 32, 'expand': 4, 'k': 3, 's': 2, 'se': 0, 'act': 'relu6'}, {'op': 'mbconv', 'out': 32, 'expand': 4, 'k': 3, 's': 1, 'se': 0, 'act': 'relu6'}]}], 'dec': 'fcn', 'width': 8}, 'input': {'kind': 'map', 'dims': [3, 90, 120]}, 'head': 'dense', 'out_dim': 11, 'flops': 89617920, 'act_mb': 35.38176, 'vram_mb': 4529.196592, 'seed_prior': 0.0, 'seed_rank': 1, 'seed_score': 71.5, 'seed_traits': ['fine_detail', 'light', 'local_conv']}`
- Origin: legacy family
- Params: 20,707
- Cost model: 89,617,920 forward FLOPs per sample (matrix multiply and convolution only), about 4529 MB training memory at batch 64 (estimate, not measured on a GPU)
- Validation (miou): 0.0982 +/- 0.0252 (3 runs on 1 split(s), 4 epochs)
- Test (mean over runs): mIoU 0.0956 (pixel acc 0.4420, kappa 0.2305), primary metric +/- 0.0065
- Hyperparameters: lr 0.003, weight decay 0.01, batch 64, dropout 0.1, AdamW, cosine, 100 epochs (full recipe), clip 5.0 (searched; not estimated)
- Model file: `model.py` is plain PyTorch (no DLNet import): the network's own source, identical to the searched network
- Search time spent on this candidate: 192.4s
- Indistinguishable-group id: 0 (same id = difference within run-to-run noise)

Short search budget only. Re-train with the full recipe (see `python train.py -h`); neighbouring ranks may reorder.

Use: `python train.py --data ...` writes model.pt and preprocess.json; then `python predict.py --data new_file` predicts in the original label units (`--export model.pt2` saves a deployable graph). Training recipe: modern {'augment': True, 'label_smoothing': 0.1}.
