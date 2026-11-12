# top1: deeplabv3plus  ['top1']

- Config: `{'family': 'deeplabv3plus', 'width': 8, 'depth': 2, 'dropout': 0.1, 'encoder': 'densenet', 'attention': 'se', 'lr': 0.001, 'wd': 0.0001, 'batch': 32, 'spec': {'op': 'encdec', 'stages': [{'op': 'seq', 'body': [{'op': 'conv_bn_act', 'out': 8, 'k': 3, 's': 1, 'act': 'relu', 'norm': 'batch'}, {'op': 'conv_bn_act', 'out': 4, 'k': 1, 's': 1, 'act': 'relu', 'norm': 'batch'}, {'op': 'dense_block', 'layers': 2, 'growth': 2, 'bn_size': 2, 'dropout': 0.0}, {'op': 'se', 'ratio': 0.25}]}, {'op': 'seq', 'body': [{'op': 'conv_bn_act', 'out': 8, 'k': 1, 's': 1, 'act': 'relu', 'norm': 'batch'}, {'op': 'pool', 'mode': 'avg', 'k': 2, 's': 2}, {'op': 'dense_block', 'layers': 2, 'growth': 4, 'bn_size': 2, 'dropout': 0.0}, {'op': 'se', 'ratio': 0.25}]}, {'op': 'seq', 'body': [{'op': 'conv_bn_act', 'out': 16, 'k': 1, 's': 1, 'act': 'relu', 'norm': 'batch'}, {'op': 'pool', 'mode': 'avg', 'k': 2, 's': 2}, {'op': 'dense_block', 'layers': 2, 'growth': 8, 'bn_size': 2, 'dropout': 0.0}, {'op': 'se', 'ratio': 0.25}]}], 'dec': 'deeplabv3plus', 'width': 8}, 'input': {'kind': 'map', 'dims': [1, 256, 256]}, 'head': 'dense', 'out_dim': 2, 'flops': 180422528, 'act_mb': 65.012512, 'vram_mb': 4160.959968, 'seed_prior': 0.879, 'seed_rank': 1, 'seed_score': 99.0, 'seed_traits': ['channel_attention', 'context', 'fine_detail', 'local_conv', 'multi_scale']}`
- Origin: legacy family
- Params: 9,950
- Cost model: 180,422,528 forward FLOPs per sample (matrix multiply and convolution only), about 4161 MB training memory at batch 32 (estimate, not measured on a GPU)
- Validation (miou): 0.2956 +/- 0.1863 (3 runs on 1 split(s), 4 epochs)
- Test (mean over runs): mIoU 0.2984 (pixel acc 0.5969, kappa 0.0000), primary metric +/- 0.1370
- Hyperparameters: lr 0.001, weight decay 0.0001, batch 32, dropout 0.1, AdamW, cosine, 100 epochs (full recipe), clip 5.0 (searched; not estimated)
- Model file: `model.py` is plain PyTorch (no DLNet import): the network's own source, identical to the searched network
- Search time spent on this candidate: 21.6s
- Indistinguishable-group id: 0 (same id = difference within run-to-run noise)

Short search budget only. Re-train with the full recipe (see `python train.py -h`); neighbouring ranks may reorder.

Use: `python train.py --data ...` writes model.pt and preprocess.json; then `python predict.py --data new_file` predicts in the original label units (`--export model.pt2` saves a deployable graph). Training recipe: modern {'augment': True, 'label_smoothing': 0.1}.
