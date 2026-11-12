# top1: tucker_cnn2d  ['top1']

- Config: `{'family': 'tucker_cnn2d', 'patch': 13, 'width': 64, 'depth': 2, 'dropout': 0.4, 'rank_ratio': 0.5, 'lr': 0.001, 'wd': 0.0001, 'batch': 64, 'spec': {'op': 'seq', 'body': [{'op': 'conv_bn_act', 'out': 64, 'k': 3, 's': 1, 'act': 'relu', 'norm': 'batch'}, {'op': 'tucker_conv', 'out': 64, 'ratio': 0.5, 'k': 3}, {'op': 'pool_flatten', 'size': 3}]}, 'input': {'kind': 'map', 'dims': [30, 13, 13]}, 'head': 'cls', 'out_dim': 16, 'flops': 10358528, 'act_mb': 0.309824, 'vram_mb': 40.298752, 'seed_prior': 0.0, 'seed_rank': 13, 'seed_score': 125.0, 'seed_traits': ['local_conv', 'position_aware']}`
- Origin: legacy family
- Params: 40,080
- Cost model: 10,358,528 forward FLOPs per sample (matrix multiply and convolution only), about 40 MB training memory at batch 64 (estimate, not measured on a GPU)
- Validation (accuracy): 0.9847 +/- 0.0038 (3 runs on 1 split(s), 30 epochs)
- Test (mean over runs): 0.9828 (bal 0.9624, kappa 0.9804), primary metric +/- 0.0035
- Hyperparameters: lr 0.001, weight decay 0.0001, batch 64, dropout 0.4, AdamW, cosine, 100 epochs (full recipe), clip 5.0 (searched; not estimated)
- Model file: `model.py` is plain PyTorch (no DLNet import): the network's own source, identical to the searched network
- Search time spent on this candidate: 31.5s
- Indistinguishable-group id: 0 (same id = difference within run-to-run noise)

Short search budget only. Re-train with the full recipe (see `python train.py -h`); neighbouring ranks may reorder.

Use: `python train.py --data ...` writes model.pt and preprocess.json; then `python predict.py --data new_file` predicts in the original label units (`--export model.pt2` saves a deployable graph). Training recipe: modern {'augment': True, 'label_smoothing': 0.1}.

Block diagram: `architecture.svg`. Training notebook (Jupyter / Colab): `../baselines.ipynb`.
