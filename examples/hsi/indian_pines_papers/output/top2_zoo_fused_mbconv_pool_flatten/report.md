# top2: zoo_fused_mbconv_pool_flatten  ['most-different']

- Config: `{'family': 'zoo_fused_mbconv_pool_flatten', 'spec': {'op': 'seq', 'body': [{'op': 'fused_mbconv', 'out': 64, 'expand': 2, 'k': 3, 's': 1, 'se': 0, 'act': 'relu'}, {'op': 'conv_bn_act', 'out': 64, 'k': 3, 's': 1, 'act': 'relu', 'norm': 'batch'}, {'op': 'conv_bn_act', 'out': 64, 'k': 3, 's': 1, 'act': 'relu', 'norm': 'batch'}, {'op': 'pool_flatten', 'size': 3}]}, 'input': {'kind': 'map', 'dims': [30, 13, 13]}, 'out_dim': 16, 'head': 'cls', 'dropout': 0.0, 'origin': 'swap<-cnn2d', 'lr': 0.003, 'wd': 0.0001, 'batch': 32, 'flops': 31712016, 'act_mb': 0.474768, 'vram_mb': 32.041216}`
- Origin: swap<-cnn2d
- Params: 103,504
- Cost model: 31,712,016 forward FLOPs per sample (matrix multiply and convolution only), about 32 MB training memory at batch 32 (estimate, not measured on a GPU)
- Validation (accuracy): 0.9834 +/- 0.0040 (2 runs on 1 split(s), 30 epochs)
- Test (mean over runs): 0.9840 (bal 0.9709, kappa 0.9817), primary metric +/- 0.0005
- Hyperparameters: lr 0.003, weight decay 0.0001, batch 32, dropout 0.0, AdamW, cosine, 100 epochs (full recipe), clip 5.0 (searched; not estimated)
- Model file: `model.py` is plain PyTorch (no DLNet import): the network's own source, identical to the searched network
- Search time spent on this candidate: 41.8s
- Indistinguishable-group id: 0 (same id = difference within run-to-run noise)

Short search budget only. Re-train with the full recipe (see `python train.py -h`); neighbouring ranks may reorder.

Use: `python train.py --data ...` writes model.pt and preprocess.json; then `python predict.py --data new_file` predicts in the original label units (`--export model.pt2` saves a deployable graph). Training recipe: modern {'augment': True, 'label_smoothing': 0.1}.

Block diagram: `architecture.svg`. Training notebook (Jupyter / Colab): `../baselines.ipynb`.
