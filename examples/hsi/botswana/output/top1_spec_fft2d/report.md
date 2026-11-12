# top1: spec_fft2d  ['top1', 'most-stable']

- Config: `{'family': 'spec_fft2d', 'patch': 9, 'width': 32, 'depth': 1, 'dropout': 0.4, 'lr': 0.001, 'wd': 0.0001, 'batch': 32, 'spec': {'op': 'seq', 'body': [{'op': 'spectral_rfft'}, {'op': 'seq', 'body': [{'op': 'conv_bn_act', 'out': 32, 'k': 3, 's': 1, 'act': 'relu', 'norm': 'batch'}, {'op': 'res', 'body': {'op': 'seq', 'body': [{'op': 'conv_bn_act', 'out': 32, 'k': 3, 's': 1, 'act': 'relu', 'norm': 'batch'}, {'op': 'conv_bn_act', 'out': 32, 'k': 3, 's': 1, 'act': 'identity', 'norm': 'batch'}]}}, {'op': 'gap', 'mode': 'avg'}]}]}, 'input': {'kind': 'map', 'dims': [30, 9, 9]}, 'head': 'cls', 'out_dim': 14, 'flops': 4479872, 'act_mb': 0.093752, 'vram_mb': 6.45296, 'seed_prior': 0.0, 'seed_rank': 35, 'seed_score': 155.0, 'seed_traits': ['local_conv']}`
- Origin: legacy family
- Params: 28,302
- Cost model: 4,479,872 forward FLOPs per sample (matrix multiply and convolution only), about 6 MB training memory at batch 32 (estimate, not measured on a GPU)
- Validation (accuracy): 0.9990 +/- 0.0055 (3 runs on 1 split(s), 30 epochs)
- Test (mean over runs): 0.9946 (bal 0.9937, kappa 0.9942), primary metric +/- 0.0008
- Hyperparameters: lr 0.001, weight decay 0.0001, batch 32, dropout 0.4, AdamW, cosine, 100 epochs (full recipe), clip 5.0
  - Adopted: searched. Estimated lr 0.0097, batch 64, wd 0.00014, B_simple 1063; validation 0.9990 (searched) vs 0.9764 (estimated)
- Model file: `model.py` is plain PyTorch (no DLNet import): the network's own source, identical to the searched network
- Search time spent on this candidate: 4.8s
- Indistinguishable-group id: 0 (same id = difference within run-to-run noise)

Short search budget only. Re-train with the full recipe (see `python train.py -h`); neighbouring ranks may reorder.

Use: `python train.py --data ...` writes model.pt and preprocess.json; then `python predict.py --data new_file` predicts in the original label units (`--export model.pt2` saves a deployable graph). Training recipe: modern {'augment': True, 'label_smoothing': 0.1}.

Block diagram: `architecture.svg`. Training notebook (Jupyter / Colab): `../baselines.ipynb`.
