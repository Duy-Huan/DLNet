# top3: zoo_fca_nin_block  []

- Config: `{'family': 'zoo_fca_nin_block', 'spec': {'op': 'seq', 'body': [{'op': 'conv_bn_act', 'out': 32, 'k': 3, 's': 1, 'act': 'relu', 'norm': 'batch'}, {'op': 'seq', 'body': [{'op': 'res', 'body': {'op': 'seq', 'body': [{'op': 'nin_block', 'out': 32, 'k': 3, 'dropout': 0.3}, {'op': 'conv_bn_act', 'out': 32, 'k': 3, 's': 1, 'act': 'identity', 'norm': 'batch'}, {'op': 'fca', 'reduction': 16, 'method': 'top16'}]}}, {'op': 'act', 'act': 'relu'}]}, {'op': 'seq', 'body': [{'op': 'res', 'body': {'op': 'seq', 'body': [{'op': 'conv_bn_act', 'out': 32, 'k': 3, 's': 1, 'act': 'relu', 'norm': 'batch'}, {'op': 'conv_bn_act', 'out': 32, 'k': 3, 's': 1, 'act': 'identity', 'norm': 'batch'}, {'op': 'fca', 'reduction': 16, 'method': 'top16'}]}}, {'op': 'act', 'act': 'relu'}]}, {'op': 'gap', 'mode': 'avg'}]}, 'input': {'kind': 'map', 'dims': [30, 9, 9]}, 'out_dim': 14, 'head': 'cls', 'dropout': 0.0, 'origin': 'swap<-fcanet2d', 'lr': 0.003, 'wd': 0.0001, 'batch': 32, 'flops': 7704832, 'act_mb': 0.197976, 'vram_mb': 13.448416}`
- Origin: swap<-fcanet2d
- Params: 48,622
- Cost model: 7,704,832 forward FLOPs per sample (matrix multiply and convolution only), about 13 MB training memory at batch 32 (estimate, not measured on a GPU)
- Validation (accuracy): 0.9969 +/- 0.0055 (3 runs on 1 split(s), 30 epochs)
- Test (mean over runs): 0.9937 (bal 0.9938, kappa 0.9932), primary metric +/- 0.0019
- Hyperparameters: lr 0.003, weight decay 0.0001, batch 32, dropout 0.0, AdamW, cosine, 100 epochs (full recipe), clip 5.0
  - Adopted: searched. Estimated lr 0.011, batch 64, wd 0.00014, B_simple 1270; validation 0.9969 (searched) vs 0.9887 (estimated)
- Model file: `model.py` is plain PyTorch (no DLNet import): the network's own source, identical to the searched network
- Search time spent on this candidate: 12.6s
- Indistinguishable-group id: 0 (same id = difference within run-to-run noise)

Short search budget only. Re-train with the full recipe (see `python train.py -h`); neighbouring ranks may reorder.

Use: `python train.py --data ...` writes model.pt and preprocess.json; then `python predict.py --data new_file` predicts in the original label units (`--export model.pt2` saves a deployable graph). Training recipe: modern {'augment': True, 'label_smoothing': 0.1}.

Block diagram: `architecture.svg`. Training notebook (Jupyter / Colab): `../baselines.ipynb`.
