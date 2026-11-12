# top3: zoo_conv_bn_act_gap  ['best-small', 'most-stable']

- Config: `{'family': 'zoo_conv_bn_act_gap', 'spec': {'op': 'seq', 'body': [{'op': 'conv_bn_act', 'out': 32, 'k': 3, 's': 1, 'act': 'relu', 'norm': 'batch'}, {'op': 'res', 'body': {'op': 'seq', 'body': [{'op': 'res', 'body': {'op': 'conv_bn_act', 'out': 32, 'k': 3, 's': 1, 'act': 'relu', 'norm': 'batch'}, 'drop_path': 0.0}, {'op': 'conv_bn_act', 'out': 32, 'k': 3, 's': 1, 'act': 'identity', 'norm': 'batch'}]}}, {'op': 'gap', 'mode': 'avg'}]}, 'input': {'kind': 'map', 'dims': [30, 13, 13]}, 'out_dim': 16, 'head': 'cls', 'dropout': 0.0, 'origin': 'wrap_res<-resnet2d', 'lr': 0.003, 'wd': 0.001, 'batch': 64, 'flops': 9151360, 'act_mb': 0.173504, 'vram_mb': 22.653184}`
- Origin: wrap_res<-resnet2d
- Params: 27,792
- Cost model: 9,151,360 forward FLOPs per sample (matrix multiply and convolution only), about 23 MB training memory at batch 64 (estimate, not measured on a GPU)
- Validation (accuracy): 0.9815 +/- 0.0042 (2 runs on 1 split(s), 30 epochs)
- Test (mean over runs): 0.9757 (bal 0.8570, kappa 0.9723), primary metric +/- 0.0015
- Hyperparameters: lr 0.003, weight decay 0.001, batch 64, dropout 0.0, AdamW, cosine, 100 epochs (full recipe), clip 5.0 (searched; not estimated)
- Model file: `model.py` is plain PyTorch (no DLNet import): the network's own source, identical to the searched network
- Search time spent on this candidate: 14.9s
- Indistinguishable-group id: 0 (same id = difference within run-to-run noise)

Short search budget only. Re-train with the full recipe (see `python train.py -h`); neighbouring ranks may reorder.

Use: `python train.py --data ...` writes model.pt and preprocess.json; then `python predict.py --data new_file` predicts in the original label units (`--export model.pt2` saves a deployable graph). Training recipe: modern {'augment': True, 'label_smoothing': 0.1}.

Block diagram: `architecture.svg`. Training notebook (Jupyter / Colab): `../baselines.ipynb`.
