# top1: mambahsi  ['top1']

- Config: `{'family': 'mambahsi', 'patch': 9, 'width': 32, 'depth': 2, 'dropout': 0.0, 'lr': 0.003, 'wd': 0.0001, 'batch': 64, 'spec': {'op': 'seq', 'body': [{'op': 'conv_bn_act', 'out': 32, 'k': 1, 's': 1, 'act': 'swish', 'norm': 'group'}, {'op': 'mambahsi_block', 'token_num': 4, 'group_num': 4}, {'op': 'pool', 'mode': 'avg', 'k': 2, 's': 2}, {'op': 'mambahsi_block', 'token_num': 4, 'group_num': 4}, {'op': 'gap', 'mode': 'avg'}]}, 'input': {'kind': 'map', 'dims': [30, 9, 9]}, 'head': 'cls', 'out_dim': 13, 'flops': 3320768, 'act_mb': 0.425508, 'vram_mb': 54.851344, 'seed_prior': 0.0, 'seed_rank': 53, 'seed_score': 154.0, 'seed_traits': ['local_conv']}`
- Origin: legacy family
- Params: 24,145
- Cost model: 3,320,768 forward FLOPs per sample (matrix multiply and convolution only), about 55 MB training memory at batch 64 (estimate, not measured on a GPU)
- Validation (accuracy): 0.9635 +/- 0.0082 (1 runs on 1 split(s), 30 epochs)
- Test (mean over runs): 0.9513 (bal 0.9375, kappa 0.9458), primary metric +/- 0.0000
- Hyperparameters: lr 0.003, weight decay 0.0001, batch 64, dropout 0.0, AdamW, cosine, 100 epochs (full recipe), clip 5.0 (searched; not estimated)
- Search time spent on this candidate: 87.7s
- Indistinguishable-group id: 0 (same id = difference within run-to-run noise)

Short search budget only. Re-train with the full recipe (see `python train.py -h`); neighbouring ranks may reorder.

Use: `python train.py --data ...` writes model.pt and preprocess.json; then `python predict.py --data new_file` predicts in the original label units (`--export model.pt2` saves a deployable graph). Training recipe: modern {'augment': True, 'label_smoothing': 0.1}.

Block diagram: `architecture.svg`. Training notebook (Jupyter / Colab): `../baselines.ipynb`.
