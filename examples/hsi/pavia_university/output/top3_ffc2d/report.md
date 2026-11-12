# top3: ffc2d  []

- Config: `{'family': 'ffc2d', 'patch': 9, 'width': 64, 'depth': 1, 'dropout': 0.0, 'lr': 0.001, 'wd': 0.001, 'batch': 32, 'spec': {'op': 'seq', 'body': [{'op': 'conv_bn_act', 'out': 64, 'k': 3, 's': 1, 'act': 'relu', 'norm': 'batch'}, {'op': 'ffc_block', 'ratio': 0.5, 'lfu': 0}, {'op': 'gap', 'mode': 'avg'}]}, 'input': {'kind': 'map', 'dims': [30, 9, 9]}, 'head': 'cls', 'out_dim': 9, 'flops': 12274560, 'act_mb': 0.336036, 'vram_mb': 22.751632, 'seed_prior': 0.0, 'seed_rank': 57, 'seed_score': 139.0, 'seed_traits': ['local_conv']}`
- Origin: legacy family
- Params: 77,833
- Cost model: 12,274,560 forward FLOPs per sample (matrix multiply and convolution only), about 23 MB training memory at batch 32 (estimate, not measured on a GPU)
- Validation (accuracy): 0.9918 +/- 0.0034 (2 runs on 1 split(s), 30 epochs)
- Test (mean over runs): 0.9925 (bal 0.9844, kappa 0.9901), primary metric +/- 0.0008
- Hyperparameters: lr 0.001, weight decay 0.001, batch 32, dropout 0.0, AdamW, cosine, 100 epochs (full recipe), clip 5.0 (searched; not estimated)
- Search time spent on this candidate: 67.1s
- Indistinguishable-group id: 0 (same id = difference within run-to-run noise)

Short search budget only. Re-train with the full recipe (see `python train.py -h`); neighbouring ranks may reorder.

Use: `python train.py --data ...` writes model.pt and preprocess.json; then `python predict.py --data new_file` predicts in the original label units (`--export model.pt2` saves a deployable graph). Training recipe: modern {'augment': True, 'label_smoothing': 0.1}.

Block diagram: `architecture.svg`. Training notebook (Jupyter / Colab): `../baselines.ipynb`.
