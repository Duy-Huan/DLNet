# top2: zoo_afno_block  ['most-different']

- Config: `{'family': 'zoo_afno_block', 'spec': {'op': 'seq', 'body': [{'op': 'conv_bn_act', 'out': 64, 'k': 3, 's': 1, 'act': 'relu', 'norm': 'batch'}, {'op': 'afno_block', 'num_blocks': 8, 'mlp_ratio': 4, 'double_skip': 0, 'dropout': 0.0}, {'op': 'gap', 'mode': 'avg'}]}, 'input': {'kind': 'map', 'dims': [30, 9, 9]}, 'out_dim': 13, 'head': 'cls', 'dropout': 0.0, 'origin': 'widen<-afno2d', 'lr': 0.003, 'wd': 0.001, 'batch': 32, 'flops': 8478080, 'act_mb': 0.41554, 'vram_mb': 27.456976}`
- Origin: widen<-afno2d
- Params: 53,901
- Cost model: 8,478,080 forward FLOPs per sample (matrix multiply and convolution only), about 27 MB training memory at batch 32 (estimate, not measured on a GPU)
- Validation (accuracy): 0.9596 +/- 0.0086 (3 runs on 1 split(s), 30 epochs)
- Test (mean over runs): 0.9634 (bal 0.9335, kappa 0.9592), primary metric +/- 0.0109
- Hyperparameters: lr 0.003, weight decay 0.001, batch 32, dropout 0.0, AdamW, cosine, 100 epochs (full recipe), clip 5.0 (searched; not estimated)
- Search time spent on this candidate: 35.2s
- Indistinguishable-group id: 0 (same id = difference within run-to-run noise)

Short search budget only. Re-train with the full recipe (see `python train.py -h`); neighbouring ranks may reorder.

Use: `python train.py --data ...` writes model.pt and preprocess.json; then `python predict.py --data new_file` predicts in the original label units (`--export model.pt2` saves a deployable graph). Training recipe: modern {'augment': True, 'label_smoothing': 0.1}.

Block diagram: `architecture.svg`. Training notebook (Jupyter / Colab): `../baselines.ipynb`.
