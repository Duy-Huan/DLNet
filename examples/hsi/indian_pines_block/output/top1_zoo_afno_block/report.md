# top1: zoo_afno_block  ['top1']

- Config: `{'family': 'zoo_afno_block', 'spec': {'op': 'seq', 'body': [{'op': 'conv_bn_act', 'out': 32, 'k': 3, 's': 1, 'act': 'relu', 'norm': 'batch'}, {'op': 'afno_block', 'num_blocks': 8, 'mlp_ratio': 4, 'double_skip': 0, 'dropout': 0.0}, {'op': 'afno_block', 'num_blocks': 4, 'mlp_ratio': 4, 'double_skip': 0, 'dropout': 0.0}, {'op': 'afno_block', 'num_blocks': 8, 'mlp_ratio': 4, 'double_skip': 0, 'dropout': 0.0}, {'op': 'gap', 'mode': 'avg'}]}, 'input': {'kind': 'map', 'dims': [30, 9, 9]}, 'out_dim': 16, 'head': 'cls', 'dropout': 0.0, 'origin': 'param<-afno2d', 'lr': 0.003, 'wd': 0.001, 'batch': 64, 'flops': 5750656, 'act_mb': 0.56032, 'vram_mb': 72.314624}`
- Origin: param<-afno2d
- Params: 37,104
- Cost model: 5,750,656 forward FLOPs per sample (matrix multiply and convolution only), about 72 MB training memory at batch 64 (estimate, not measured on a GPU)
- Validation (accuracy): 0.4668 +/- 0.0430 (2 runs on 2 split(s), 30 epochs)
- Test (mean over runs): 0.5131 (bal 0.4721, kappa 0.4329), primary metric +/- 0.0147
- Hyperparameters: lr 0.003, weight decay 0.001, batch 64, dropout 0.0, AdamW, cosine, 100 epochs (full recipe), clip 5.0 (searched; not estimated)
- Search time spent on this candidate: 56.5s
- Indistinguishable-group id: 0 (same id = difference within run-to-run noise)

Short search budget only. Re-train with the full recipe (see `python train.py -h`); neighbouring ranks may reorder.

Use: `python train.py --data ...` writes model.pt and preprocess.json; then `python predict.py --data new_file` predicts in the original label units (`--export model.pt2` saves a deployable graph). Training recipe: modern {'augment': True, 'label_smoothing': 0.1}.

Block diagram: `architecture.svg`. Training notebook (Jupyter / Colab): `../baselines.ipynb`.
