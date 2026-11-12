# top3: zoo_gfnet_block  ['best-small']

- Config: `{'family': 'zoo_gfnet_block', 'spec': {'op': 'seq', 'body': [{'op': 'res', 'body': {'op': 'conv_bn_act', 'out': 32, 'k': 3, 's': 1, 'act': 'relu', 'norm': 'batch'}, 'drop_path': 0.0}, {'op': 'gfnet_block', 'mlp_ratio': 4, 'dropout': 0.0}, {'op': 'gfnet_block', 'mlp_ratio': 4, 'dropout': 0.0}, {'op': 'gap', 'mode': 'avg'}]}, 'input': {'kind': 'map', 'dims': [30, 9, 9]}, 'out_dim': 16, 'head': 'cls', 'dropout': 0.0, 'origin': 'wrap_res<-gfnet2d', 'lr': 0.003, 'wd': 0.001, 'batch': 32, 'flops': 4210432, 'act_mb': 0.394432, 'vram_mb': 25.77024}`
- Origin: wrap_res<-gfnet2d
- Params: 32,912
- Cost model: 4,210,432 forward FLOPs per sample (matrix multiply and convolution only), about 26 MB training memory at batch 32 (estimate, not measured on a GPU)
- Validation (accuracy): 0.4137 +/- 0.0308 (2 runs on 2 split(s), 30 epochs)
- Test (mean over runs): 0.4629 (bal 0.4854, kappa 0.3724), primary metric +/- 0.0057
- Hyperparameters: lr 0.003, weight decay 0.001, batch 32, dropout 0.0, AdamW, cosine, 100 epochs (full recipe), clip 5.0 (searched; not estimated)
- Search time spent on this candidate: 31.7s
- Indistinguishable-group id: 0 (same id = difference within run-to-run noise)

Short search budget only. Re-train with the full recipe (see `python train.py -h`); neighbouring ranks may reorder.

Use: `python train.py --data ...` writes model.pt and preprocess.json; then `python predict.py --data new_file` predicts in the original label units (`--export model.pt2` saves a deployable graph). Training recipe: modern {'augment': True, 'label_smoothing': 0.1}.

Block diagram: `architecture.svg`. Training notebook (Jupyter / Colab): `../baselines.ipynb`.
