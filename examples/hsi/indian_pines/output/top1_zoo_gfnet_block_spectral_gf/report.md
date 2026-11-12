# top1: zoo_gfnet_block_spectral_gf  ['top1']

- Config: `{'family': 'zoo_gfnet_block_spectral_gf', 'spec': {'op': 'seq', 'body': [{'op': 'spectral_gf'}, {'op': 'seq', 'body': [{'op': 'conv_bn_act', 'out': 64, 'k': 3, 's': 1, 'act': 'relu', 'norm': 'batch'}, {'op': 'gfnet_block', 'mlp_ratio': 4, 'dropout': 0.0}, {'op': 'gap', 'mode': 'avg'}]}]}, 'input': {'kind': 'map', 'dims': [30, 9, 9]}, 'out_dim': 16, 'head': 'cls', 'dropout': 0.0, 'origin': 'delete<-fourier_ss', 'lr': 0.001, 'wd': 0.001, 'batch': 32, 'flops': 8109824, 'act_mb': 0.425272, 'vram_mb': 28.138752}`
- Origin: delete<-fourier_ss
- Params: 57,584
- Cost model: 8,109,824 forward FLOPs per sample (matrix multiply and convolution only), about 28 MB training memory at batch 32 (estimate, not measured on a GPU)
- Validation (accuracy): 0.9850 +/- 0.0038 (3 runs on 1 split(s), 30 epochs)
- Test (mean over runs): 0.9861 (bal 0.9835, kappa 0.9841), primary metric +/- 0.0014
- Hyperparameters: lr 0.001, weight decay 0.001, batch 32, dropout 0.0, AdamW, cosine, 100 epochs (full recipe), clip 5.0 (searched; not estimated)
- Search time spent on this candidate: 52.1s
- Indistinguishable-group id: 0 (same id = difference within run-to-run noise)

Short search budget only. Re-train with the full recipe (see `python train.py -h`); neighbouring ranks may reorder.

Use: `python train.py --data ...` writes model.pt and preprocess.json; then `python predict.py --data new_file` predicts in the original label units (`--export model.pt2` saves a deployable graph). Training recipe: modern {'augment': True, 'label_smoothing': 0.1}.

Block diagram: `architecture.svg`. Training notebook (Jupyter / Colab): `../baselines.ipynb`.
