# top2: zoo_gfnet_block_spectral_gf_vgg_stage  ['most-stable', 'most-different']

- Config: `{'family': 'zoo_gfnet_block_spectral_gf_vgg_stage', 'spec': {'op': 'seq', 'body': [{'op': 'spectral_gf'}, {'op': 'seq', 'body': [{'op': 'vgg_stage', 'out': 64, 'n': 3, 'norm': 'batch', 'pool': 1}, {'op': 'gfnet_block', 'mlp_ratio': 4, 'dropout': 0.0}, {'op': 'gfnet_block', 'mlp_ratio': 4, 'dropout': 0.0}, {'op': 'gap', 'mode': 'avg'}]}]}, 'input': {'kind': 'map', 'dims': [30, 9, 9]}, 'out_dim': 16, 'head': 'cls', 'dropout': 0.0, 'origin': 'swap<-fourier_ss', 'lr': 0.001, 'wd': 0.001, 'batch': 32, 'flops': 25362176, 'act_mb': 0.922936, 'vram_mb': 61.798656}`
- Origin: swap<-fourier_ss
- Params: 170,672
- Cost model: 25,362,176 forward FLOPs per sample (matrix multiply and convolution only), about 62 MB training memory at batch 32 (estimate, not measured on a GPU)
- Validation (accuracy): 0.9849 +/- 0.0038 (2 runs on 1 split(s), 30 epochs)
- Test (mean over runs): 0.9867 (bal 0.9864, kappa 0.9848), primary metric +/- 0.0001
- Hyperparameters: lr 0.001, weight decay 0.001, batch 32, dropout 0.0, AdamW, cosine, 100 epochs (full recipe), clip 5.0 (searched; not estimated)
- Search time spent on this candidate: 80.9s
- Indistinguishable-group id: 0 (same id = difference within run-to-run noise)

Short search budget only. Re-train with the full recipe (see `python train.py -h`); neighbouring ranks may reorder.

Use: `python train.py --data ...` writes model.pt and preprocess.json; then `python predict.py --data new_file` predicts in the original label units (`--export model.pt2` saves a deployable graph). Training recipe: modern {'augment': True, 'label_smoothing': 0.1}.

Block diagram: `architecture.svg`. Training notebook (Jupyter / Colab): `../baselines.ipynb`.
