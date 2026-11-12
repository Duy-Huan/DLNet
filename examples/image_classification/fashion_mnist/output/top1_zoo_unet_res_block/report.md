# top1: zoo_unet_res_block  ['top1']

- Config: `{'family': 'zoo_unet_res_block', 'spec': {'op': 'seq', 'body': [{'op': 'seq', 'body': [{'op': 'unet_res_block', 'out': 16, 'k': 3, 's': 1}, {'op': 'unet_res_block', 'out': 16, 'k': 3, 's': 1}]}, {'op': 'seq', 'body': [{'op': 'unet_res_block', 'out': 32, 'k': 3, 's': 2}, {'op': 'unet_res_block', 'out': 32, 'k': 3, 's': 1}]}, {'op': 'seq', 'body': [{'op': 'unet_res_block', 'out': 64, 'k': 3, 's': 2}, {'op': 'unet_res_block', 'out': 64, 'k': 3, 's': 1}]}]}, 'input': {'kind': 'map', 'dims': [1, 28, 28]}, 'out_dim': 10, 'head': 'cls', 'dropout': 0.0, 'origin': 'preset:nnunet_res', 'lr': 0.003, 'wd': 0.0001, 'batch': 64, 'flops': 36780288, 'act_mb': 1.229608, 'vram_mb': 160.134816, 'seed_prior': 0.0, 'seed_rank': 20, 'seed_score': 73.0, 'seed_traits': ['group_norm', 'local_conv']}`
- Origin: preset:nnunet_res
- Params: 171,562
- Cost model: 36,780,288 forward FLOPs per sample (matrix multiply and convolution only), about 160 MB training memory at batch 64 (estimate, not measured on a GPU)
- Validation (accuracy): 0.8529 +/- 0.0102 (2 runs on 1 split(s), 30 epochs)
- Test (mean over runs): 0.8679 (bal 0.8679, kappa 0.8532), primary metric +/- 0.0038
- Hyperparameters: lr 0.003, weight decay 0.0001, batch 64, dropout 0.0, AdamW, cosine, 100 epochs (full recipe), clip 5.0 (searched; not estimated)
- Model file: `model.py` is plain PyTorch (no DLNet import): the network's own source, identical to the searched network
- Search time spent on this candidate: 250.3s
- Indistinguishable-group id: 0 (same id = difference within run-to-run noise)

Short search budget only. Re-train with the full recipe (see `python train.py -h`); neighbouring ranks may reorder.

Use: `python train.py --data ...` writes model.pt and preprocess.json; then `python predict.py --data new_file` predicts in the original label units (`--export model.pt2` saves a deployable graph). Training recipe: modern {'augment': True, 'label_smoothing': 0.1}.

Block diagram: `architecture.svg`. Training notebook (Jupyter / Colab): `../baselines.ipynb`.
