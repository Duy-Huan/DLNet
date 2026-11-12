# top2: zoo_gfnet_block_patchify_pos  ['most-different']

- Config: `{'family': 'zoo_gfnet_block_patchify_pos', 'spec': {'op': 'seq', 'body': [{'op': 'patchify', 'dim': 64, 'patch': 4}, {'op': 'pos', 'mode': 'learned'}, {'op': 'gfnet_block', 'mlp_ratio': 4, 'dropout': 0.1}, {'op': 'wukong', 'layers': 3, 'lcb': 40, 'fmb': 40, 'rank_k': 8, 'dropout': 0.1}]}, 'input': {'kind': 'map', 'dims': [1, 28, 28]}, 'out_dim': 10, 'head': 'cls', 'dropout': 0.0, 'origin': 'random', 'lr': 0.003, 'wd': 0.01, 'batch': 64, 'flops': 5917440, 'act_mb': 0.434248, 'vram_mb': 61.49392, 'seed_prior': -0.506, 'seed_rank': 9, 'seed_score': 83.0, 'seed_traits': ['global_attention', 'position_aware']}`
- Origin: random
- Params: 369,386
- Cost model: 5,917,440 forward FLOPs per sample (matrix multiply and convolution only), about 61 MB training memory at batch 64 (estimate, not measured on a GPU)
- Validation (accuracy): 0.8287 +/- 0.0109 (2 runs on 1 split(s), 30 epochs)
- Test (mean over runs): 0.8242 (bal 0.8242, kappa 0.8046), primary metric +/- 0.0017
- Hyperparameters: lr 0.003, weight decay 0.01, batch 64, dropout 0.0, AdamW, cosine, 100 epochs (full recipe), clip 5.0 (searched; not estimated)
- Model file: `model.py` is plain PyTorch (no DLNet import): the network's own source, identical to the searched network
- Search time spent on this candidate: 132.1s
- Indistinguishable-group id: 1 (same id = difference within run-to-run noise)

Short search budget only. Re-train with the full recipe (see `python train.py -h`); neighbouring ranks may reorder.

Use: `python train.py --data ...` writes model.pt and preprocess.json; then `python predict.py --data new_file` predicts in the original label units (`--export model.pt2` saves a deployable graph). Training recipe: modern {'augment': True, 'label_smoothing': 0.1}.

Block diagram: `architecture.svg`. Training notebook (Jupyter / Colab): `../baselines.ipynb`.
