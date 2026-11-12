# top1: zoo_gfnet_block_patchify_pos  ['top1', 'most-stable']

- Config: `{'family': 'zoo_gfnet_block_patchify_pos', 'spec': {'op': 'seq', 'body': [{'op': 'patchify', 'dim': 64, 'patch': 4}, {'op': 'pos', 'mode': 'learned'}, {'op': 'gfnet_block', 'mlp_ratio': 4, 'dropout': 0.1}, {'op': 'wukong', 'layers': 3, 'lcb': 40, 'fmb': 40, 'rank_k': 8, 'dropout': 0.1}]}, 'input': {'kind': 'map', 'dims': [3, 64, 64]}, 'out_dim': 37, 'head': 'cls', 'dropout': 0.0, 'origin': 'random', 'lr': 0.003, 'wd': 0.01, 'batch': 64, 'flops': 24668800, 'act_mb': 1.50082, 'vram_mb': 199.825744, 'seed_prior': -1.169, 'seed_rank': 1, 'seed_score': 91.0, 'seed_traits': ['global_attention', 'position_aware']}`
- Origin: random
- Params: 482,549
- Cost model: 24,668,800 forward FLOPs per sample (matrix multiply and convolution only), about 200 MB training memory at batch 64 (estimate, not measured on a GPU)
- Validation (accuracy): 0.1077 +/- 0.0114 (3 runs on 1 split(s), 4 epochs)
- Test (mean over runs): 0.1009 (bal 0.1009, kappa 0.0759), primary metric +/- 0.0067
- Hyperparameters: lr 0.003, weight decay 0.01, batch 64, dropout 0.0, AdamW, cosine, 100 epochs (full recipe), clip 5.0
  - Adopted: searched. Estimated lr 0.0041, batch 512, wd 0.028, B_simple 805; validation 0.1077 (searched) vs 0.1140 (estimated)
- Model file: `model.py` is plain PyTorch (no DLNet import): the network's own source, identical to the searched network
- Search time spent on this candidate: 83.7s
- Indistinguishable-group id: 0 (same id = difference within run-to-run noise)

Short search budget only. Re-train with the full recipe (see `python train.py -h`); neighbouring ranks may reorder.

Use: `python train.py --data ...` writes model.pt and preprocess.json; then `python predict.py --data new_file` predicts in the original label units (`--export model.pt2` saves a deployable graph). Training recipe: modern {'augment': True, 'label_smoothing': 0.1}.

Block diagram: `architecture.svg`. Training notebook (Jupyter / Colab): `../baselines.ipynb`.
