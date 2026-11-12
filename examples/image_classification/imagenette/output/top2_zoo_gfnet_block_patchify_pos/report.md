# top2: zoo_gfnet_block_patchify_pos  ['most-different']

- Config: `{'family': 'zoo_gfnet_block_patchify_pos', 'spec': {'op': 'seq', 'body': [{'op': 'patchify', 'dim': 64, 'patch': 4}, {'op': 'pos', 'mode': 'learned'}, {'op': 'gfnet_block', 'mlp_ratio': 4, 'dropout': 0.1}, {'op': 'wukong', 'layers': 3, 'lcb': 40, 'fmb': 40, 'rank_k': 8, 'dropout': 0.1}]}, 'input': {'kind': 'map', 'dims': [3, 64, 64]}, 'out_dim': 10, 'head': 'cls', 'dropout': 0.0, 'origin': 'random', 'lr': 0.003, 'wd': 0.01, 'batch': 64, 'flops': 24665344, 'act_mb': 1.500712, 'vram_mb': 199.78384, 'seed_prior': -1.713, 'seed_rank': 1, 'seed_score': 91.0, 'seed_traits': ['global_attention', 'position_aware']}`
- Origin: random
- Params: 480,794
- Cost model: 24,665,344 forward FLOPs per sample (matrix multiply and convolution only), about 200 MB training memory at batch 64 (estimate, not measured on a GPU)
- Validation (accuracy): 0.5417 +/- 0.0203 (1 runs on 1 split(s), 30 epochs)
- Test (mean over runs): 0.5417 (bal 0.5417, kappa 0.4907), primary metric +/- 0.0000
- Hyperparameters: lr 0.003, weight decay 0.01, batch 64, dropout 0.0, AdamW, cosine, 100 epochs (full recipe), clip 5.0 (searched; not estimated)
- Model file: `model.py` is plain PyTorch (no DLNet import): the network's own source, identical to the searched network
- Search time spent on this candidate: 98.5s
- Indistinguishable-group id: 1 (same id = difference within run-to-run noise)

Short search budget only. Re-train with the full recipe (see `python train.py -h`); neighbouring ranks may reorder.

Use: `python train.py --data ...` writes model.pt and preprocess.json; then `python predict.py --data new_file` predicts in the original label units (`--export model.pt2` saves a deployable graph). Training recipe: modern {'augment': True, 'label_smoothing': 0.1}.
