# top2: zoo_ffn_patchify_pos  ['most-different']

- Config: `{'family': 'zoo_ffn_patchify_pos', 'spec': {'op': 'seq', 'body': [{'op': 'patchify', 'dim': 64, 'patch': 4}, {'op': 'pos', 'mode': 'learned'}, {'op': 'ffn', 'ratio': 2, 'act': 'relu', 'dropout': 0.1}, {'op': 'tt_linear', 'out': 128, 'rank': 8}]}, 'input': {'kind': 'map', 'dims': [3, 32, 32]}, 'out_dim': 10, 'head': 'cls', 'dropout': 0.0, 'origin': 'random', 'lr': 0.001, 'wd': 0.01, 'batch': 32, 'flops': 4065792, 'act_mb': 0.213544, 'vram_mb': 14.097056, 'seed_prior': -1.0, 'seed_rank': 77, 'seed_score': 15.0, 'seed_traits': ['global_attention', 'position_aware']}`
- Origin: random
- Params: 26,890
- Cost model: 4,065,792 forward FLOPs per sample (matrix multiply and convolution only), about 14 MB training memory at batch 32 (estimate, not measured on a GPU)
- Validation (accuracy): 0.4842 +/- 0.0144 (3 runs on 1 split(s), 30 epochs)
- Test (mean over runs): 0.4714 (bal 0.4714, kappa 0.4127), primary metric +/- 0.0106
- Hyperparameters: lr 0.001, weight decay 0.01, batch 32, dropout 0.0, AdamW, cosine, 100 epochs (full recipe), clip 5.0 (searched; not estimated)
- Model file: `model.py` is plain PyTorch (no DLNet import): the network's own source, identical to the searched network
- Search time spent on this candidate: 95.9s
- Indistinguishable-group id: 1 (same id = difference within run-to-run noise)

Short search budget only. Re-train with the full recipe (see `python train.py -h`); neighbouring ranks may reorder.

Use: `python train.py --data ...` writes model.pt and preprocess.json; then `python predict.py --data new_file` predicts in the original label units (`--export model.pt2` saves a deployable graph). Training recipe: modern {'augment': True, 'label_smoothing': 0.1}.
