# top2: zoo_ffn_gated_attn_block_gfnet_block  ['most-different']

- Config: `{'family': 'zoo_ffn_gated_attn_block_gfnet_block', 'spec': {'op': 'seq', 'body': [{'op': 'patchify', 'dim': 64, 'patch': 1}, {'op': 'pos', 'mode': 'sine'}, {'op': 'gated_attn_block', 'heads': 2, 'ratio': 2, 'gate': 'elementwise', 'causal': 1}, {'op': 'ffn', 'ratio': 4, 'act': 'gelu', 'dropout': 0.0}, {'op': 'gfnet_block', 'mlp_ratio': 4, 'dropout': 0.1}]}, 'input': {'kind': 'map', 'dims': [30, 9, 9]}, 'out_dim': 14, 'head': 'cls', 'dropout': 0.0, 'origin': 'random', 'lr': 0.003, 'wd': 0.01, 'batch': 32, 'flops': 19908352, 'act_mb': 0.99564, 'vram_mb': 65.64016, 'seed_prior': -1.284, 'seed_rank': 175, 'seed_score': 2.0, 'seed_traits': ['global_attention', 'position_aware']}`
- Origin: random
- Params: 119,950
- Cost model: 19,908,352 forward FLOPs per sample (matrix multiply and convolution only), about 66 MB training memory at batch 32 (estimate, not measured on a GPU)
- Validation (accuracy): 0.9990 +/- 0.0055 (3 runs on 1 split(s), 30 epochs)
- Test (mean over runs): 0.9941 (bal 0.9943, kappa 0.9936), primary metric +/- 0.0002
- Hyperparameters: lr 0.003, weight decay 0.01, batch 32, dropout 0.0, AdamW, cosine, 100 epochs (full recipe), clip 5.0
  - Adopted: searched. Estimated lr 0.002, batch 16, wd 0.0071, B_simple 10; validation 0.9990 (searched) vs 1.0000 (estimated)
- Model file: `model.py` is plain PyTorch (no DLNet import): the network's own source, identical to the searched network
- Search time spent on this candidate: 32.0s
- Indistinguishable-group id: 0 (same id = difference within run-to-run noise)

Short search budget only. Re-train with the full recipe (see `python train.py -h`); neighbouring ranks may reorder.

Use: `python train.py --data ...` writes model.pt and preprocess.json; then `python predict.py --data new_file` predicts in the original label units (`--export model.pt2` saves a deployable graph). Training recipe: modern {'augment': True, 'label_smoothing': 0.1}.

Block diagram: `architecture.svg`. Training notebook (Jupyter / Colab): `../baselines.ipynb`.
