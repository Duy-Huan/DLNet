# top3: zoo_fnet_block_patchify  ['best-small']

- Config: `{'family': 'zoo_fnet_block_patchify', 'spec': {'op': 'seq', 'body': [{'op': 'patchify', 'dim': 64, 'patch': 4}, {'op': 'fnet_block', 'ratio': 2, 'dropout': 0.0}]}, 'input': {'kind': 'map', 'dims': [3, 32, 32]}, 'out_dim': 10, 'head': 'cls', 'dropout': 0.0, 'origin': 'random', 'lr': 0.003, 'wd': 0.01, 'batch': 32, 'flops': 2491648, 'act_mb': 0.164136, 'vram_mb': 10.834592, 'seed_prior': -1.0, 'seed_rank': 58, 'seed_score': 34.0, 'seed_traits': ['global_attention', 'position_aware']}`
- Origin: random
- Params: 20,618
- Cost model: 2,491,648 forward FLOPs per sample (matrix multiply and convolution only), about 11 MB training memory at batch 32 (estimate, not measured on a GPU)
- Validation (accuracy): 0.3872 +/- 0.0141 (3 runs on 1 split(s), 30 epochs)
- Test (mean over runs): 0.4125 (bal 0.4125, kappa 0.3472), primary metric +/- 0.0080
- Hyperparameters: lr 0.003, weight decay 0.01, batch 32, dropout 0.0, AdamW, cosine, 100 epochs (full recipe), clip 5.0 (searched; not estimated)
- Model file: `model.py` is plain PyTorch (no DLNet import): the network's own source, identical to the searched network
- Search time spent on this candidate: 57.1s
- Indistinguishable-group id: 2 (same id = difference within run-to-run noise)

Short search budget only. Re-train with the full recipe (see `python train.py -h`); neighbouring ranks may reorder.

Use: `python train.py --data ...` writes model.pt and preprocess.json; then `python predict.py --data new_file` predicts in the original label units (`--export model.pt2` saves a deployable graph). Training recipe: modern {'augment': True, 'label_smoothing': 0.1}.

Block diagram: `architecture.svg`. Training notebook (Jupyter / Colab): `../baselines.ipynb`.
