# StarNet-S1 (paper baseline)

- Title: Rewrite the Stars
- Venue: CVPR 2024
- Url: https://arxiv.org/abs/2403.19967
- Code: https://github.com/ma-xu/Rewrite-the-Stars
- Fidelity level: B: verified against the authors' code (same weights give the same outputs)
- Check against https://github.com/ma-xu/Rewrite-the-Stars @ c999eb5: parameters 2,869,168 vs 2,869,168, max |output difference| 0.0
- Parameters: 2,869,168 (paper: 2,900,000, -1.1%)

Reported by the paper (not measured here):

- ImageNet-1k: accuracy 0.735 (224 x 224, 300 epochs, top-1)

Authors' training recipe: `{'optimizer': 'adamw', 'lr': 0.003, 'weight_decay': 0.025, 'batch': 2048, 'epochs': 300}`
