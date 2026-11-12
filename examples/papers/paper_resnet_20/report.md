# ResNet-20 (paper baseline)

- Title: Deep Residual Learning for Image Recognition
- Venue: CVPR 2016
- Url: https://arxiv.org/abs/1512.03385
- Fidelity level: C: written from the paper, NOT verified against the authors' code
- Parameters: 272,378 (paper: 270,000, +0.9%)

Reported by the paper (not measured here):

- CIFAR-10: accuracy 0.9125 (45k/5k split, 64k iterations, test error 8.75%)

Authors' training recipe: `{'optimizer': 'sgd', 'lr': 0.1, 'weight_decay': 0.0001, 'batch': 128, 'epochs': 164}`
