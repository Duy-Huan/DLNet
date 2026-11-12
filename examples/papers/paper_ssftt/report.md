# SSFTT (paper baseline)

- Title: Spectral-Spatial Feature Tokenization Transformer for Hyperspectral Image Classification
- Venue: IEEE Transactions on Geoscience and Remote Sensing, 2022
- Url: https://doi.org/10.1109/TGRS.2022.3144158
- Code: https://github.com/zgr6010/HSI_SSFTT
- Fidelity level: B: verified against the authors' code (same weights give the same outputs)
- Check against https://github.com/zgr6010/HSI_SSFTT @ 994f5d7: parameters 153,224 vs 153,224, max |output difference| 0.0
- Parameters: 153,224

Authors' training recipe: `{'optimizer': 'adam', 'lr': 0.001, 'batch': 64, 'epochs': 100}`
