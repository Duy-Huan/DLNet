# QA/QC: `hsi/indian_pines_block`

| Bước | Kết quả | Chi tiết |
|---|---|---|
| 1_design | ĐẠT | seconds=604.6, packages=['top1_zoo_afno_block', 'top2_afno2d', 'top3_zoo_gfnet_block'] |
| 2_package_tests | ĐẠT |  |
| 3_train_top1 | ĐẠT | seconds=31.0, epochs=30, package=top1_zoo_afno_block, weights_kb=156 |
| 4_predict_export | ĐẠT | seconds=12.3 |
| 5_qc | ĐẠT | metric=accuracy, test_from_predictions=0.5037, test_printed_by_train_py=0.5037427224840588, splits_disjoint=True, n_test=3607, cube=[145, 145, 200], classes=16 |

Chỉ số trên tập TEST (tính lại từ predictions.npy của gói top-1 đã train đầy đủ):

```
{
 "accuracy": 0.5037427224840588,
 "balanced_accuracy": 0.47488732531351985,
 "kappa": 0.4287639230681778,
 "per_class_recall": [
  0.0,
  0.2302,
  0.7829,
  0.0,
  0.0,
  0.9162,
  0.0,
  1.0,
  0.0,
  0.4984,
  0.9135,
  0.7106,
  1.0,
  0.3191,
  0.2775,
  0.0
 ]
}
```

So với baseline cổ điển cùng split (leaderboard): `{"beats_reference": true, "val_margin": 0.17779, "val_noise": 0.04301, "test_margin": 0.07999, "test_noise": 0.00832, "top1_val": 0.4668, "best_reference": "random_forest_spectral_spatial", "best_reference_val": 0.28901}`

Xem `leaderboard.md` (xếp hạng tìm kiếm), các thư mục `top*_*/` (gói PyTorch), các log `1_..5_`.
