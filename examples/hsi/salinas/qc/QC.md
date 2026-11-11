# QA/QC: `hsi/salinas`

| Bước | Kết quả | Chi tiết |
|---|---|---|
| 1_design | ĐẠT | seconds=594.9, packages=['top1_afno2d', 'top2_ffc2d', 'top3_cnn2d'] |
| 2_package_tests | ĐẠT |  |
| 3_train_top1 | ĐẠT | seconds=88.3, epochs=30, package=top1_afno2d, weights_kb=359 |
| 4_predict_export | ĐẠT | seconds=67.7 |
| 5_qc | ĐẠT | metric=accuracy, test_from_predictions=0.9941, test_printed_by_train_py=0.9941111945037815, splits_disjoint=True, n_test=51963, cube=[512, 217, 204], classes=16 |

Chỉ số trên tập TEST (tính lại từ predictions.npy của gói top-1 đã train đầy đủ):

```
{
 "accuracy": 0.9941111945037815,
 "balanced_accuracy": 0.9949425856659131,
 "kappa": 0.9934398098631998,
 "per_class_recall": [
  1.0,
  1.0,
  1.0,
  0.9888,
  0.9782,
  1.0,
  1.0,
  0.9978,
  1.0,
  0.9943,
  1.0,
  0.9995,
  0.9989,
  0.9912,
  0.9751,
  0.9954
 ]
}
```

So với baseline cổ điển cùng split (leaderboard): `{"beats_reference": true, "val_margin": 0.02031, "val_noise": 0.00302, "test_margin": 0.02069, "test_noise": 0.00044, "top1_val": 0.99723, "best_reference": "random_forest_spectral_spatial", "best_reference_val": 0.97692}`

Xem `leaderboard.md` (xếp hạng tìm kiếm), các thư mục `top*_*/` (gói PyTorch), các log `1_..5_`.
