# QA/QC: `hsi/pavia_university`

| Bước | Kết quả | Chi tiết |
|---|---|---|
| 1_design | ĐẠT | seconds=624.0, packages=['top1_spec_fft2d', 'top2_zoo_afno_block_moe', 'top3_ffc2d'] |
| 2_package_tests | ĐẠT |  |
| 3_train_top1 | ĐẠT | seconds=42.3, epochs=30, package=top1_spec_fft2d, weights_kb=268 |
| 4_predict_export | ĐẠT | seconds=36.4 |
| 5_qc | ĐẠT | metric=accuracy, test_from_predictions=0.9928, test_printed_by_train_py=0.9928157420486095, splits_disjoint=True, n_test=41062, cube=[610, 340, 103], classes=9 |

Chỉ số trên tập TEST (tính lại từ predictions.npy của gói top-1 đã train đầy đủ):

```
{
 "accuracy": 0.9928157420486095,
 "balanced_accuracy": 0.9855913431027731,
 "kappa": 0.9904787379225902,
 "per_class_recall": [
  0.9923,
  1.0,
  0.935,
  0.9878,
  1.0,
  1.0,
  0.9765,
  0.9887,
  0.9901
 ]
}
```

So với baseline cổ điển cùng split (leaderboard): `{"beats_reference": true, "val_margin": 0.04278, "val_noise": 0.0034, "test_margin": 0.03747, "test_noise": 0.00049, "top1_val": 0.99378, "best_reference": "random_forest_spectral_spatial", "best_reference_val": 0.95099}`

Xem `leaderboard.md` (xếp hạng tìm kiếm), các thư mục `top*_*/` (gói PyTorch), các log `1_..5_`.
