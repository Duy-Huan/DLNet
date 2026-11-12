# QA/QC: `image_classification/oxford_pets_breeds`

| Bước | Kết quả | Chi tiết |
|---|---|---|
| 1_design | ĐẠT | seconds=591.3, packages=['top1_zoo_gfnet_block_patchify_pos', 'top2_resnet'] |
| 2_package_tests | ĐẠT |  |
| 3_train_top1 | ĐẠT | seconds=153.7, epochs=30, package=top1_zoo_gfnet_block_patchify_pos, weights_kb=1897 |
| 4_predict_export | ĐẠT | seconds=7.7 |
| 5_qc | ĐẠT | metric=accuracy, test_from_predictions=0.1662, test_printed_by_train_py=0.1662162162162162, splits_disjoint=True, n_test=740, image=[64, 64, 3], classes=37 |

Chỉ số trên tập TEST (tính lại từ predictions.npy của gói top-1 đã train đầy đủ):

```
{
 "accuracy": 0.1662162162162162,
 "balanced_accuracy": 0.1662162162162162,
 "kappa": 0.14305555555555555,
 "per_class_recall": [
  0.15,
  0.1,
  0.0,
  0.3,
  0.05,
  0.1,
  0.15,
  0.3,
  0.1,
  0.05,
  0.1,
  0.15,
  0.2,
  0.1,
  0.0,
  0.35,
  0.05,
  0.0,
  0.4,
  0.25,
  0.05,
  0.05,
  0.55,
  0.35,
  0.0,
  0.15,
  0.05,
  0.2,
  0.2,
  0.2,
  0.0,
  0.15,
  0.15,
  0.35,
  0.3,
  0.3,
  0.2
 ]
}
```

So với baseline cổ điển cùng split (leaderboard): `{"beats_reference": false, "val_margin": -0.01667, "val_noise": 0.01139, "test_margin": 0.00225, "test_noise": 0.01107, "top1_val": 0.10766, "best_reference": "random_forest", "best_reference_val": 0.12432}`

Xem `leaderboard.md` (xếp hạng tìm kiếm), các thư mục `top*_*/` (gói PyTorch), các log `1_..5_`.
