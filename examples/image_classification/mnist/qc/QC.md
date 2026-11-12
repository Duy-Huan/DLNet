# QA/QC: `image_classification/mnist`

| Bước | Kết quả | Chi tiết |
|---|---|---|
| 1_design | ĐẠT | seconds=608.7, packages=['top1_cnn', 'top2_resnet', 'top3_zoo_gfnet_block_patchify_pos'] |
| 2_package_tests | ĐẠT |  |
| 3_train_top1 | ĐẠT | seconds=66.3, epochs=30, package=top1_cnn, weights_kb=295 |
| 4_predict_export | ĐẠT | seconds=4.8 |
| 5_qc | ĐẠT | metric=accuracy, test_from_predictions=0.9883, test_printed_by_train_py=0.9883333333333333, splits_disjoint=True, n_test=1200, image=[28, 28], classes=10 |

Chỉ số trên tập TEST (tính lại từ predictions.npy của gói top-1 đã train đầy đủ):

```
{
 "accuracy": 0.9883333333333333,
 "balanced_accuracy": 0.9883333333333333,
 "kappa": 0.9870370370370369,
 "per_class_recall": [
  0.9917,
  0.9917,
  0.975,
  0.9833,
  0.9917,
  0.9917,
  1.0,
  0.9833,
  0.9917,
  0.9833
 ]
}
```

So với baseline cổ điển cùng split (leaderboard): `{"beats_reference": true, "val_margin": 0.05417, "val_noise": 0.00331, "test_margin": 0.06, "test_noise": 0.0031, "top1_val": 0.98667, "best_reference": "hist_gradient_boosting", "best_reference_val": 0.9325}`

Xem `leaderboard.md` (xếp hạng tìm kiếm), các thư mục `top*_*/` (gói PyTorch), các log `1_..5_`.
