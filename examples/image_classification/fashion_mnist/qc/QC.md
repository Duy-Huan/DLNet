# QA/QC: `image_classification/fashion_mnist`

| Bước | Kết quả | Chi tiết |
|---|---|---|
| 1_design | ĐẠT | seconds=568.0, packages=['top1_zoo_unet_res_block', 'top2_zoo_gfnet_block_patchify_pos', 'top3_resnet'] |
| 2_package_tests | ĐẠT |  |
| 3_train_top1 | ĐẠT | seconds=155.7, epochs=30, package=top1_zoo_unet_res_block, weights_kb=675 |
| 4_predict_export | ĐẠT | seconds=7.9 |
| 5_qc | ĐẠT | metric=accuracy, test_from_predictions=0.875, test_printed_by_train_py=0.875, splits_disjoint=True, n_test=1200, image=[28, 28], classes=10 |

Chỉ số trên tập TEST (tính lại từ predictions.npy của gói top-1 đã train đầy đủ):

```
{
 "accuracy": 0.875,
 "balanced_accuracy": 0.875,
 "kappa": 0.8611111111111112,
 "per_class_recall": [
  0.8667,
  0.9833,
  0.875,
  0.875,
  0.85,
  0.975,
  0.4917,
  0.9167,
  0.9583,
  0.9583
 ]
}
```

So với baseline cổ điển cùng split (leaderboard): `{"beats_reference": true, "val_margin": 0.02375, "val_noise": 0.01022, "test_margin": 0.03625, "test_noise": 0.00977, "top1_val": 0.85292, "best_reference": "hist_gradient_boosting", "best_reference_val": 0.82917}`

Xem `leaderboard.md` (xếp hạng tìm kiếm), các thư mục `top*_*/` (gói PyTorch), các log `1_..5_`.
