# QA/QC: `image_classification/imagenette`

| Bước | Kết quả | Chi tiết |
|---|---|---|
| 1_design | ĐẠT | seconds=603.6, packages=['top1_vgg', 'top2_zoo_gfnet_block_patchify_pos', 'top3_resnet'] |
| 2_package_tests | ĐẠT |  |
| 3_train_top1 | ĐẠT | seconds=124.2, epochs=30, package=top1_vgg, weights_kb=112 |
| 4_predict_export | ĐẠT | seconds=8.2 |
| 5_qc | ĐẠT | metric=accuracy, test_from_predictions=0.6067, test_printed_by_train_py=0.6066666666666667, splits_disjoint=True, n_test=600, image=[64, 64, 3], classes=10 |

Chỉ số trên tập TEST (tính lại từ predictions.npy của gói top-1 đã train đầy đủ):

```
{
 "accuracy": 0.6066666666666667,
 "balanced_accuracy": 0.6066666666666667,
 "kappa": 0.562962962962963,
 "per_class_recall": [
  0.75,
  0.7333,
  0.65,
  0.2,
  0.6,
  0.6667,
  0.6,
  0.4667,
  0.65,
  0.75
 ]
}
```

So với baseline cổ điển cùng split (leaderboard): `{"beats_reference": true, "val_margin": 0.18667, "val_noise": 0.01996, "test_margin": 0.21, "test_noise": 0.02004, "top1_val": 0.605, "best_reference": "random_forest", "best_reference_val": 0.41833}`

Xem `leaderboard.md` (xếp hạng tìm kiếm), các thư mục `top*_*/` (gói PyTorch), các log `1_..5_`.
