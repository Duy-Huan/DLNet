# QA/QC: `image_classification/cifar10`

| Bước | Kết quả | Chi tiết |
|---|---|---|
| 1_design | ĐẠT | seconds=603.3, packages=['top1_cnn', 'top2_zoo_ffn_patchify_pos', 'top3_zoo_fnet_block_patchify'] |
| 2_package_tests | ĐẠT |  |
| 3_train_top1 | ĐẠT | seconds=87.7, epochs=30, package=top1_cnn, weights_kb=296 |
| 4_predict_export | ĐẠT | seconds=6.2 |
| 5_qc | ĐẠT | metric=accuracy, test_from_predictions=0.6475, test_printed_by_train_py=0.6475, splits_disjoint=True, n_test=1200, image=[32, 32, 3], classes=10 |

Chỉ số trên tập TEST (tính lại từ predictions.npy của gói top-1 đã train đầy đủ):

```
{
 "accuracy": 0.6475,
 "balanced_accuracy": 0.6475000000000001,
 "kappa": 0.6083333333333333,
 "per_class_recall": [
  0.75,
  0.8417,
  0.5333,
  0.3333,
  0.5,
  0.6167,
  0.7,
  0.6917,
  0.8083,
  0.7
 ]
}
```

So với baseline cổ điển cùng split (leaderboard): `{"beats_reference": true, "val_margin": 0.21306, "val_noise": 0.01404, "test_margin": 0.21, "test_noise": 0.01399, "top1_val": 0.61639, "best_reference": "hist_gradient_boosting", "best_reference_val": 0.40333}`

Xem `leaderboard.md` (xếp hạng tìm kiếm), các thư mục `top*_*/` (gói PyTorch), các log `1_..5_`.
