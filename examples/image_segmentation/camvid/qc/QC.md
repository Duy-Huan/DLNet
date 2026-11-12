# QA/QC: `image_segmentation/camvid`

| Bước | Kết quả | Chi tiết |
|---|---|---|
| 1_design | ĐẠT | seconds=430.1, packages=['top1_fcn'] |
| 2_package_tests | ĐẠT |  |
| 3_train_top1 | ĐẠT | seconds=343.8, epochs=30, package=top1_fcn, weights_kb=118 |
| 4_predict_export | ĐẠT | seconds=17.4 |
| 5_qc | ĐẠT | metric=miou, test_from_predictions=0.3532, test_printed_by_train_py=0.3531989240090706, splits_disjoint=True, n_test_images=141, image=[90, 120, 3] |

Chỉ số trên tập TEST (tính lại từ predictions.npy của gói top-1 đã train đầy đủ):

```
{
 "miou": 0.3531989240090706,
 "pixel_accuracy": 0.8047934851349203,
 "per_class_iou": [
  0.8929,
  0.6322,
  0.0006,
  0.8631,
  0.5029,
  0.5346,
  0.0014,
  0.0,
  0.4559,
  0.0,
  0.0016
 ]
}
```

So với baseline cổ điển cùng split (leaderboard): `{"beats_reference": false, "val_margin": -0.168, "val_noise": 0.02515, "test_margin": -0.17283, "test_noise": 0.02476, "top1_val": 0.09823, "best_reference": "pixel_random_forest_5x5", "best_reference_val": 0.26623}`

Xem `leaderboard.md` (xếp hạng tìm kiếm), các thư mục `top*_*/` (gói PyTorch), các log `1_..5_`.
