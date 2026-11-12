# QA/QC: `image_segmentation/isbi2012_em`

| Bước | Kết quả | Chi tiết |
|---|---|---|
| 1_design | ĐẠT | seconds=540.6, packages=['top1_deeplabv3plus'] |
| 2_package_tests | ĐẠT |  |
| 3_train_top1 | ĐẠT | seconds=72.9, epochs=30, package=top1_deeplabv3plus, weights_kb=86 |
| 4_predict_export | ĐẠT | seconds=5.9 |
| 5_qc | ĐẠT | metric=miou, test_from_predictions=0.3953, test_printed_by_train_py=0.3952751159667969, splits_disjoint=True, n_test_images=6, image=[256, 256] |

Chỉ số trên tập TEST (tính lại từ predictions.npy của gói top-1 đã train đầy đủ):

```
{
 "miou": 0.3952751159667969,
 "pixel_accuracy": 0.7905502319335938,
 "per_class_iou": [
  0.0,
  0.7906
 ]
}
```

So với baseline cổ điển cùng split (leaderboard): `{"beats_reference": false, "val_margin": -0.35995, "val_noise": 0.18628, "test_margin": -0.38641, "test_noise": 0.1868, "top1_val": 0.29558, "best_reference": "pixel_random_forest_5x5", "best_reference_val": 0.65553}`

Xem `leaderboard.md` (xếp hạng tìm kiếm), các thư mục `top*_*/` (gói PyTorch), các log `1_..5_`.
