# QA/QC: `image_segmentation/oxford_pets_trimap`

| Bước | Kết quả | Chi tiết |
|---|---|---|
| 1_design | ĐẠT | seconds=593.1, packages=['top1_deconvnet', 'top2_deeplabv3'] |
| 2_package_tests | ĐẠT |  |
| 3_train_top1 | ĐẠT | seconds=192.3, epochs=30, package=top1_deconvnet, weights_kb=106 |
| 4_predict_export | ĐẠT | seconds=12.6 |
| 5_qc | ĐẠT | metric=miou, test_from_predictions=0.5795, test_printed_by_train_py=0.5794929027296695, splits_disjoint=True, n_test_images=296, image=[96, 96, 3] |

Chỉ số trên tập TEST (tính lại từ predictions.npy của gói top-1 đã train đầy đủ):

```
{
 "miou": 0.5794929027296695,
 "pixel_accuracy": 0.8139919704861112,
 "per_class_iou": [
  0.7971,
  0.6511,
  0.2902
 ]
}
```

So với baseline cổ điển cùng split (leaderboard): `{"beats_reference": true, "val_margin": 0.22249, "val_noise": 0.0287, "test_margin": 0.21723, "test_noise": 0.02872, "top1_val": 0.57849, "best_reference": "pixel_random_forest_5x5", "best_reference_val": 0.356}`

Xem `leaderboard.md` (xếp hạng tìm kiếm), các thư mục `top*_*/` (gói PyTorch), các log `1_..5_`.
