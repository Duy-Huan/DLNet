# QA/QC: `hsi/indian_pines`

| Bước | Kết quả | Chi tiết |
|---|---|---|
| 1_design | ĐẠT | seconds=595.4, packages=['top1_zoo_gfnet_block_spectral_gf', 'top2_zoo_gfnet_block_spectral_gf_vgg_stage', 'top3_spec_fft2d'] |
| 2_package_tests | ĐẠT |  |
| 3_train_top1 | ĐẠT | seconds=32.2, epochs=30, package=top1_zoo_gfnet_block_spectral_gf, weights_kb=230 |
| 4_predict_export | ĐẠT | seconds=9.8 |
| 5_qc | ĐẠT | metric=accuracy, test_from_predictions=0.9848, test_printed_by_train_py=0.9847523786289338, splits_disjoint=True, n_test=8198, cube=[145, 145, 200], classes=16 |

Chỉ số trên tập TEST (tính lại từ predictions.npy của gói top-1 đã train đầy đủ):

```
{
 "accuracy": 0.9847523786289338,
 "balanced_accuracy": 0.9838696186666209,
 "kappa": 0.9826128610261888,
 "per_class_recall": [
  1.0,
  0.9921,
  0.9789,
  0.9683,
  0.9354,
  0.9897,
  1.0,
  1.0,
  1.0,
  0.9871,
  0.9873,
  0.9558,
  1.0,
  0.997,
  0.9903,
  0.96
 ]
}
```

So với baseline cổ điển cùng split (leaderboard): `{"beats_reference": true, "val_margin": 0.02504, "val_noise": 0.00379, "test_margin": 0.03269, "test_noise": 0.00129, "top1_val": 0.98504, "best_reference": "random_forest_spectral_spatial", "best_reference_val": 0.96}`

Xem `leaderboard.md` (xếp hạng tìm kiếm), các thư mục `top*_*/` (gói PyTorch), các log `1_..5_`.
