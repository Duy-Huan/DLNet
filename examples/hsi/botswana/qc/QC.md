# QA/QC: `hsi/botswana`

| Bước | Kết quả | Chi tiết |
|---|---|---|
| 1_design | ĐẠT | seconds=543.0, packages=['top1_spec_fft2d', 'top2_zoo_ffn_gated_attn_block_gfnet_block', 'top3_zoo_fca_nin_block'] |
| 2_package_tests | ĐẠT |  |
| 3_train_top1 | ĐẠT | seconds=6.4, epochs=30, package=top1_spec_fft2d, weights_kb=117 |
| 4_predict_export | ĐẠT | seconds=20.6 |
| 5_qc | ĐẠT | metric=accuracy, test_from_predictions=0.9919, test_printed_by_train_py=0.9919168591224018, splits_disjoint=True, n_test=2598, cube=[1476, 256, 145], classes=14 |

Chỉ số trên tập TEST (tính lại từ predictions.npy của gói top-1 đã train đầy đủ):

```
{
 "accuracy": 0.9919168591224018,
 "balanced_accuracy": 0.9905395307438342,
 "kappa": 0.9912420907449616,
 "per_class_recall": [
  1.0,
  1.0,
  1.0,
  1.0,
  0.9581,
  0.9721,
  1.0,
  1.0,
  1.0,
  1.0,
  1.0,
  1.0,
  0.9907,
  0.9467
 ]
}
```

So với baseline cổ điển cùng split (leaderboard): `{"beats_reference": true, "val_margin": 0.00821, "val_noise": 0.00552, "test_margin": 0.00539, "test_noise": 0.00195, "top1_val": 0.99897, "best_reference": "random_forest_spectral_spatial", "best_reference_val": 0.99077}`

Xem `leaderboard.md` (xếp hạng tìm kiếm), các thư mục `top*_*/` (gói PyTorch), các log `1_..5_`.
