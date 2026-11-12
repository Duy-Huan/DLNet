# QA/QC: `hsi/ksc`

| Bước | Kết quả | Chi tiết |
|---|---|---|
| 1_design | ĐẠT | seconds=595.7, packages=['top1_mambahsi', 'top2_zoo_afno_block', 'top3_zoo_mbconv_res2net_block'] |
| 2_package_tests | ĐẠT |  |
| 3_train_top1 | ĐẠT | seconds=148.1, epochs=30, package=top1_mambahsi, weights_kb=107 |
| 4_predict_export | ĐẠT | seconds=762.2 |
| 5_qc | ĐẠT | metric=accuracy, test_from_predictions=0.941, test_printed_by_train_py=0.941021337808679, splits_disjoint=True, n_test=4171, cube=[512, 614, 176], classes=13 |

Chỉ số trên tập TEST (tính lại từ predictions.npy của gói top-1 đã train đầy đủ):

```
{
 "accuracy": 0.941021337808679,
 "balanced_accuracy": 0.9059914569056982,
 "kappa": 0.9342021173243147,
 "per_class_recall": [
  1.0,
  0.8103,
  0.9461,
  0.7822,
  0.7442,
  0.9672,
  0.8235,
  0.9855,
  1.0,
  0.7778,
  0.9761,
  0.9677,
  0.9973
 ]
}
```

So với baseline cổ điển cùng split (leaderboard): `{"beats_reference": false, "val_margin": -0.00385, "val_noise": 0.00823, "test_margin": -0.0211, "test_noise": 0.00333, "top1_val": 0.96346, "best_reference": "random_forest_spectral_spatial", "best_reference_val": 0.96731}`

Xem `leaderboard.md` (xếp hạng tìm kiếm), các thư mục `top*_*/` (gói PyTorch), các log `1_..5_`.
