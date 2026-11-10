---
name: add-domain-plugin
description: Thêm một plugin loại dữ liệu mới (image, timeseries, ...) vào DLNet theo giao diện DomainPlugin.
triggers: plugin, domain, detect, prepare, loader, new data type
---
# Thêm domain plugin

Tham chiếu mẫu: `dlnet/plugins/tabular.py` (đơn giản), `dlnet/plugins/hsi.py` + `dlnet/hsi_data.py` + `dlnet/hsi_lib.py` (đầy đủ). Giao diện: `dlnet/plugins/__init__.py`.

1. Đọc `agent.md` §14.1 và skill `evaluate-honestly` (chia dữ liệu quyết định độ tin cậy của mọi kết quả).
2. Viết `dlnet/<x>_data.py` (loader, chia split, build data dict) và `dlnet/<x>_lib.py` (thư viện mô hình, chỉ phụ thuộc torch). Hai file này được **copy nguyên vào từng gói xuất ra**, nên không import gì từ phần còn lại của `dlnet`.
3. Viết `dlnet/plugins/<x>.py` kế thừa `DomainPlugin`: `name`, `model_file`, `extra_files`, `train_template`, `test_template`, `metric_keys`, `add_arguments`, `detect` (trả điểm 0–1), `prepare`, `profile`, `generate`, `build_model`, `reference`, `data_config`.
4. Đăng ký trong `_registry()`.
5. `prepare` trả data dict có `Xtr/Xva/Xte`, `ytr/yva/yte`, `y_raw`, `Xnp`, `in_dim`, `out_dim`, `task`, `primary`, `info`. Mọi bước fit (chuẩn hóa, PCA) chỉ dùng tập train.
6. `reference` phải có baseline cổ điển **cùng mức thông tin đầu vào** với mô hình sâu (ví dụ HSI: cả phổ và phổ+không gian).
7. `generate` phải bỏ cấu hình không build được (bắt `RuntimeError`/`ValueError`) và áp ngân sách tham số. Ghi rõ nếu ngân sách là heuristic.
8. Test trong `tests/test_<x>.py`: loader roundtrip, kiểm tra không rò rỉ split, mọi họ mô hình forward đúng shape, một lần chạy end-to-end nhỏ. Test nặng dùng `pytest.importorskip("torch")`.
9. Chạy `ruff check .` và `python3 -m pytest -q`, cập nhật `agent.md` §14 và `README.md`, rồi theo skill `git-dev-main`.
