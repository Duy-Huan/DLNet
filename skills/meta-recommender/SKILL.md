---
name: meta-recommender
description: Dùng, đo và sửa recommender meta-learning (kho kết quả các lần chạy trước bỏ phiếu thứ tự họ mô hình).
triggers: meta-learning, meta store, recommender, warm start, family order, leave-one-out
---
# Recommender meta-learning (agent.md §14.3)

1. Mã: `dlnet/meta.py`. Bật bằng `dlnet design --meta-store <file.jsonl>`: trước khi tìm kiếm, k dataset gần nhất (cùng plugin, theo meta-feature đã chuẩn hóa) bỏ phiếu **thứ tự** họ mô hình; sau khi chạy, lần chạy này được ghi thêm vào kho.
2. Recommender chỉ **sắp thứ tự**, không loại họ nào. Muốn cho nó loại ứng viên phải có số đo trước (quy tắc §16.1 mục 4–5).
3. Đo: `python scripts/bench_meta.py` (leave-one-dataset-out, so với thứ tự toàn cục và với ngẫu nhiên). Ghi số đo vào `agent.md` §14.3 và `CHANGELOG.md`; nêu rõ số dataset.
4. Thêm meta-feature: sửa `meta.features` (lấy từ `profile.json`), giữ thứ tự cũ ở đầu danh sách; bản ghi cũ có số feature khác sẽ bị bỏ qua khi so khoảng cách.
5. Nhãn "họ tốt nhất" lấy từ điểm validation rung 1 (ít epoch), nên nhiễu. Đừng kết luận từ dưới vài chục dataset.
6. Test: `tests/test_arrays.py` (phần meta). Xong thì theo skill `git-dev-main`.
