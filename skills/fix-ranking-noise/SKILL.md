---
name: fix-ranking-noise
description: Làm thứ hạng Top-K ổn định khi validation nhỏ hoặc nhiễu (bước 1 đã cài đặt: `--cv-splits`; bước 2, 3 còn là kế hoạch).
triggers: ranking, noise, cross-validation, seeds, block, unstable, top-k
---
# Thứ hạng nhiễu

Hiện trạng đã biết (`agent.md` §9 mục 7, 8): chia theo khối cho validation chỉ vài khối, nên top1 theo val có thể kém hạng 3 trên test (Pavia: 0.859 so với 0.886). Độ lệch chuẩn theo seed không phản ánh nhiễu này.

Đã có (v0.2.0): `--cv-splits K` (`dlnet/cli.py` tạo K lần chia, `search.eval_runs` chạy lần i với seed i trên lần chia i, `export.mean_reference` lấy trung bình baseline). Khi sửa phần này, giữ test `test_finalists_are_ranked_on_several_splits`.

Kế hoạch còn lại:
1. Với `block`: lặp chọn lại phân khối (nhiều `seed` cho phép chia), huấn luyện ứng viên còn lại ở vòng cuối trên K phân khối khác nhau, xếp hạng theo trung bình và độ lệch.
2. Chọn khối cân bằng lớp để mọi lớp có mặt trong train và test (hiện chỉ thử tối đa 300 lần ngẫu nhiên).
3. Dùng ngưỡng nhiễu theo số mẫu validation và số phân khối, thay cho độ lệch chuẩn theo seed. Nhóm các ứng viên không phân biệt được thành "indistinguishable group" (đã có trong export).
4. Test: tạo bài toán nhỏ có thứ hạng đã biết, kiểm tra thứ hạng ổn định giữa các seed; kiểm tra không rò rỉ giữa các fold.
5. Cập nhật `agent.md` §9, §12.3 và chi phí thời gian (K lần huấn luyện làm tăng thời gian: tính lại budget).
