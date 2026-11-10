---
name: evaluate-honestly
description: Giao thức chia dữ liệu, baseline tham chiếu và quy tắc kết luận thắng/không thắng, để không báo kết quả ảo.
triggers: split, leakage, reference, baseline, verdict, acceptance, overfit, selection bias
---
# Đánh giá trung thực

Bài học từ các lần chạy thật: hai lỗi lớn nhất là **rò rỉ dữ liệu** và **baseline tham chiếu không công bằng**.

## Chia dữ liệu
- Mọi bước fit (scaler, PCA, tiền xử lý) chỉ dùng tập train.
- Dữ liệu có cấu trúc không gian/thời gian: chia theo khối hoặc theo thời gian, bỏ các mẫu mà cửa sổ (patch, window) chạm sang split khác. Có test tự động kiểm tra điều này.
- Chia ngẫu nhiên theo pixel/mẫu chỉ là chế độ "văn liệu": báo kèm cảnh báo rằng OA có thể bị thổi phồng.
- Báo số mẫu bị loại, và các lớp vắng trong train hoặc test.

## Baseline tham chiếu
- Phải cùng mức thông tin với mô hình sâu (mô hình dùng patch thì baseline phải có đặc trưng không gian).
- Tinh chỉnh siêu tham số của baseline trên validation (ví dụ `C` của SVM).
- Tabular: logistic/ridge, RandomForest, HistGradientBoosting.

## Quy tắc chấp nhận (bảo thủ)
Top1 chỉ được tuyên bố "thắng" khi:
1. vượt baseline validation tốt nhất một biên lớn hơn max(độ lệch chuẩn theo seed, ngưỡng nhiễu nhị thức), và
2. vượt **mọi** baseline trên test.

Ngược lại: ghi "chưa thắng" và nói rõ không kết luận được hơn kém.

## Báo cáo
- Mỗi con số là một lần chạy duy nhất trừ khi nói khác. Ghi seed, budget, phần cứng.
- Validation nhỏ (ít khối) làm thứ hạng nhiễu: xem skill `fix-ranking-noise`.
- Xem `agent.md` §12.2.
