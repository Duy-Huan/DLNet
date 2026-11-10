---
name: deep-unfolding
description: Thêm hoặc sửa mạng deep unfolding (LISTA, ...) cho bài toán nghịch có mô hình thuận, và so sánh trung thực với thuật toán lặp gốc.
triggers: unfolding, unrolling, LISTA, ISTA, FISTA, ADMM, inverse problem, forward model, sparse recovery
---
# Deep unfolding (agent.md §5)

1. Mã: `dlnet/unfold.py` (solver và mạng, chỉ phụ thuộc torch, được copy vào gói xuất ra làm `model.py`), `dlnet/unfold_run.py` (lệnh `dlnet unfold`, leaderboard, gói).
2. Theo skill `implement-standard-algorithm`: đọc paper trong `ref/05_*`, ghi phương trình trong docstring, đánh dấu phần "chưa đối chiếu".
3. Bất biến bắt buộc có test: mạng **chưa huấn luyện** phải bằng đúng K vòng của thuật toán lặp gốc (khởi tạo từ tham số của thuật toán). Nhờ đó mọi cải thiện là do học.
4. So sánh công bằng: thuật toán lặp chạy **cùng số vòng K**, siêu tham số (λ) tune trên validation; kết luận trên test; "thắng" khi mean + std của mạng thấp hơn solver cổ điển tốt nhất.
5. Đo: `python scripts/bench_unfold.py`. Ghi bảng vào `agent.md` §5 và `CHANGELOG.md`, nêu rõ dữ liệu tổng hợp hay thật.
6. Thêm solver mới (ADMM, AMP): thêm hàm lặp + lớp `nn.Module` tương ứng + test bất biến ở mục 3 + một dòng trong bảng so sánh của `unfold_run.run`.
