---
name: implement-standard-algorithm
description: Thêm hoặc sửa bộ ước lượng chất lượng mạng / hyperparameter theo thuật toán chuẩn trong ref/optimazeDL, không tự chế công thức.
triggers: zero-cost proxy, synflow, snip, grasp, zico, naswot, hyperparameter, learning rate, batch size, weight decay, hyperband, estimator
---
# Implement theo thuật toán chuẩn (agent.md §16)

1. Mở `ref/optimazeDL/README.md`, tìm thuật toán và file PDF. Chưa có thì dùng skill `find-verify-papers` tải vào đó (nhóm `zc_`, `wq_`, `hp_`, `bm_`).
2. Đọc phần phương pháp của paper (`pdftotext -layout file.pdf -`) và mã chính thức (`ref/optimazeDL/OFFICIAL_CODE.md`, thư mục `official_code/` hoặc tải lại theo commit ghim). Ghi lại công thức, siêu tham số mặc định của tác giả, và chỗ mã khác paper.
3. Implement trong `dlnet/` đúng công thức. Docstring nêu: paper, arXiv id, số phương trình, và dòng "chưa đối chiếu" cho mọi chi tiết lấy từ trí nhớ hoặc thói quen (ví dụ làm mượt EMA).
4. Test với đáp án giải tích (dựng đầu vào mà kết quả biết trước, như `noise_scale_from_norms` phải trả đúng B_simple) trong `tests/test_regression.py`, thêm test tính xác định theo seed.
5. Ước lượng chỉ là đề xuất: kiểm chứng bằng huấn luyện cùng ngân sách và seed, chỉ chấp nhận khi tốt hơn quá nhiễu seed (`hparams.refine` là mẫu).
6. Zero-cost proxy: trước khi loại ứng viên phải đo Spearman với kết quả huấn luyện thật trên dữ liệu của DLNet và so với "đếm tham số". Nhiều proxy chỉ áp dụng cho CNN (Zen-Score cần BN/Conv), kiểm điều kiện áp dụng trước.
7. Cập nhật cột "Trạng thái" trong `ref/optimazeDL/README.md`, `CHANGELOG.md`, rồi theo skill `git-dev-main`.

Đã có: `dlnet/hparams.py` (LR range test, gradient noise scale, normalized weight decay) và `dlnet/proxies.py` (NASWOT, SynFlow, grad_norm; chỉ đo, không loại ứng viên). Đo lại bằng `python scripts/bench_search.py --kind tabular|hsi`; kết quả hiện tại ở `agent.md` §16.3. Chỉ bật việc loại ứng viên bằng proxy khi phép đo trên dữ liệu thật cho thấy proxy hơn đếm tham số rõ rệt và ổn định qua seed.
