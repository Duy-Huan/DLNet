---
name: audit-and-regression
description: Audit mã nguồn, biến mỗi lỗi tìm được thành regression test (đỏ trước, xanh sau) và giữ test trong CI.
triggers: audit, regression, test, bug, CI, review, flaky, coverage
---
# Audit và regression test

1. **Audit bằng cách đọc mã**, theo các nhóm rủi ro đã từng gặp ở DLNet:
   - Rò rỉ dữ liệu (fit trên toàn bộ dữ liệu, impute bằng thống kê toàn cục, patch/cửa sổ chạm split khác).
   - Ngân sách thời gian (hết giờ trước khi có kết quả, so sánh số epoch khác nhau).
   - Thao tác phá hủy (xóa thư mục `--out`).
   - Split rỗng hoặc quá nhỏ, danh sách ứng viên rỗng (lỗi IndexError khó hiểu).
   - Gói xuất ra không tự chạy được, hoặc dựng lại dữ liệu khác với lúc search.
2. **Viết test trước, chạy cho thấy đỏ**, rồi mới sửa mã, rồi chạy cho thấy xanh. Test đỏ vì lỗi của chính test (fixture sai) thì sửa test, không sửa mã.
3. Test nằm trong `tests/test_regression.py` (marker `regression`) với docstring bắt đầu bằng `Was: ...` mô tả lỗi cũ. Test chạy CLI và gói xuất ra nằm trong `tests/test_e2e.py` (marker `e2e`). Dữ liệu giả lập dùng fixture ở `tests/conftest.py`, không tải gì.
4. Ngoài lỗi, khóa luôn các quy tắc không được lệch: bảng chân trị của `verdict`, giá trị vàng (kappa, công thức TTLinear, PatchStore), tính tất định của split.
4b. **Bao phủ**: `coverage run -m pytest -q && coverage combine && coverage report` (cấu hình ở `pyproject.toml`, đếm cả lệnh CLI chạy trong tiến trình con; sàn `fail_under`, chỉ được nâng). Tìm module/nhánh chưa chạy, viết test cho **kết quả người dùng thấy** (thông báo lỗi, file đầu ra), không viết test chỉ để tăng con số. Nhánh không bao giờ chạy được thì xóa.
4c. **Test không phụ thuộc tốc độ máy**: test nào khẳng định chất lượng sau `design` phải dùng ngân sách epoch (`--budget 240ep`), không dùng giây; ngân sách giây chỉ để kiểm tra hành vi hết giờ.
5. Chạy đúng như CI: `ruff check . && ruff format --check . && python3 -m pytest -q -m "not e2e" -rs` rồi `python3 -m pytest -q -m e2e`. Test bị skip là lỗi (CI fail khi có skip).
6. Cập nhật `.github/README.md` nếu thêm job hoặc check bắt buộc, ghi phát hiện vào `agent.md` §9, rồi theo skill `git-dev-main`. Sau khi push, kiểm tra kết quả Actions.
