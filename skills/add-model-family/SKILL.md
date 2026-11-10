---
name: add-model-family
description: Thêm họ mô hình hoặc module (kể cả tensor decomposition) vào thư viện của một plugin.
triggers: model, family, library, module, tensor, Tucker, TT, low-rank, SPACE
---
# Thêm họ mô hình

1. Xác định module hiện thực **trục nào** của paper 2302.11529 (computation / routing / aggregation / training) và ghi vào docstring. Nếu dựa vào paper khác, kiểm tra trong `ref/` và ghi tên.
2. Thêm một công thức (hàm `cấu hình → spec`) vào `dlnet/zoo/families.py` và tên họ vào danh sách của miền; module mới đi vào kho (`dlnet/zoo/ops_*.py`, skill `module-zoo`). Các file `library.py`/`array_lib.py`/`hsi_lib.py`/`blocks.py` đã bị gỡ.
3. Khi dựng lại mạng nổi tiếng từ trí nhớ, ghi rõ "chưa đối chiếu code gốc" trong docstring. Không gọi là bản tái hiện chính thức.
4. Thêm không gian tìm kiếm vào `SPACE` của plugin (rộng vừa phải, giá trị rời rạc). Hyperparameter train nằm ở `TRAIN`.
5. Với module tensor: hai nhánh cần so sánh công bằng với bản dense (cùng width, ít tham số hơn). Biến thể đã có: `TTLinear`, `LowRankLinear`, `TuckerConv`.
6. Cost model v0 là đếm tham số của mô hình đã khởi tạo (`count_params`). Phải bỏ các cấu hình lỗi shape thay vì làm sập vòng sinh.
7. Test: forward đúng shape với batch 2, backward chạy được, số tham số của biến thể nén nhỏ hơn bản dense.
8. HSI: ưu tiên thêm **khối** vào registry thay vì viết mô hình riêng: `@register(kind, name)` trên lớp con của `Block` trong `hsi_lib.py`, hiện thực `features(...)` (suy luận shape tĩnh, `raise ValueError` khi đầu vào không hợp lệ). Khối tự xuất hiện trong họ `composed`; muốn có họ preset thì thêm một dòng vào `PRESETS` và `SPACE`. Test `test_static_shape_inference_matches_the_built_blocks` tự kiểm tra khối mới.
9. Cập nhật `agent.md` §13 (registry / danh mục khối) và theo skill `git-dev-main`.
10. Khối dùng chung cho nhiều plugin (backbone, decoder, attention, routing) đặt trong `dlnet/blocks.py` với decorator `@catalog(kind, name, source)`; plugin copy `blocks.py` vào gói qua `extra_files`, file thư viện import bằng `try: from . import blocks / except ImportError: import blocks`.
11. Lấy khối từ repo ngoài: kiểm tra giấy phép trước. Không có giấy phép thì **không sao chép mã**, chỉ cài lại từ phương pháp. Ghi repo, commit, giấy phép và những gì đã đổi vào `THIRD_PARTY.md`.
12. Module mới cho nhiều plugin nay đi vào kho `dlnet/zoo` (skill `module-zoo`), không còn thêm vào từng `*_lib.py`; họ cũ trong `library.py`/`array_lib.py`/`hsi_lib.py` chỉ giữ để tương thích.
