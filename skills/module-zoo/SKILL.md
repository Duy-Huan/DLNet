---
name: module-zoo
description: Thêm module vào kho dlnet/zoo, mở rộng ngữ pháp hoặc phép đột biến, đối chiếu module với mã gốc, và chạy tìm kiếm kiến trúc.
triggers: zoo, module, op, grammar, mutation, crossover, evolve, architecture search, spec, torchvision, backbone, block
---
# Kho module và tìm kiếm kiến trúc (agent.md §13.6)

1. **Thêm module**: viết hàm trong `dlnet/zoo/ops_*.py` với `@op(name, kinds, kout=None, space={...}, tags={...}, source="repo + lớp gốc")`. Hàm nhận `(dims, kind, **params)` và trả `nn.Module` một tensor vào, một tensor ra; `ValueError` khi tham số không hợp lệ với shape. Mọi tham số phải có trong `space` (giá trị đầu là mặc định). Thẻ `block`, `down`, `attn`, `stem`, `adapt`, `norm`, `act` quyết định module được lấy mẫu, chèn, hoán đổi ở đâu: chọn thẻ đúng thì ngữ pháp tự dùng.
2. **Khối từ repo ngoài**: tải repo vào thư mục scratchpad, đọc mã gốc, ghi commit và lớp vào `THIRD_PARTY.md` (cả phần **chưa** port và lý do). Giấy phép không xét (công cụ nội bộ). Viết lại cho tổng quát theo kích thước và số chiều, không phụ thuộc gói ngoài.
3. **Đối chiếu số học**: cài tạm repo gốc rồi thêm khối vào `scripts/verify_numeric.py` (chép trọng số theo tên hoặc thứ tự, so đầu ra eval). Cái này mạnh hơn so số tham số; làm cả hai nếu được.
3a. **Thư viện ảnh chính thống** (timm, smp, MONAI, torchvision): cài tạm `pip install --no-deps --target <scratch>/vlibs timm==1.0.31 segmentation-models-pytorch==0.6.0 monai==1.6.1 torchvision==0.29.2 safetensors huggingface_hub pyyaml`, rồi thêm khối vào `scripts/verify_vision_libs.py` và chạy `PYTHONPATH=<scratch>/vlibs python scripts/verify_vision_libs.py`. Định nghĩa lớp con theo **đúng thứ tự** của bản gốc để chép trọng số theo thứ tự; decoder nhận danh sách đặc trưng `[input, f0, f1, ...]` như smp.
3b. **Đối chiếu cấu trúc**: nếu repo cài được tạm (`pip install --no-deps --target /tmp/x`), thêm khối vào `scripts/verify_zoo.py` (so số tham số và shape với khối gốc cùng cấu hình) rồi ghim số vào `tests/test_zoo.py::GOLDEN`. Không đối chiếu được thì ghi "viết từ trí nhớ" trong `source`.
4. **Test**: không cần viết test riêng cho forward/backward và tham số: `tests/test_zoo.py` tự quét mọi op trên mọi kiểu đã khai báo và mọi giá trị trong `space`. Chỉ thêm test đáp án đóng cho công thức riêng (xem `test_mamba_selective_scan...`).
5. **Ngữ pháp**: sửa `dlnet/zoo/space.py` (`sample`, `MUTATIONS`, `crossover`) chỉ khi cần dạng cấu trúc mới; mỗi phép phải qua `core.check`. Preset mới thêm vào `dlnet/zoo/presets.py` (phải vừa mọi kích thước đầu vào được cấp: test `test_every_preset_fits_every_input_shape_it_is_offered`).
6. **Kiểu tensor mới** (đồ thị, id): thêm vào `KINDS` và `ND` trong `core.py`, cầu nối từ/đến các kiểu cũ, và `zoo_input` của plugin tương ứng.
7. **Chạy tìm kiếm**: `python -m dlnet design ... --zoo 12 --evolve 2 --evolve-pop 8`; `--no-zoo-presets` để chỉ lấy mẫu ngẫu nhiên; `python -m dlnet zoo` liệt kê module. So zoo với họ cũ bằng `python scripts/bench_zoo.py` rồi ghi số vào `agent.md` §13.6.
8. Cập nhật `CHANGELOG.md` rồi theo skill `release`.
9. **Cost model**: `python -m dlnet design ... --max-flops 50M --max-vram 4000`; `dlnet/zoo/cost.py` đo trên thiết bị `meta`. Op mới có phép tính không phải ma trận/tích chập (quét, elementwise nặng) sẽ bị đếm thiếu FLOPs: ghi vào docstring của op.
