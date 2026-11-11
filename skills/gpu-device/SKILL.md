---
name: gpu-device
description: Viết và kiểm tra mã chạy đúng trên GPU (CUDA/MPS) khi không có GPU: quy tắc tạo tensor đúng thiết bị, kiểm bằng thiết bị meta, mixed precision.
triggers: gpu, cuda, device, mps, amp, mixed precision, bf16, fp16, channels_last, meta device
---
# Mã chạy trên GPU (kể cả khi máy không có GPU)

Bối cảnh: `agent.md` §9.3. Thiết bị và độ chính xác nằm trong `dlnet/core.py` (`pick_device`, `amp_dtype`, `Trainer`), được chép vào mọi gói xuất ra.

1. **Op mới trong zoo / mô hình mới**: mọi tensor tạo trong `forward` phải theo thiết bị của đầu vào: `torch.arange(n, device=x.device)`, `torch.zeros_like(x)`, `x.new_ones(...)`, hoặc `register_buffer` trong `__init__`. Không gọi `.cuda()` hay `.to("cuda")` trong mô hình.
2. **Kiểm không cần GPU**: chạy `forward` ở chế độ train và eval trên thiết bị `meta` (`m.to("meta")(x.to("meta"))`). `meta` báo "not on the expected device" giống CUDA khi trộn tensor CPU. `tests/test_device.py::test_zoo_architectures_create_no_cpu_tensor_in_forward` làm việc này cho ngữ pháp zoo; thêm kiểu tensor mới thì thêm vào tham số của test.
3. **Vòng huấn luyện mới** (ngoài `core.Trainer`): lấy dữ liệu qua `Trainer.split` / `Trainer.batch_of`, hoặc chép batch sang `next(model.parameters()).device`; kiểm bằng `Trainer(..., device="meta").fit(1)`.
4. **Mixed precision**: loss tính trên `out.float()`; float16 cần `GradScaler` (có sẵn trong `Trainer`); bf16 thì không. Thử trên CPU bằng `--amp bf16` / `--amp fp16` (chậm, chỉ để kiểm đường mã).
5. **Không đổi đường CPU**: sau khi sửa `core.py`, so leaderboard của cùng lệnh (`--budget Nep --threads 2 --no-cache`) với bản trước: phải giống từng bit.
6. **Báo cáo trung thực**: chưa chạy trên GPU thật thì ghi rõ "chưa đo trên GPU"; không suy ra tốc độ từ CPU.
