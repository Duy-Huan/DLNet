# Changelog

Định dạng theo [Keep a Changelog](https://keepachangelog.com/vi/1.1.0/), phiên bản theo [SemVer](https://semver.org/lang/vi/) (trước 1.0: tăng minor khi thêm tính năng hoặc đổi hành vi, tăng patch khi sửa lỗi).

Quy tắc: mọi thay đổi người dùng nhìn thấy được phải thêm một dòng vào `[Unreleased]` trong cùng commit. Khi phát hành, đổi `[Unreleased]` thành `[x.y.z] - YYYY-MM-DD`, tăng version trong `pyproject.toml` và `dlnet/__init__.py`. Xem `agent.md` §11 và `skills/release/SKILL.md`.

## [Unreleased]

## [0.1.0] - 2026-11-10

Bản đầu tiên: prototype chạy được từ dữ liệu đến Top-K baseline PyTorch cho dữ liệu bảng và ảnh siêu phổ (HSI).

### Thêm
- Lệnh `python -m dlnet design --data <file> --budget 5m --topk 3 --out dlnet_out`: profile dữ liệu, sinh ứng viên, tìm kiếm (successive halving, nhiều seed), xếp hạng đa dạng và xuất Top-K gói baseline (`model.py`, `train.py`, `config.yaml`, `tests/`, `report.md`) cùng `leaderboard.md/json`.
- Kiến trúc plugin theo loại dữ liệu (`DomainPlugin`) với router tự nhận dạng: plugin `tabular` (MLP, ResMLP, Tensor-Train ResMLP, low-rank MLP, FT-Transformer) và plugin `hsi` (CNN 1D/2D/3D, Tucker-2 CNN, HybridSN-style).
- HSI: đọc `.mat/.npz/.npy`, chia `pixel` và `block` (chống rò rỉ patch), PCA stem fit trên train, ước lượng multilinear rank (HOSVD), baseline SVM/RF với đặc trưng phổ và phổ cộng không gian.
- Ước lượng từ dữ liệu: intrinsic dimension (TwoNN), số mũ scaling α≈4/d, năng lượng PCA.
- Quy tắc chấp nhận bảo thủ: top1 chỉ "thắng" khi vượt baseline trên validation (hơn nhiễu) và vượt mọi baseline trên test.
- `agent.md` (kiến trúc, lộ trình, giao thức đánh giá, quy trình Git và CI/CD), `skills/` (9 quy trình đã đúc kết) và quy tắc chọn skill trước khi làm, `ref/` (39 paper).
- CI/CD: `ci.yml` (lint, test py3.10–3.12, regression, e2e), `promote.yml` (CI xanh thì tự đưa `dev` lên `main`), `release.yml` (tag và GitHub Release theo version).
- Bộ test: unit, regression (khóa các lỗi tìm được khi audit) và e2e (CLI thật rồi chạy gói đã xuất).

### Sửa
- Hết ngân sách thời gian trước rung đầu không còn làm sập tìm kiếm; ứng viên bị bỏ lại khi hết giờ không còn bị xếp hạng cùng ứng viên train nhiều epoch hơn.
- `--out` không còn xóa cả thư mục: chỉ thay thế output cũ của DLNet, từ chối thư mục lạ.
- Điền giá trị thiếu dùng median của tập train (trước đây dùng toàn bộ dữ liệu, rò rỉ test); dòng thiếu nhãn bị loại.
- Dữ liệu quá nhỏ, split rỗng hoặc không có ứng viên nào cho thông báo lỗi rõ ràng thay vì lỗi khó hiểu; lớp có từ 3 mẫu luôn có mặt trong mọi split.

### Hạn chế đã biết
- Xếp hạng khi chia theo khối còn nhiễu (validation chỉ vài khối); chưa có cross-validation theo khối.
- Ngân sách thời gian chỉ được kiểm tra giữa các ứng viên nên có thể bị vượt.
- Chưa có zero-cost proxy, cost model FLOPs/VRAM, Tensor-LoRA, deep unfolding; chưa có plugin image và timeseries.
- Kết quả trên Indian Pines và Pavia University là một lần chạy ngắn trên CPU: với chia theo khối, mạng sâu chưa vượt rõ baseline cổ điển.

[Unreleased]: https://github.com/Duy-Huan/DLNet/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Duy-Huan/DLNet/releases/tag/v0.1.0
