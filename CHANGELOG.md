# Changelog

Định dạng theo [Keep a Changelog](https://keepachangelog.com/vi/1.1.0/), phiên bản theo [SemVer](https://semver.org/lang/vi/) (trước 1.0: tăng minor khi thêm tính năng hoặc đổi hành vi, tăng patch khi sửa lỗi).

Quy tắc: mọi thay đổi người dùng nhìn thấy được phải thêm một dòng vào `[Unreleased]` trong cùng commit. Khi phát hành, đổi `[Unreleased]` thành `[x.y.z] - YYYY-MM-DD`, tăng version trong `pyproject.toml` và `dlnet/__init__.py`. Xem `agent.md` §11 và `skills/release/SKILL.md`.

## [Unreleased]

## [0.3.0] - 2026-11-10

Mốc M4 (xem `agent.md` §8, §13): thư viện mở rộng và tự ghép mô hình theo grammar có kiểu.

### Thêm
- HSI: registry khối có kiểu (`hsi_lib.BLOCKS`, `@register`): mã hóa phổ (`cnn1d`, `transformer`), không gian (`conv`, `res`, `dwsep`, `convnext`, `tucker`), chung (`conv3d`, `ssrn`), hợp nhất (`concat`, `gated`). Mỗi khối tự suy ra số đặc trưng đầu ra mà không cần tạo mô hình.
- Grammar `Model := (Spectral ∥ Spatial | Spectral | Spatial | Joint) → Fusion → Head`: `plan_composed` loại cấu hình ghép sai trước khi tạo mô hình; họ `composed` lấy mẫu từ grammar.
- Năm họ preset mới là cấu hình của registry: `resnet2d`, `dwsep2d`, `convnext2d`, `spec_transformer`, `ssrn`. Tổng cộng HSI có 11 họ.

### Đổi
- Tìm kiếm duyệt ứng viên xen kẽ theo họ (round-robin). Trước đây hết ngân sách giữa rung đầu thì các họ đứng sau không bao giờ được huấn luyện.

### Hạn chế đã biết
- Các khối `res`, `convnext`, `ssrn`, `transformer` viết lại từ trí nhớ, chưa đối chiếu mã của tác giả và chưa tái hiện số liệu trong paper; không phải bản tái hiện chính thức.
- Chưa có khối Mamba/SSM, ViT/Swin, cross-attention. Registry mới chỉ có ở plugin HSI.

## [0.2.0] - 2026-11-10

Mốc M2 và M3 (xem `agent.md` §8): xếp hạng ổn định trên nhiều lần chia dữ liệu, mỗi con số có ± std.

### Thêm
- Xếp hạng trên nhiều lần chia dữ liệu: `--cv-splits K` huấn luyện các finalist trên K lần chia khác nhau (seed chia = `--seed` + i, tiền xử lý fit lại trên train của từng lần chia). HSI chia theo khối mặc định K = 3; còn lại mặc định 1. Độ lệch chuẩn và nhóm "chưa phân biệt được" nay gồm cả nhiễu do cách chia. Baseline tham chiếu được lấy trung bình trên cùng các lần chia.
- Leaderboard và `report.md` ghi số lần chạy, số lần chia và ± std của điểm test.
- Hyperparameter đi kèm mỗi baseline: `dlnet/hparams.py` ước lượng lr (LR range test, Smith), batch size (gradient noise scale, McCandlish) và weight decay (normalized, AdamW), kiểm chứng bằng huấn luyện lại và chỉ chấp nhận khi tốt hơn rõ rệt. Khối `hyperparameters` có trong `config.yaml`, `leaderboard.json`, `report.md`; `leaderboard.md` thêm cột lr, wd, batch. 20% cuối `--budget` dành cho bước này.
- `ref/optimazeDL/`: 22 paper về ước lượng chất lượng mạng và hyperparameter (SNIP, GraSP, ZiCo, Zen-NAS, TE-NAS, GradSign, WeightWatcher, Hyperband, BOHB, ASHA, Optuna, PBT, ...) cùng mã chính thức đã ghim commit; quy tắc `agent.md` §16 và skill `implement-standard-algorithm`: phải tham khảo và implement theo thuật toán chuẩn.
- Môi trường Python tái lập: `scripts/setup_env.sh` (Linux/macOS) và `scripts/setup_env.ps1` (Windows) tạo `.venv`, cài torch CPU (tự lùi về PyPI nếu chỉ mục bị chặn) cùng DLNet và công cụ dev, rồi tự kiểm tra; `.python-version` (3.12) và `requirements.txt`/`requirements-dev.txt` ghim phiên bản đã kiểm tra, để pull về chỉ cần `python -m venv .venv`, activate, `pip install -r requirements.txt`. Không commit venv vì vượt giới hạn 100 MB/file của GitHub và không mang đi máy khác được.
- Lệnh `dlnet selfcheck` (chứng minh cài đặt chạy đúng, ~20 giây) và `dlnet --version`.
- CI: job `env-setup` chạy quy trình cài đặt như người dùng (script, kích hoạt `.venv`, `dlnet selfcheck`) trên Ubuntu, macOS và Windows.
- Test chặn commit nhầm venv hoặc file lớn hơn 50 MB.

### Sửa
- Seed nay quyết định cả khởi tạo trọng số (trước đây mô hình được tạo trước `torch.manual_seed` nên cùng seed vẫn cho trọng số khác nhau giữa các lần chạy).

### Hạn chế đã biết
- Chia theo khối vẫn bỏ nhiều pixel ở mép khối và có thể thiếu lớp trong train hoặc test; nhiều lần chia chỉ làm lộ nhiễu này trong ± std, không làm nó mất đi.
- Gói xuất ra huấn luyện trên lần chia đầu tiên (`split_seed`); các lần chia khác chỉ dùng để xếp hạng.
- Ước lượng hyperparameter chưa cho thấy lợi ích trên dữ liệu thật: trong lần chạy thử, mọi đề xuất đều thua giá trị đã tìm và bị loại.

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

[Unreleased]: https://github.com/Duy-Huan/DLNet/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/Duy-Huan/DLNet/releases/tag/v0.3.0
[0.2.0]: https://github.com/Duy-Huan/DLNet/releases/tag/v0.2.0
[0.1.0]: https://github.com/Duy-Huan/DLNet/releases/tag/v0.1.0
