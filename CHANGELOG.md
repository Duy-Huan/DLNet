# Changelog

Định dạng theo [Keep a Changelog](https://keepachangelog.com/vi/1.1.0/), phiên bản theo [SemVer](https://semver.org/lang/vi/) (trước 1.0: tăng minor khi thêm tính năng hoặc đổi hành vi, tăng patch khi sửa lỗi).

Quy tắc: mọi thay đổi người dùng nhìn thấy được phải thêm một dòng vào `[Unreleased]` trong cùng commit. Khi phát hành, đổi `[Unreleased]` thành `[x.y.z] - YYYY-MM-DD`, tăng version trong `pyproject.toml` và `dlnet/__init__.py`. Xem `agent.md` §11 và `skills/release/SKILL.md`.

## [Unreleased]

## [0.8.0] - 2026-11-10

Mốc M9 (xem `agent.md` §8, §14.3): plugin ảnh, chuỗi thời gian và recommender meta-learning.

### Thêm
- Plugin `image`: `dlnet design --data images.npz` với `X` dạng (N, H, W) hoặc (N, H, W, C) và `y`. Họ mô hình: `cnn`, `resnet`, `vit`.
- Plugin `timeseries`: `dlnet design --data series.npz` với `X` dạng (N, T) hoặc (N, T, C) và `y`. Họ mô hình: `tcn`, `gru`, `ts_transformer`.
- Cả hai hỗ trợ phân loại và hồi quy (`--task`), tên mảng khác (`--x-key`, `--y-key`), xuất gói `top<i>_*/` chạy độc lập như các plugin khác. Router tự nhận loại dữ liệu theo số chiều của `X`.
- `--meta-store file.jsonl`: recommender meta-learning. Mỗi lần chạy được ghi vào kho; các lần chạy trước trên dataset tương tự bỏ phiếu thứ tự họ mô hình (ghi trong `leaderboard.json`, khóa `meta_recommendation`). Chỉ sắp thứ tự, không loại họ nào.
- `scripts/bench_meta.py`: đo recommender bằng leave-one-dataset-out trên 12 dataset bảng.
- Skill mới: `deep-unfolding`, `meta-recommender`.

### Hạn chế đã biết
- Recommender **chưa đạt** tiêu chí của M9: họ tốt nhất nằm trong top-3 gợi ý ở 7/12 dataset (58%; ngẫu nhiên 51%; mục tiêu ≥80%). Kho 12 dataset quá nhỏ và nhãn lấy từ 4 epoch đầu nên nhiễu.
- Plugin ảnh và chuỗi thời gian mới chạy trên dữ liệu tổng hợp nhỏ; chưa có số liệu trên dữ liệu thật. Các họ mô hình viết từ trí nhớ, chưa đối chiếu mã tác giả. Chưa có Mamba/SSM và ConvNeXt cho ảnh. Không có augmentation.
- Mảng `(N, S, S)` được hiểu là ảnh xám vuông; chuỗi thời gian có T = C cần `--plugin timeseries`.

## [0.7.0] - 2026-11-10

Mốc M8 (xem `agent.md` §5, §8): deep unfolding cho bài toán nghịch có mô hình thuận.

### Thêm
- Lệnh mới `dlnet unfold --data problem.npz [--layers 4,8,16] [--epochs 100] [--seeds 3] [--shared]`: với bài toán `y = A x + nhiễu` (file `.npz` chứa `A`, `Y`, `X`), huấn luyện mạng unfolding LISTA cho từng độ sâu K, so với ISTA và FISTA chạy đúng K vòng (λ tune trên validation), ghi `leaderboard.md` / `leaderboard.json` và gói `top<i>_lista_K<k>/` (`model.py`, `weights.pt`, `train.py`, `config.yaml`, test).
- `dlnet/unfold.py`: `ista`, `fista`, `LISTA` (dạng weight-coupled của Chen et al. 2018), `nmse_db`, `tune_lambda`. Mạng chưa huấn luyện bằng đúng K vòng ISTA.
- `scripts/bench_unfold.py`: sinh bài toán khôi phục thưa tổng hợp và chạy phép so sánh.

### Hạn chế đã biết
- Đo trên một bài toán tổng hợp (m=64, n=128, SNR 30 dB): LISTA hơn solver cổ điển tốt nhất 3,3–6,3 dB ở cùng K = 4, 8, 16; lợi thế giảm khi K tăng. Chưa đo trên dữ liệu thật và chưa so với solver chạy đến hội tụ.
- Chỉ hỗ trợ `A` là ma trận tường minh và nghiệm thưa. Chưa có ADMM/AMP, support selection, lịch huấn luyện theo tầng của paper.

## [0.6.0] - 2026-11-10

Mốc M7 (xem `agent.md` §4, §8): module tensor và phép đo nén.

### Thêm
- Họ mô hình `tr_resmlp` cho dữ liệu bảng: ResMLP có lớp ẩn dạng Tensor-Ring (`TRLinear`). `design` nay sinh ứng viên từ 6 họ.
- `dlnet/tensor_lora.py`: `TensorLoRALinear` (lớp tuyến tính đóng băng cộng delta huấn luyện được, format `lora`, `tt`, `tr`, `kronecker`), `apply_tensor_lora` / `merge_tensor_lora` cho cả mô hình, `tt_svd` để phân rã trọng số có sẵn. Delta bằng 0 lúc khởi tạo.
- `dlnet/tensor.py`: HOSVD cắt cụt (`hosvd`, `tucker_to_tensor`), `multilinear_rank`, cận sai số và `compress_report` cho tensor dữ liệu.
- `scripts/bench_tensor.py`: đo nén tham số, Tensor-LoRA và HOSVD trên dữ liệu digits của scikit-learn.

### Hạn chế đã biết
- Đo trên digits (3 seed): lớp TT/TR giảm 81–89% tham số, accuracy thay đổi trong mức nhiễu seed (≤0,4 điểm). Low-rank thường cho kết quả tương đương, chưa có bằng chứng TT/TR hơn.
- Tensor-LoRA là module thư viện, chưa được `dlnet design` dùng (chưa có backbone pretrained). Trong phép đo nhỏ, adapter kém fine-tune toàn bộ 9–30 điểm và pretrain không hơn huấn luyện từ đầu.
- HOSVD chưa dùng làm tiền xử lý HSI (nhân tử không gian cần cả pixel test).

## [0.5.0] - 2026-11-10

Mốc M6 (xem `agent.md` §8, §16.4): ước lượng intrinsic dimension có đo độ ổn định.

### Thêm
- `profile.json` có khối `intrinsic_dim_estimators`: hai bộ ước lượng (TwoNN và MLE của Levina–Bickel) chạy trên 5 mẫu con, kèm mean ± std, mức lệch giữa hai bộ và cờ `stable`. Có cho cả plugin tabular và HSI.
- `profile.json` có `capacity_hint` (độ rộng tối thiểu gợi ý, số mũ scaling ≈ 4/d). Chỉ là thông tin, không loại ứng viên.
- Plugin HSI ghi `multilinear_rank_levels`: rank Tucker ở mức năng lượng 0,90 / 0,95 / 0,99.
- `scripts/bench_estimators.py`: đo các bộ ước lượng trên 5 bộ dữ liệu biết trước số chiều và 5 bộ dữ liệu thật của scikit-learn.

### Hạn chế đã biết
- Ước lượng ổn định qua mẫu con nhưng thiên thấp khi số chiều lớn (d=20 ra 13–16 với 1000 điểm). Trên dữ liệu thật hai bộ lệch nhau 21–34%.
- Hai bộ ước lượng viết từ trí nhớ, chưa đối chiếu paper gốc. `capacity_hint` chưa được kiểm chứng là dự đoán đúng độ rộng tốt nhất.

## [0.4.0] - 2026-11-10

Mốc M5 (xem `agent.md` §8, §16.3): zero-cost proxy có đo tương quan, cache để chạy tiếp, phép đo tiết kiệm thời gian.

### Thêm
- `dlnet/proxies.py`: zero-cost proxy NASWOT (theo mã chính thức), SynFlow, grad_norm và baseline đếm tham số, có test với đáp án giải tích. Mỗi lần chạy `design` đo tương quan hạng Spearman của từng proxy với điểm validation thật sau rung 1 và ghi vào `leaderboard.json` (`zero_cost_proxies`) và `leaderboard.md`. Proxy **không** được dùng để loại ứng viên. Tắt bằng `--no-proxies`.
- Cache để chạy tiếp: các lần huấn luyện đã xong ở giai đoạn nhiều seed và bước kiểm chứng hyperparameter được lưu trong `<out>/.dlnet_cache.json`; chạy lại đúng lệnh cũ thì dùng lại kết quả thay vì huấn luyện lại. Tắt bằng `--no-cache`.
- `scripts/bench_search.py`: đo thời gian successive halving so với huấn luyện đầy đủ mọi ứng viên và tương quan của proxy với kết quả huấn luyện đầy đủ.

### Đổi
- `--out` trỏ vào thư mục lạ nay bị từ chối ngay khi bắt đầu, không phải sau khi đã dùng hết ngân sách.

### Hạn chế đã biết
- Đo trên dữ liệu tổng hợp (2 seed mỗi loại, 16 epoch): successive halving giảm 50–70% thời gian; chọn đúng top-1 của huấn luyện đầy đủ ở dữ liệu bảng (2/2) nhưng không ở HSI (0/2, điểm validation của top-1 thấp hơn 0,6–2,5 điểm, cùng cỡ với nhiễu của tập validation 160 mẫu). Chưa đo trên dữ liệu thật.
- Không proxy nào hơn đếm tham số một cách ổn định (|Spearman| ≤ 0,6, SynFlow còn tương quan âm trên dữ liệu bảng), nên chưa dùng proxy để loại ứng viên. NASWOT không tính được cho mạng không có ReLU (mọi họ tabular).
- Cache chưa bao gồm successive halving: chạy lại vẫn lặp lại giai đoạn này. Chưa có cost model FLOPs/VRAM.

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

[Unreleased]: https://github.com/Duy-Huan/DLNet/compare/v0.8.0...HEAD
[0.8.0]: https://github.com/Duy-Huan/DLNet/releases/tag/v0.8.0
[0.7.0]: https://github.com/Duy-Huan/DLNet/releases/tag/v0.7.0
[0.6.0]: https://github.com/Duy-Huan/DLNet/releases/tag/v0.6.0
[0.5.0]: https://github.com/Duy-Huan/DLNet/releases/tag/v0.5.0
[0.4.0]: https://github.com/Duy-Huan/DLNet/releases/tag/v0.4.0
[0.3.0]: https://github.com/Duy-Huan/DLNet/releases/tag/v0.3.0
[0.2.0]: https://github.com/Duy-Huan/DLNet/releases/tag/v0.2.0
[0.1.0]: https://github.com/Duy-Huan/DLNet/releases/tag/v0.1.0
