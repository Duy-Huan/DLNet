# ref/optimazeDL — thuật toán chuẩn để ước lượng chất lượng mạng & hyperparameter

Gồm PDF (arXiv, đã đối chiếu tiêu đề với trang đầu PDF) và mã chính thức của tác giả (`official_code/`, chỉ những repo có giấy phép cho phép sao chép; còn lại ghi trong `OFFICIAL_CODE.md` kèm commit để tải lại).

**Quy tắc (agent.md §16):** muốn thêm/sửa bất kỳ bộ ước lượng chất lượng hay hyperparameter nào, đọc paper và mã chính thức ở đây trước, implement đúng công thức, ghi rõ chỗ nào đã đối chiếu và chỗ nào lấy từ trí nhớ. Không tự chế công thức.

Cột "Trạng thái": **đã implement** = có trong `dlnet/` và có test với đáp án giải tích; **chưa** = mới có tài liệu.

## A. Ước lượng chất lượng mạng không cần huấn luyện (zero-cost proxy)

| File PDF | Thuật toán | Ghi chú | Mã chính thức | Trạng thái |
|---|---|---|---|---|
| `../03_synflow-pruning-without-data_2006.05467.pdf` | SynFlow | điểm theo luồng synaptic, không cần dữ liệu | Synaptic-Flow (chỉ index) | chưa |
| `../03_naswot-zero-cost_2006.04647.pdf` | NASWOT | logdet của kernel mã kích hoạt ReLU | `official_code/nas-without-training/` | chưa |
| `../03_zero-cost-proxies-lightweight-nas_2101.08134.pdf` | Zero-cost proxies (grad_norm, snip, grasp, fisher, synflow, jacob_cov) | bản so sánh gốc | zero-cost-nas (Samsung; không truy cập được) | chưa |
| `zc_snip-connection-sensitivity_1810.02340.pdf` | SNIP | độ nhạy kết nối tại khởi tạo | Synaptic-Flow/Pruners (index) | chưa |
| `zc_grasp-gradient-signal-preservation_2002.07376.pdf` | GraSP | bảo toàn tín hiệu gradient | Synaptic-Flow/Pruners (index) | chưa |
| `zc_zico-gradient-inverse-cv_2301.11300.pdf` | ZiCo | nghịch đảo hệ số biến thiên của gradient | `official_code/ZiCo/` | chưa |
| `zc_zen-nas_2102.01063.pdf` | Zen-Score | độ nhạy với nhiễu đầu vào + BN scaling (CNN) | `official_code/ZenNAS/` | chưa |
| `zc_te-nas-ntk-linear-regions_2102.11535.pdf` | TE-NAS | NTK condition number + số vùng tuyến tính | `official_code/TENAS/` | chưa |
| `zc_gradsign_2110.08616.pdf` | GradSign | độ đồng nhất dấu gradient theo mẫu | (chưa tìm repo) | chưa |

## B. Chất lượng trọng số sau huấn luyện (không cần dữ liệu test)

| File PDF | Thuật toán | Trạng thái |
|---|---|---|
| `wq_weightwatcher-heavy-tailed-selfreg_1810.01075.pdf` | Heavy-tailed self-regularization (cơ sở của WeightWatcher) | chưa |
| `wq_weightwatcher-predicting-quality-trends_2002.06716.pdf` | Dự đoán xu hướng chất lượng từ phổ trọng số (alpha) | chưa |

Mã: `CalculatedContent/WeightWatcher` (Apache-2.0, file lớn nên chỉ index).

## C. Tìm kiếm / ước lượng hyperparameter

| File PDF | Thuật toán | Dùng ở đâu | Trạng thái |
|---|---|---|---|
| `hp_lr-range-test-cyclical-lr_1506.01186.pdf` | LR range test (Smith) | `dlnet/hparams.py: lr_range_test` | **đã implement** |
| `hp_critical-batch-size_1812.06162.pdf` | Gradient noise scale / critical batch size (McCandlish) | `hparams.gradient_noise_scale`, `noise_scale_from_norms`, `suggest_batch` | **đã implement** |
| `hp_adamw-decoupled-weight-decay_1711.05101.pdf` | AdamW + normalized weight decay | `hparams.normalized_wd` (AdamW đã dùng ở `core.Trainer`) | **đã implement** |
| `hp_successive-halving-jamieson_1502.07943.pdf` | Successive halving | `dlnet/search.py` (rung 4/12/30, eta=3) | đã implement (bản đơn giản) |
| `hp_hyperband_1603.06560.pdf` | Hyperband | chưa; successive halving hiện chạy một bracket | chưa |
| `hp_asha_1810.05934.pdf` | ASHA (halving bất đồng bộ) | chưa | chưa |
| `hp_bohb_1807.01774.pdf` | BOHB (Bayes + Hyperband) | chưa | chưa |
| `hp_optuna_1907.10902.pdf` | Optuna (TPE, pruning) | chưa | chưa |
| `hp_pbt_1711.09846.pdf` | Population Based Training | chưa | chưa |
| `hp_freeze-thaw-bo_1406.3896.pdf` | Freeze-thaw BO (ngoại suy learning curve) | chưa | chưa |
| `hp_super-convergence_1708.07120.pdf` | One-cycle / super-convergence (dùng LR range test) | chưa | chưa |
| `hp_dont-decay-lr-increase-batch_1711.00489.pdf` | Tăng batch thay vì giảm LR | chưa | chưa |

## D. Benchmark để kiểm chứng bộ ước lượng

| File PDF | Nội dung |
|---|---|
| `bm_nas-bench-101_1902.09635.pdf` | NAS-Bench-101 |
| `bm_nas-bench-201_2001.00326.pdf` | NAS-Bench-201 (nơi các proxy ở nhóm A được đo tương quan hạng) |

Bất kỳ proxy nào ở nhóm A trước khi đưa vào xếp hạng phải được đo tương quan hạng (Spearman) với kết quả huấn luyện thật trên dữ liệu của DLNet, và so với baseline "đếm tham số". Không có phép đo đó thì không được dùng để loại ứng viên.

## Đã thử và bỏ

- TransRate, LogME (chọn mô hình tiền huấn luyện để chuyển giao): không phù hợp với bài toán huấn luyện từ đầu của DLNet. Id arXiv 2106.09427 thuộc bài khác (quant-ph), không dùng.
