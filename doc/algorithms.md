# Thuật toán trong DLNet

Mỗi mục: thuật toán gì, nguồn (paper / mã chính thức), nằm ở đâu trong mã, điều gì đã hoặc chưa đối chiếu, và số đo nếu có. PDF ở `ref/` (mục lục `ref/README.md`, `ref/optimazeDL/README.md`). Số đo đầy đủ: `agent.md` §12, §16.

## 1. Tìm kiếm và lịch huấn luyện

| Thuật toán | Nguồn | Mã | Ghi chú |
|---|---|---|---|
| Successive halving (mặc định) | Jamieson & Talwalkar, arXiv:1502.07943 | `search.run_search` | rung 4 / 12 / 30 epoch, η = 3, giữ tối thiểu 2 × Top-K (DLNet, khác paper); **mỗi rung một phần ngân sách bằng nhau**, phần dư chuyển sang rung sau (Algorithm 1 của paper, từ v0.21); chỉ so ứng viên cùng số epoch, kể cả ở xếp hạng cuối (`equalize`, `final_order`, từ v0.20); ngân sách giây: ứng viên chưa train được định giá bằng một bước đo thật (`Trainer.probe_epoch_seconds`, v1.0), không bằng FLOPs |
| Hyperband (`--scheduler hyperband`) | Li et al., arXiv:1603.06560, Algorithm 1 | `search.hyperband_brackets`, `run_hyperband` | R = 36 epoch, đơn vị 4 epoch, η = 3; Table 1 của paper lệch Algorithm 1 ở s = 1..3, DLNet theo Algorithm 1 (có test) |
| Aging (regularized) evolution | Real et al., arXiv:1802.01548, Algorithm 1 | `zoo/space.tournament`, `aging_children`, vòng `[evolve]` trong `cli.py` | tournament S lấy có hoàn lại, con thêm bên phải, già nhất bị loại; chạy theo lô; DLNet thêm lai ghép xác suất 0,3; từ v0.21 chỉ chạy một thế hệ khi phần ngân sách tìm kiếm còn lại đủ huấn luyện mọi con tới 12 epoch (đo: ở 150 epoch tiến hóa thua trên test 5/5 cặp) |
| Giai đoạn nhiều seed, nhiều lần chia | DLNet | `search.stage2`, `eval_runs`, `more_runs` | độ lệch chuẩn gồm cả nhiễu do cách chia khi `--cv-splits` > 1; từ v0.21 ngân sách còn lại được dùng để chạy lại các mô hình tốt nhất (tới 3 lần) |

## 2. Hướng tối ưu (`--objective`)

| Hướng | Nguồn | Công thức |
|---|---|---|
| `params`, `flops` | phương pháp ε-ràng buộc (viết từ trí nhớ) | nhỏ nhất trong các mô hình có điểm ≥ điểm tốt nhất − `tolerance` |
| `balanced` | MnasNet, Tan et al., arXiv:1807.11626, phương trình 2 | `ACC × (COST / T)^w`, w = −0,07 (ràng buộc mềm của paper); điểm không nằm trong [0, 1] được chuẩn hóa min-max |
| `pareto` | NSGA-II (Deb et al. 2002, viết từ trí nhớ); LEMONADE arXiv:1804.09081 | sắp xếp không bị trội, rồi crowding distance trong mỗi mặt |

Cùng một hàm `objectives.order` được dùng giữa các rung, trong tournament và khi xếp Top-K. Test đáp án đóng: `tests/test_objectives.py`.

## 3. Chọn hạt giống và học từ lần chạy trước

| Thành phần | Nguồn | Mã |
|---|---|---|
| Data card ảnh (phổ công suất, độ dài tương quan, mật độ cạnh, phụ thuộc vị trí, tương quan kênh, 1-NN, probe tích chập ngẫu nhiên) | thống kê ảnh tự nhiên (Field 1987; van der Schaaf & van Hateren 1996), độ phức tạp phân loại (Ho & Basu 2002), đặc trưng tích chập ngẫu nhiên (Jarrett 2009, Saxe 2011); đều viết từ trí nhớ | `datacard.py` |
| Quy tắc đặc trưng → đặc tính kiến trúc, ngữ cảnh (`--context`) | heuristic của DLNet, mỗi quy tắc in lý do | `seeds.trait_weights`, `CONTEXT`, `OP_TRAITS` |
| Tổng hợp xếp hạng | Borda count có trọng số | `seeds.borda`; từ v0.21 prior viết tay có trọng số 0 khi đã có surrogate và không có `--context` (đo: Spearman của prior âm trên 3 bộ ảnh, và prior làm Borda kém đi: +0,54 so với +0,66) |
| k bộ dữ liệu gần nhất bỏ phiếu thứ tự họ | Feurer et al. AAAI 2015 (viết từ trí nhớ) | `meta.recommend` |
| Portfolio tham lam | Auto-sklearn 2.0, arXiv:2007.04074, Algorithm 1 | `meta.portfolio` |
| Vote đa số của proxy | Abdelfattah et al., arXiv:2101.08134 §4 (synflow, jacob_cov → naswot, snip) | `seeds.proxy_vote` (đọc thành so sánh từng cặp, chưa đối chiếu mã) |
| **Surrogate zero-shot ZAP** | Öztürk et al., ICML 2022, arXiv:2206.08476, §3.2, phương trình 3–4: `f(x, φ)` học bằng loss xếp hạng cặp `−log σ(f_j − f_k)` trên meta-dataset | `surrogate.py`; `x` = đặc tính kiến trúc + log tham số + log FLOPs + số module, `φ` = data card; MLP 2×32, 3 thành viên (không chép kích thước mạng của paper) |
| Meta-dataset có sẵn | `scripts/build_meta.py` trên 14 bộ ảnh (CIFAR-10, CIFAR-10 xám, CIFAR mèo/chó, CIFAR-100 20 lớp, hoa 5 lớp, MNIST/Fashion các cỡ, MNIST màu, digits, 2 bộ tổng hợp), 24 kiến trúc ngẫu nhiên mỗi bộ, 4 epoch | `dlnet/data/meta_image.jsonl` (tắt bằng `--no-builtin-meta`) |

**Đo leave-one-dataset-out** (`scripts/bench_surrogate.py`, 14 bộ × 24 kiến trúc): học trên 13 bộ, xếp hạng bộ còn lại.

| Bộ xếp hạng | Spearman TB | regret@1 TB | best trong top-3 |
|---|---|---|---|
| Borda(ZAP, ZAP không φ, kNN) | +0,67 | **0,061** | 4/14 |
| ZAP (có data card) | +0,54 | **0,061** | 3/14 |
| ZAP không data card | **+0,69** | 0,075 | 5/14 |
| kNN theo họ | +0,55 | 0,096 | **9/14** |
| prior (quy tắc tay) | +0,19 | 0,104 | 3/14 |
| đếm tham số | +0,33 | 0,108 | 8/14 |
| ngẫu nhiên | 0 | 0,175 | 1,8/14 |

Kết luận: học từ các lần chạy trước tốt hơn hẳn quy tắc tay và đếm tham số; nhưng với 14 bộ, thêm data card **chưa** làm Spearman tốt hơn (chỉ giảm regret@1). `design` dùng Borda của ZAP có và không có data card cho thứ tự hạt giống.

**Kết quả đo cũ của prior** (`scripts/bench_prior.py`, 20 ứng viên ngẫu nhiên, 4 epoch, MNIST/Fashion-MNIST con 3000 ảnh và hai bộ tổng hợp): điểm prior theo data card có Spearman −0,18…+0,35 với validation thật và không lần nào trúng top-3; đếm tham số đạt +0,45…+0,83 trên MNIST/Fashion. Vì vậy `--seeds` mặc định không loại ứng viên.

## 3b. AI advisor (tùy chọn)

AI CLI ngoài xếp hạng ứng viên và đề xuất spec; câu trả lời được coi là dữ liệu, kiểm tĩnh, rồi vào Borda như một bộ xếp hạng. **Chưa đo** hiệu quả trên dữ liệu thật. Chi tiết: [ai-advisor.md](ai-advisor.md).

## 4. Zero-cost proxy (chỉ đo, không loại)

| Proxy | Nguồn | Đối chiếu |
|---|---|---|
| NASWOT | Mellor et al., arXiv:2006.04647 | theo `score_networks.py` chính thức |
| SynFlow | Tanaka et al., arXiv:2006.05467 | công thức paper; dùng làm điểm cả mạng theo Abdelfattah (chưa đối chiếu mã) |
| grad_norm, SNIP | Abdelfattah et al., arXiv:2101.08134, phương trình 1 | công thức paper |
| ZiCo | Li et al., arXiv:2301.11300 | theo `compute_zico.py` chính thức (có test so công thức) |

Quy tắc (`agent.md` §16.1): proxy chỉ được dùng để loại sau khi đo Spearman trên dữ liệu của DLNet và hơn đếm tham số một cách ổn định. Đến v0.15 chưa proxy nào đạt.

## 5. Hyperparameter của từng baseline

| Thuật toán | Nguồn | Mã |
|---|---|---|
| LR range test | Smith, arXiv:1506.01186 §3.3 | `hparams.lr_range_test`; đề xuất = LR tại cực tiểu loss / 10 (thực hành của LR finder fastai cho optimizer thích nghi; trước v0.21 là /2, đề xuất cao gấp 10–20 lần LR đã tìm và 0/15 được nhận) |
| Gradient noise scale → batch | McCandlish et al., arXiv:1812.06162, phương trình 2.9, phụ lục A.1 | `hparams.gradient_noise_scale`, `suggest_batch` |
| Normalized weight decay | Loshchilov & Hutter, arXiv:1711.05101, phụ lục B.1 | `hparams.normalized_wd` |

Đề xuất chỉ được nhận khi huấn luyện lại cùng ngân sách cho validation tốt hơn quá nhiễu (độ lệch qua các lần chạy, sàn là sai số lấy mẫu của validation). `--hparams auto` (mặc định, v0.21) chỉ ước lượng cho finalist có ≥ 2 lần chạy; chi phí ước lượng được tính vào ngân sách epoch.

## 6. Ước lượng dữ liệu và chi phí

| Thành phần | Nguồn | Mã |
|---|---|---|
| Intrinsic dimension (TwoNN, MLE) | Facco 2017; Levina & Bickel 2004 (viết từ trí nhớ) | `profile.py` |
| FLOPs, bộ nhớ kích hoạt, VRAM | `torch.utils.flop_counter` trên thiết bị `meta` | `zoo/cost.py` (VRAM chưa đối chiếu GPU; FLOPs bỏ qua phép elementwise) |
