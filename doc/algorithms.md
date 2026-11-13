# Thuật toán trong DLNet

Mỗi mục: thuật toán gì, nguồn (paper / mã chính thức), nằm ở đâu trong mã, điều gì đã hoặc chưa đối chiếu, và số đo nếu có. PDF ở `ref/` (mục lục `ref/README.md`, `ref/optimazeDL/README.md`). Số đo đầy đủ: `agent.md` §12, §16.

## 1. Tìm kiếm và lịch huấn luyện

| Thuật toán | Nguồn | Mã | Ghi chú |
|---|---|---|---|
| Successive halving (mặc định) | Jamieson & Talwalkar, arXiv:1502.07943 | `search.run_search` | rung 4 / 12 / 30 epoch, η = 3, giữ tối thiểu 2 × Top-K (DLNet, khác paper); **mỗi rung một phần ngân sách bằng nhau**, phần dư chuyển sang rung sau (Algorithm 1 của paper, từ v0.21); chỉ so ứng viên cùng số epoch, kể cả ở xếp hạng cuối (`equalize`, `final_order`, từ v0.20); ngân sách giây: ứng viên chưa train được định giá bằng một bước đo thật (`Trainer.probe_epoch_seconds`, v1.0), không bằng FLOPs |
| Hyperband (`--scheduler hyperband`) | Li et al., arXiv:1603.06560, Algorithm 1 | `search.hyperband_brackets`, `run_hyperband` | R = 36 epoch, đơn vị 4 epoch, η = 3; Table 1 của paper lệch Algorithm 1 ở s = 1..3, DLNet theo Algorithm 1 (có test) |
| Xếp hạng mỗi vòng (`--rung-score val\|tse\|val+tse`, `--rungs`, v1.5) | TSE: Ru et al., NeurIPS 2021, arXiv:2006.04492, mã gốc rubinxin/TSE@5f55142 | `search._ranked`, `search.tse`, `core.Trainer.epoch_losses` | TSE = tổng loss huấn luyện trung bình của từng epoch (thấp hơn là tốt hơn); `val+tse` = tổng điểm Borda của hai thứ hạng, hòa thì xét val. Mặc định `val`. Đo (`qa/reports/v1.5.0/`): sau 1 epoch TSE tương quan với điểm epoch 6 tốt hơn val (Spearman 0,60 so với 0,44), nhưng val giữ phần đầu tốt hơn (regret khi giữ 1/3: 0,0029 so với 0,0046 ở epoch 2); đầu-cuối ở 150 epoch trên 3 bộ ảnh, `val` và `val+tse` cho cùng top-1. Không có lợi ích đo được nên không đổi mặc định |
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

## 7. v1.0: khối và họ mới, số đo tóm tắt

Nguồn và kết quả đối chiếu số học với mã gốc của từng khối: [THIRD_PARTY.md](../THIRD_PARTY.md) (5 job CI, 32 + 8 + 19 + 9 phép so; trùng từng bit hoặc lệch ≤ 8,3e-7). Số đo đầy đủ: `agent.md` §9.9.

| Miền | Họ mới | Kết quả (cùng split, cùng epoch với họ cũ) |
|---|---|---|
| Bảng | `realmlp` | nhất 1/3 bộ sklearn (breast_cancer 0,9855), chênh trong độ lệch seed |
| Ảnh | `mambavision`, `kat` | digits: 0,9420 và 0,8922, thua cnn 0,9909 |
| Chuỗi thời gian | `xlstm`, `gated_deltanet`, `timemixer`, `timexer` | xlstm và gated_deltanet ngang tcn (≥ 0,97 trên cả hai bài); timexer ở mức ngẫu nhiên trên bài cần so kênh |
| HSI | `fahm`, `mambahsi_plus`, `ssftt`, `mambahsi` | Indian Pines chia pixel, 2 seed: OA 0,9733 / 0,9732 / 0,9620 / 0,9419; resnet2d 0,9841 |
| Đồ thị, gợi ý | `gps_net`, `polynormer_net`, `sgformer_net`, `finalmlp`, `dcnv3`, `wukong` | chưa đo trên dữ liệu thật |

**Họ của v1.4, cùng giao thức** (`scripts/bench_v1.py`, 3 seed, trung bình ± độ lệch):

| Dữ liệu | Họ mới | So với họ cũ |
|---|---|---|
| Bảng | `tide_mlp` | wine 0,983 ± 0,012 (realmlp 0,983, mlp 0,966); breast_cancer 0,983 (realmlp 0,986); diabetes RMSE 60,5 (mlp 56,0): ngang trên phân loại, kém trên hồi quy |
| Chuỗi thời gian | `tsmixer`, `dlinear` | `ts_freq`: tsmixer 0,863 ± 0,131, dlinear 0,355 (tcn 0,979, itransformer 1,000); `ts_crosschannel`: cả hai ở mức ngẫu nhiên (tcn 0,992). DLinear là mô hình tuyến tính cho dự báo: trên phân loại theo tần số với pha ngẫu nhiên nó không học được, đúng như lý thuyết; giữ làm mốc so sánh |
| Ảnh | `pvt`, `inceptionnext`, `focalnet` | digits 8x8: 0,976 / 0,972 / 0,961, thua cnn 0,989 và resnet 0,988, chậm hơn cnn 5–8 lần |
| Đồ thị, gợi ý | `gatv2_net`, `gcnii_net`, `appnp_net`, `sgc_net`, `xdeepfm`, `autoint`, `fibinet` | chưa đo trên dữ liệu thật |

Chưa họ mới nào thắng họ cũ ở ngân sách nhỏ trên CPU; các họ có vòng lặp Python (Mamba, xLSTM, Gated DeltaNet) chậm hơn CNN hàng chục tới hàng trăm lần.

**Mã của mạng xuất ra** (`dlnet/standalone.py`): `model.py` trong mỗi gói là nguồn PyTorch thuần của chính mạng, sinh bằng cách ghi lại đối số constructor khi dựng mạng rồi gom nguồn các lớp được dùng; kiểm bằng cách so trọng số cùng seed và đầu ra với mạng gốc (xem [architecture.md](architecture.md)).
