# Changelog

Định dạng theo [Keep a Changelog](https://keepachangelog.com/vi/1.1.0/), phiên bản theo [SemVer](https://semver.org/lang/vi/) (trước 1.0: tăng minor khi thêm tính năng hoặc đổi hành vi, tăng patch khi sửa lỗi).

Quy tắc: mọi thay đổi người dùng nhìn thấy được phải thêm một dòng vào `[Unreleased]` trong cùng commit. Khi phát hành, đổi `[Unreleased]` thành `[x.y.z] - YYYY-MM-DD`, tăng version trong `pyproject.toml` và `dlnet/__init__.py`. Xem `agent.md` §11 và `skills/release/SKILL.md`.

## [Unreleased]

- Sửa hiệu năng: vòng quét của Mamba (`selective_scan`) và Gated DeltaNet dùng `unbind` thay cho chỉ số từng bước; backward trước đây cấp phát một tensor 0 bằng cả khối cho mỗi bước (O(L²)). Bước huấn luyện FAHM 13,8 s → 2,1 s, MambaHSI 5,3 s → 1,3 s, MambaHSI+ 23,1 s → 5,8 s (batch 64, 1 luồng CPU); kết quả số học không đổi. Có regression test.

- Quy tắc tài liệu: mọi thay đổi hay nâng cấp phải cập nhật `README.md` và `doc/` (agent.md §17 quy tắc 0); `python scripts/release.py docs` kiểm trong job `lint` (lỗi trên commit phát hành, cảnh báo trên commit thường); `tests/test_docs.py` kiểm số module trong README và link README.

Mốc M27: **v1.0**, làm đầy kho khối theo khảo sát `qa/reports/v0.24.0/module-survey.md`; `agent.md` §9.9.

### Thêm
- **33 op mới** (100 → 133), viết lại bằng torch thuần theo mã chính thức (commit ghim), so số học với mã gốc trên GitHub Actions:
  - attention / chuẩn hóa: `gated_attn_block` (Gated Attention, NeurIPS 2025), `diff_attn_block` (Differential Transformer, ICLR 2025), `dyt` (DyT, CVPR 2025), `rope_vit_block` (RoPE 2-D);
  - tích chập: `wtconv`, `wtconvnext_block` (WTConv), `uib`, `mobile_mqa` (MobileNetV4), `lsnet_block` (LSNet, CVPR 2025), `star_block` (StarNet), `efficientvit_block` (EfficientViT), `msca`, `msca_block` (SegNeXt);
  - chuỗi: `gated_deltanet_block` (Gated DeltaNet, ICLR 2025), `mlstm_block` (xLSTM), `mambavision_block` (MambaVision, CVPR 2025), `gr_kan`, `kat_block` (KAT, ICLR 2025);
  - HSI: `mambahsi_block`, `mambahsi_plus_block`, `ssftt`, `fahm`;
  - bảng / chuỗi thời gian: `realmlp` (RealMLP, NeurIPS 2024), `timemixer` (ICLR 2024), `timexer` (NeurIPS 2024);
  - đồ thị: `gps_layer`, `rwse`, `graph_lin` (GraphGPS), `polynormer` (ICLR 2024), `sgformer` (NeurIPS 2023);
  - recommender: `finalmlp`, `dcnv3`, `wukong`.
- **17 họ mới**:
  - HSI: `mambahsi`, `mambahsi_plus`, `ssftt`, `fahm`;
  - bảng: `realmlp`;
  - chuỗi thời gian: `timemixer`, `timexer`, `xlstm`, `gated_deltanet`;
  - ảnh: `mambavision`, `kat`;
  - đồ thị: `gps_net`, `polynormer_net`, `sgformer_net`;
  - recommender: `finalmlp`, `dcnv3`, `wukong`.
- Đối chiếu số học (CI `verify-next`, `verify-hsi`, `verify-domain`, `verify-seq2`): xem THIRD_PARTY.md mục v1.0. `tests/test_v1.py`: đáp án đóng, tính nhân quả, autocast, `torch.export`.
- `scripts/bench_v1.py`: đo cùng split các họ mới với họ cũ (bảng, ảnh, chuỗi); số trong `agent.md` §9.9.

### Chưa làm
- OverLoCK (NATTEN, kiến trúc ba nhánh), xLSTM-Mixer (sLSTM), TabR/ModernNCA (truy hồi kNN khi huấn luyện), mô hình pretrained (proxy chặn).

## [0.24.0] - 2026-11-11

Mốc M26: module miền tần số (FFT/iFFT), `agent.md` §9.8.

### Thêm
- `dlnet/zoo/ops_freq.py`: các op miền tần số viết theo mã chính thức: `global_filter`, `gfnet_block` (GFNet), `afno`, `afno_block` (AFNO), `spectral_conv`, `fno_block` (FNO), `ffc_block` (FFC), `fca` (FcaNet, DCT), `fnet_block` (FNet); cho ảnh siêu phổ: `spectral_gf` (lọc Fourier dọc trục phổ) và `spectral_rfft` (hệ số Fourier của phổ từng pixel). Ngữ pháp tìm kiếm tự dùng các op này.
- Họ HSI mới: `gfnet2d`, `fno2d`, `afno2d`, `ffc2d`, `fcanet2d`, `spec_fft2d`, `fourier_ss`; khối không gian `gfnet`, `fno`, `afno`, `ffc`, `fca` cho mô hình ghép (`composed`).
- Đo trên Indian Pines (`scripts/bench_freq_hsi.py`, 3 seed): không họ tần số nào thắng ResNet quá nhiễu về OA; `spec_fft2d` với đủ 200 băng đạt OA 0,987 (ResNet 0,986) và AA 0,984 (ResNet 0,969; hơn ở 2/3 seed). FNO/AFNO kém CNN trên patch 9×9. Số đầy đủ: `agent.md` §9.8.
- `scripts/verify_freq.py` + job CI `verify-freq`: so số học GFNet, AFNO, FFC, FcaNet với mã gốc (ghim commit); `tests/test_freq.py`: đáp án đóng cho tích chập vòng, góc tần số FNO, FNet, cơ sở DCT, autocast và torch.export.
- Đối chiếu số học các họ của v0.23.0 với mã gốc: **7/7 khớp** (TabM, PLR, PatchTST trùng từng bit; iTransformer ≤ 4,8e-7). `verify_modern.py` chép trọng số theo tên.
- CI `verify-modern.yml`: chạy `scripts/verify_modern.py` trên GitHub Actions với mã gốc ghim phiên bản (tabm 0.0.3, rtdl_num_embeddings 0.0.12, commit PatchTST/iTransformer).

## [0.23.0] - 2026-11-11

Mốc M25 (Mốc 4 của lộ trình sau kiểm định, `agent.md` §9.7).

### Chưa làm
- Mô hình pretrained (proxy chặn nguồn trọng số); tìm kiếm dựa mô hình (DEHB, PriorBand); graph transformer, DCNv2, Mamba cho HSI.

### Thêm
- Họ bảng **`tabm`**, **`tabm_plr`**: TabM (Gorishniy et al. ICLR 2025) với head `ensemble` (mỗi thành viên một lớp đầu ra và một loss; dự đoán trung bình xác suất), tùy chọn embedding số PLR(lite).
- Họ chuỗi thời gian **`patchtst`** (Nie et al. ICLR 2023) và **`itransformer`** (Liu et al. ICLR 2024).
- `scripts/verify_modern.py`: đối chiếu số học các khối mới với mã chính thức (`tabm`, `rtdl_num_embeddings`, repo PatchTST, iTransformer). **Chưa chạy** trong bản này: các khối mới mới được đối chiếu bằng đọc mã.
- Đo cùng split, 3 seed: TabM+PLR 1,000 so với MLP 0,957 trên wine, các bộ bảng khác trong nhiễu; PatchTST/iTransformer 1,00 so với Transformer cũ 0,37 trên một bộ chuỗi tổng hợp.

## [0.22.0] - 2026-11-11

Mốc M24 (Mốc 3 của lộ trình sau kiểm định, `agent.md` §9.6).

### Thêm
- **`--recipe modern|plain`** (mặc định `modern`): augmentation cho ảnh (dịch, lật ngang cho ảnh màu và phân đoạn) và label smoothing 0,1; huấn luyện đầy đủ có warm-up rồi cosine. Đo: CIFAR-10 3k +10 điểm test (2/2 seed); ảnh xám nhỏ không đổi có ý nghĩa.
- Gói xuất ra dùng được ngay: `train.py` lưu `model.pt` và `preprocess.json`; **`predict.py`** dự đoán file mới (CSV / `.npz` / cube) theo nhãn gốc cho mọi loại dữ liệu; `--export model.pt2` (torch.export) hoặc `model.onnx`; `inference.py` để dùng trong mã Python.
- Leaderboard: `search_coverage` (số ứng viên được huấn luyện) và cảnh báo khi Top-K thiếu.

### Sửa
- `train.py` chạy được từ bất kỳ thư mục nào.
- `.npz` có tên mảng khác `X`/`y` không còn bị đưa nhầm sang plugin HSI; thông báo liệt kê các mảng và cách đặt tên.
- Thông báo rõ cho: file không tồn tại, `--max-params abc`, giới hạn loại hết ứng viên, `--target` sai (gợi ý cột gần đúng).
- Tài liệu cài đặt: torch CPU trước `requirements.txt`, Python 3.10–3.13; yêu cầu của từng loại gói.

## [0.21.0] - 2026-11-11

Mốc M23 (Mốc 2 của lộ trình sau kiểm định, `agent.md` §9.5).

### Đổi
- Successive halving chia ngân sách đều cho các rung (4/12/30 epoch): ít ứng viên hơn nhưng Top-K được huấn luyện sâu hơn (đo 150 epoch: 12 → 30 epoch; test của top-1 không đổi có ý nghĩa, +0,005 ± 0,02).
- Tiến hóa tự bỏ qua khi ngân sách không đủ huấn luyện con tới 12 epoch.
- Ngân sách còn lại chạy lại các finalist tốt nhất (đo nhiễu thật) trước khi ước lượng hyperparameter.
- **`--hparams auto|on|off`** (mặc định `auto`: chỉ ước lượng cho finalist có ≥ 2 lần chạy); chi phí ước lượng tính vào ngân sách epoch; LR đề xuất = LR tại cực tiểu loss / 10.
- Prior viết tay không còn bỏ phiếu chọn hạt giống khi đã có surrogate (và không có `--context`).
- Hyperband và successive halving bù Top-K bằng ứng viên của rung trước khi rung cuối thiếu người.

## [0.20.0] - 2026-11-11

Mốc M22 (Mốc 1 của lộ trình sau kiểm định, `agent.md` §9.4, `qa/reports/v0.19.0/SUMMARY.md`).

### Sửa
- `--scheduler hyperband` không còn sập (`KeyError: 'val0'`) khi ngân sách epoch hết đúng hạn.
- Một ứng viên ném lỗi hoặc phân kỳ (loss NaN/inf, điểm không hữu hạn) bị loại và ghi vào `failed_candidates`, không còn làm dừng lần chạy hay làm sai thứ tự xếp hạng.
- Xếp hạng cuối chỉ so mô hình cùng số epoch: mô hình ít epoch được huấn luyện tiếp nếu còn ngân sách, nếu không thì xếp sau. Leaderboard có cột `epochs` và `runs`.
- "± 0" khi chỉ có một lần chạy được thay bằng sai số lấy mẫu của tập validation; nhóm "không phân biệt được" dùng nhiễu này.
- Vai trò `best-small` không còn gán cho mô hình lớn hơn top-1; `most-stable` chỉ gán khi có ≥ 2 lần chạy.
- Kết luận "thắng baseline" cần hơn trên test quá sai số lấy mẫu, và tính nhiễu cho bài hồi quy.
- CSV: cột số có vài giá trị lạ được đọc thành số thay vì one-hot hàng trăm cột; `--target` sai báo lỗi rõ.
- Ước lượng VRAM của mô hình rất nhỏ không còn bị làm tròn thành 0.

### Thêm
- Mục **Warnings** đầu `leaderboard.md` (và `warnings` trong JSON): top-1 ít epoch, một lần chạy, Top-K lẫn epoch, ứng viên lỗi, HSI chia pixel (rò rỉ không gian), cột CSV bất thường.
- Đội agent kiểm định chất lượng: `qa/CHECKLIST.md` (34 tiêu chí: độ tin cậy, mức hiện đại, tính khả thi, thuật toán), 4 agent trong `.claude/agents/`, skill `quality-audit`. Kết quả kiểm định v0.19.0 và lộ trình: `qa/reports/v0.19.0/SUMMARY.md`, `agent.md` §18.1.

## [0.19.0] - 2026-11-11

Mốc M21 (xem `agent.md` §9.3): huấn luyện trên GPU.

### Thêm
- **`--device auto|cpu|cuda|cuda:N|mps`**: huấn luyện trên GPU CUDA (hoặc Apple MPS) khi có; `auto` chọn GPU nếu torch thấy nó. Dữ liệu được chép lên GPU một lần; HSI cắt patch ngay trên GPU.
- **`--amp auto|off|bf16|fp16`**: mixed precision (bfloat16, hoặc float16 kèm loss scaling trên GPU cũ); `channels_last` cho ảnh trên CUDA. Trên CPU giữ float32.
- Gói xuất ra: `train.py --device ... --amp ...` (mặc định tự dùng GPU nếu có).
- MCP: tool `dlnet_design` nhận `device`, `threads`.
- Khối `budget` của leaderboard ghi thiết bị và độ chính xác.

### Kiểm tra
- Chưa chạy trên GPU thật (môi trường phát triển và CI không có GPU). Đã kiểm: mọi ứng viên của 8 plugin chạy `forward` (train và eval) trên thiết bị `meta`, nơi lỗi lệch thiết bị lộ ra như trên CUDA; vòng huấn luyện chạy trọn trên `meta`; autocast bf16/fp16 và `GradScaler` chạy trên CPU với cùng độ chính xác như float32; đường CPU float32 cho kết quả giống từng bit với v0.18.0.

## [0.18.0] - 2026-11-11

Mốc M20 (xem `agent.md` §9.2): ổn định khi chạy song song và ngân sách được tôn trọng.

### Thêm
- **`--threads N`** (`dlnet/threads.py`): giới hạn luồng của torch, BLAS và OpenMP (scikit-learn, torch) cho cả lần chạy. Mặc định `0` = số CPU còn rảnh. Phụ thuộc mới: `threadpoolctl` (vốn đi kèm scikit-learn).
- Báo cáo ngân sách theo giai đoạn: khối `budget` trong `leaderboard.json`, mục "Budget" trong `leaderboard.md`, dòng `[budget]` trên màn hình; cảnh báo khi vượt quá 10%.

### Sửa
- Nhiều lệnh `design` chạy cùng lúc có thể treo hàng chục phút trong baseline HistGradientBoosting do OpenMP tranh luồng (4 lệnh 30 s: > 10 phút; nay 30–31 s).
- Ngân sách không còn bị vượt bởi một lần huấn luyện bắt đầu sát hạn: DLNet chỉ bắt đầu một lần chạy khi dự báo nó xong trước hạn (tìm kiếm, Hyperband, giai đoạn nhiều seed, ước lượng hyperparameter). Với ngân sách epoch, số epoch đã train không vượt ngân sách.
- File JSON luôn hợp lệ: `NaN`/`inf` (ví dụ intrinsic dimension của dữ liệu trùng mẫu trong `profile.json`) được ghi thành `null`.

## [0.17.0] - 2026-11-11

Mốc M19 (xem `agent.md` §9.1, §18): regression test toàn bộ, ngân sách tái lập được.

### Thêm
- **`--budget 400ep`**: ngân sách theo số epoch huấn luyện (`dlnet/clock.py`). Cùng lệnh cho cùng leaderboard trên máy nhanh hay chậm; các bước không huấn luyện không tính.
- Đo bao phủ (`coverage`, đếm cả lệnh CLI trong tiến trình con): 90% → 97%; job CI `coverage` với sàn 95% (`pyproject.toml`).
- Test mới: `tests/test_cli_flows.py` (mọi nhánh tùy chọn của `design`), `tests/test_coverage_gaps.py` (kiểm tra đầu vào, plugin, gói GAN, MCP, advisor, lỗi spec, RepVGG `fuse`, `train_full`).
- `--plugin autoencoder` / `reconstruction` là tên khác của `image_reconstruction` (trợ giúp CLI đã nói vậy nhưng trước đây báo lỗi).
- `agent.md` §18: lộ trình nâng cấp đề xuất (ổn định, hiệu quả, thông minh).

### Sửa
- `design` crash khi advisor lỗi ở lần gọi đầu nhưng trả lời ở vòng tiến hóa; khối `ai_advisor` nay luôn có khi dùng advisor.
- Dữ liệu hằng hoặc trùng điểm: không còn cảnh báo 0/0 trong ước lượng intrinsic dimension; `pca_dims_for_95pct_energy` không vượt số đặc trưng.
- Các test e2e khẳng định chất lượng dùng ngân sách epoch, không còn đỏ khi máy chậm.

## [0.16.0] - 2026-11-10

Mốc M17 và M18 (xem `agent.md` §16.6, `doc/algorithms.md` §3, `doc/ai-advisor.md`).

### Thêm
- **Surrogate zero-shot ZAP** (`dlnet/surrogate.py`, Öztürk et al. 2022): học ánh xạ (data card, đặc tính kiến trúc) → thứ hạng bằng loss xếp hạng cặp; tham gia xếp hạng hạt giống (`--surrogate-weight`).
- **Meta-dataset có sẵn** `dlnet/data/meta_image.jsonl`: 14 bộ ảnh (CIFAR-10/100, hoa, MNIST, Fashion-MNIST, ...) × 24 kiến trúc; tắt bằng `--no-builtin-meta`. `--meta-store` nay ghi data card và kết quả từng ứng viên.
- `scripts/build_meta.py`, `scripts/bench_surrogate.py` (leave-one-dataset-out): Borda(ZAP) Spearman +0,67 so với +0,19 của quy tắc tay và +0,33 của đếm tham số.
- **AI advisor tùy chọn** (`--advisor claude|codex|gemini`, `--advisor-cmd`): AI CLI xếp hạng hạt giống và đề xuất kiến trúc trước khi tìm kiếm và ở mỗi thế hệ tiến hóa; câu trả lời được kiểm tra như dữ liệu. `leaderboard.json` có khối `ai_advisor`.
- **MCP server** `dlnet mcp` (stdio, JSON-RPC): tool `dlnet_profile`, `dlnet_candidates`, `dlnet_zoo`, `dlnet_check_spec`, `dlnet_design`, `dlnet_leaderboard`; cấu hình cho Claude Code, Codex CLI, Antigravity, Gemini CLI trong `doc/ai-advisor.md`.
- `--extra-specs file.json`: thêm kiến trúc viết tay (hoặc do AI viết) vào pool.
- Skill `ai-advisor`.

## [0.15.0] - 2026-11-10

Mốc M16 (xem `doc/user-guide.md` §5, `doc/algorithms.md` §2, `agent.md` §17): hướng tối ưu và bộ tài liệu.

### Thêm
- **`--objective accuracy | params | flops | balanced | pareto`** (`dlnet/objectives.py`): nhỏ nhất trong `--tolerance` của mô hình tốt nhất (ε-ràng buộc); phần thưởng MnasNet `acc × (cost/T)^w` (`--target-cost`, `--cost-weight`, `--cost params|flops`); xếp hạng Pareto (sắp xếp không bị trội + crowding distance của NSGA-II). Áp dụng ở mọi bước: giữa các rung, tournament của tiến hóa, Top-K. `leaderboard.md` luôn có bảng Pareto front.
- `--max-params`; `--goal balanced`; seed selection theo `--objective`.
- **Thư mục `doc/`**: `README.md` (mục lục, phiên bản tài liệu), `user-guide.md`, `architecture.md`, `algorithms.md`, và `cli.md`, `zoo.md` sinh tự động bằng `scripts/gen_docs.py`.
- `python scripts/release.py bump X.Y.Z`: sửa phiên bản ở mọi nơi, mục CHANGELOG, phiên bản tài liệu, sinh lại tài liệu; `release.py check` báo lỗi khi `doc/README.md` lệch phiên bản; `tests/test_docs.py` báo lỗi khi tài liệu sinh tự động lệch mã hoặc có link hỏng.

## [0.14.0] - 2026-11-10

Mốc M15 (xem `agent.md` §16.5): phân tích dữ liệu, chọn hạt giống, thuật toán tìm kiếm chuẩn.

### Thêm
- **Data card ảnh** (`dlnet/datacard.py`): kích thước, mất cân bằng lớp, độ dốc phổ công suất, năng lượng tần số cao, độ dài tương quan, mật độ cạnh, độ phụ thuộc vị trí, tương quan kênh màu, 1-NN, và hai linear probe (pixel so với đặc trưng tích chập ngẫu nhiên). In ở `[datacard]`, ghi vào `leaderboard.json`.
- **Chọn hạt giống** (`dlnet/seeds.py`, `--seeds N`, `--context "..."`, `--goal small`): đặc tính kiến trúc đọc từ module trong spec, quy tắc theo data card và ngữ cảnh (in lý do), tổng hợp xếp hạng Borda với meta-learning (kNN + portfolio tham lam của Auto-sklearn 2.0) và, khi bật `--seed-proxies`, vote đa số của zero-cost proxy (Abdelfattah 2021). Luôn giữ một ResNet làm mốc.
- **Hyperband** (`--scheduler hyperband`, Li et al. 2018, Algorithm 1) và **aging evolution** (Real et al. 2019, tournament `--tournament`, quần thể `--evolve-population`) thay cho chọn 6 cha mẹ tốt nhất.
- Proxy **SNIP** và **ZiCo** (theo mã chính thức); báo cáo Spearman có thêm `vote` và `seed_prior`.
- `scripts/bench_seeds.py`: so tất cả / ngẫu nhiên N / theo dữ liệu N / theo dữ liệu N + Hyperband cùng ngân sách.

### Đo (xem `agent.md` §16.5)
- Prior theo data card **chưa** dự đoán được thứ hạng huấn luyện (Spearman −0,18…+0,35 trên MNIST, Fashion-MNIST, hai bộ tổng hợp); vì vậy `--seeds` mặc định 0 (không loại ứng viên) và proxy vẫn không được dùng để loại.

### Sửa
- Ảnh nhiều kênh được lưu liền mạch trong bộ nhớ: trước đây stride channels-last làm torch 2.15 crash ("double free or corruption") khi backward đa luồng.
- Bước zero-cost proxy được 10% ngân sách tính từ lúc nó bắt đầu (trước đây data card và baseline đã tiêu hết, báo cáo proxy biến mất).
- Autoencoder được so với PCA có số thành phần không vượt kích thước mã của nó (trước đây luôn so với PCA-32).

## [0.13.0] - 2026-11-10

Mốc M14 (xem `agent.md` §13.8, `THIRD_PARTY.md`): khối ảnh từ thư viện chính thống.

### Thêm
- **Decoder phân đoạn của segmentation_models_pytorch 0.6.0**: `unet`, `unetpp` (UNet++), `fpn`, `deeplabv3`, `deeplabv3plus`, `pspnet`, `linknet`, `upernet`, `segformer`, tùy chọn `attention: scse`; họ phân đoạn tương ứng; preset; phép đột biến `decoder`.
- **13 khối từ timm và MONAI** (`dlnet/zoo/ops_img.py`): `scse`, `coord_attn`, `gc_block`, `grn`, `sk_conv`, `split_attn`, `ghost_bottleneck`, `res2net_block`, `convmixer_block`, `repvgg_block` (có `fuse()`), `poolformer_block`, `segres_block`, `unet_res_block`; preset phân loại `ghostnet`, `repvgg`, `poolformer`, `convmixer`, `segresnet`, `nnunet_res`, `res2net`, `sknet`, `resnest`, `resnet_cbam/eca/coord/gc`.
- `scripts/verify_vision_libs.py`: đối chiếu số học với timm, torchvision, smp, MONAI (39/39 khớp).

### Đổi
- `cbam`, `eca`, `inception`, `window_attn`, `mixer_block` viết lại theo mã gốc (timm, torchvision); decoder `unet`, `fpn`, `aspp` thay bằng bản smp. Số tham số và kết quả của các mô hình dùng chúng khác v0.12; `dec: "aspp"` vẫn đọc được (là `deeplabv3`).
- `window_attn` không còn đòi kích thước chia hết cho cửa sổ (đệm như torchvision).

## [0.12.0] - 2026-11-10

Mốc M13 (xem `agent.md` §13.7, `THIRD_PARTY.md`): port đồ thị, id/recommender, autoencoder/VAE, GAN.

### Thêm
- **Kiểu `graph`** và plugin `graph` (`.npz` với `X`, `A`, `y`): lớp GCN, GAT, GraphSAGE, GIN và readout; họ `gcn_net`, `gat_net`, `sage_net`, `gin_net`, `graph_mlp`; baseline cổ điển trên thống kê đồ thị. Hai bộ lọc lưới của rasbt (`grid_graph_avg`, `grid_spectral_conv`).
- **Kiểu `ids`** và plugin `recsys` (`.npz` với `ids`, `y`): `embed_fields`, `gmf`, `pair_dots`; họ `mf`, `ncf`, `fm`, `dlrm`; baseline one-hot.
- **Tác vụ `reconstruction`** và plugin `image_reconstruction` (`.npz` chỉ có `X`): họ `conv_ae`, `vae` (`vae_bottleneck` với KL), baseline PCA; metric MSE, PSNR.
- **Lệnh `dlnet gan`**: DCGAN từ zoo, xếp hạng theo MMD², baseline Gaussian và ảnh train; gói xuất ra có `sample.py`.
- Zoo có preset và bộ lấy mẫu cho `graph` và `ids`; `head: "none"` cho mô hình không cần head.

### Sửa
- Cost model nay cũng áp cho ứng viên con của vòng tiến hóa (trước đó cột MFLOPs của chúng bị trống).
- `moe` bị từ chối trên đầu vào `ids` (cổng cần giá trị thực); phép chiếu residual báo `ValueError` cho kiểu không hỗ trợ.

### Hạn chế đã biết
- Đồ thị: chỉ đồ thị dày nhỏ; trên dữ liệu tổng hợp GNN **kém** baseline thống kê đồ thị. Recommender: chỉ điểm (pointwise), chưa có loss cặp BPR; kết quả MF tốt vì dữ liệu tổng hợp sinh đúng theo phân rã. Autoencoder: PCA 32 thành phần thắng conv-AE trên digits; baseline PCA không tự khớp kích thước mã.
- GAN: chỉ DCGAN, chỉ MMD² trong không gian pixel (không có FID); chưa có CVAE và GAN có điều kiện.

## [0.11.0] - 2026-11-10

Mốc M12 (xem `agent.md` §13.6, `THIRD_PARTY.md`): một thư viện duy nhất, cost model FLOPs/VRAM, đối chiếu số học với mã gốc.

### Thêm
- **Cost model** (`dlnet/zoo/cost.py`) cho mọi ứng viên: FLOPs mỗi mẫu (ma trận và tích chập), bộ nhớ kích hoạt và VRAM huấn luyện ước lượng, đo trên thiết bị `meta` (không tốn bộ nhớ). `--max-flops 50M` và `--max-vram 4000` loại ứng viên trước khi huấn luyện; leaderboard có cột MFLOPs và MB, `report.md` có dòng cost model.
- **Đối chiếu số học với mã gốc** (`scripts/verify_numeric.py`): chép trọng số từ khối gốc sang khối của zoo rồi so đầu ra. 24/24 khối khớp, gồm Mamba và Mamba-2 (`selective_scan_ref` và `Mamba` của `mamba_ssm`, sai khác 0), khối transformer và conv1d residual của cortex, ViT encoder block, Swin PatchMerging, MobileNetV3, RegNet, ConvNeXt, EfficientNet, ShuffleNet, SqueezeNet, ResNet/ResNeXt.
- Module HSI trong zoo: `center_spectrum`, `to_vol`, `vol_to_map`, `tucker_conv`, `pool_flatten`; `conv_bn_act` và `pool` nhận nhân theo từng trục (`k=[7,3,3]`) và `pad: "valid"`; `par` có `merge: "gated"`.
- `dlnet/zoo/families.py`: 33 họ mô hình (bảng, ảnh, phân đoạn, chuỗi thời gian, HSI) viết thành công thức sinh spec zoo.

- **Mamba-2** (`mamba2_block`, SSD theo khối, PyTorch thuần): khớp `Mamba2Simple` của `mamba_ssm` đến 2,4e-7 và `ssd_minimal` đến 0; không giới hạn độ dài như Mamba bản quét tuần tự.

### Đổi
- **Gỡ `library.py`, `array_lib.py`, `hsi_lib.py`, `blocks.py` và lệnh `blocks`**: mọi họ là công thức trên zoo nên chỉ còn một registry, một bộ kiểm tra tĩnh, một cost model và mọi đột biến/lai ghép áp được cho cả họ cũ. `model.py` trong gói xuất ra là shim một dòng dùng thư mục `zoo/`. 22/33 họ có số tham số trùng tuyệt đối với bản cũ, phần còn lại lệch ≤ 10%.
- `transformer_block` theo cortex: mặc định không bias (`bias=0`), thêm `cortex_skip` để tái hiện nhánh skip của cortex.
- `conv1d_resid` dùng đúng LayerNorm theo kênh của cortex. `pool` làm tròn lên khi kích thước lẻ.

### Sửa
- Phép đối chiếu số học phát hiện và sửa: SE của MobileNetV3 đặt trên kênh đã mở rộng; khác biệt giữa transformer của cortex và transformer chuẩn (nay có tùy chọn).

### Hạn chế đã biết
- VRAM là ước lượng, chưa đối chiếu với đo trên GPU (không có GPU ở môi trường phát triển). FLOPs không tính phép elementwise, chuẩn hóa, kích hoạt và vòng lặp quét của Mamba.
- Chưa đối chiếu với mã gốc: `window_attn` (không có relative position bias), ASPP/FPN/FCN rút gọn, CBAM, ECA, Inception, MLP-Mixer, GLU, DCN, FT-Transformer, U-Net.
- Các họ lệch ≤ 10% so với bản cũ (xem `THIRD_PARTY.md`), nên kết quả huấn luyện của chúng có thể khác bản trước.

## [0.10.0] - 2026-11-10

Mốc M11 (xem `agent.md` §13.6, `THIRD_PARTY.md`): refactor lõi thành kho module có kiểu và tìm kiếm kiến trúc.

### Thêm
- **`dlnet/zoo/`**: kho module có kiểu dùng chung cho mọi plugin (bảng, HSI, ảnh, phân đoạn, chuỗi thời gian). 54 module có khai báo kiểu vào/ra, không gian tham số, thẻ chức năng và nguồn gốc; gom từ rasbt/deeplearning-models, prescient-design/cortex, ylongqi/openrec, yamsgithub/modular_deep_learning, pytorch/vision và state-spaces/mamba (giấy phép không xét vì dùng nội bộ). Liệt kê: `python -m dlnet zoo`; phần chưa port và lý do: `THIRD_PARTY.md`.
- **Kiến trúc là cây spec JSON** ghép từ module và các toán tử `seq`, `res`, `par`, `rep`, `moe`, `encdec`. `zoo.check` kiểm tra tĩnh kiểu, shape và số tham số (dựng trên thiết bị `meta`) trước khi huấn luyện.
- **Tìm kiếm kiến trúc**: `design` nay thêm ứng viên từ zoo (`--zoo N`, mặc định 12: tối đa một nửa là preset kiến trúc nổi tiếng viết dưới dạng spec, còn lại là mẫu ngẫu nhiên từ ngữ pháp) và chạy tiến hóa (`--evolve G` thế hệ, `--evolve-pop` con mỗi thế hệ) bằng 8 phép đột biến và lai ghép theo tầng. Nhãn nguồn gốc `origin` có trong `leaderboard.json` và `report.md`; khối `module_zoo` trong `leaderboard.json` ghi các thế hệ.
- Gói xuất ra có thêm thư mục `zoo/` khi kiến trúc đến từ zoo; vẫn chạy độc lập.
- `scripts/verify_zoo.py`: so 14 khối đã port với khối gốc của torchvision (số tham số và shape): 14/14 khớp. Các con số được ghim vào test.
- `scripts/bench_zoo.py`: so top-1 của họ cũ với zoo + tiến hóa cùng ngân sách.
- Skill mới `module-zoo`.

### Đổi
- Lớp TT/TR/low-rank của `library.py` nay lấy từ `dlnet/zoo/ops_vec.py` (một bản duy nhất).
- Các họ cũ (`library.py`, `array_lib.py`, `hsi_lib.py`, `blocks.py`) vẫn chạy song song với zoo.

### Sửa
- Khối MobileNetV3 phát hiện sai khi đối chiếu với torchvision (SE phải đặt trên kênh đã mở rộng): đã sửa trước khi phát hành.

### Hạn chế đã biết
- **Chưa có bằng chứng zoo tìm được kiến trúc tổng quát hóa tốt hơn họ cũ.** Đo 12 cặp (6 bộ dữ liệu nhỏ hoặc tổng hợp, 2 seed, 100 s): validation top-1 của zoo cao hơn 8/12, nhưng test chỉ cao hơn 4/12 và kém 7/12; phần lớn lợi thế validation là thiên lệch lựa chọn. Người thắng của zoo thường đến từ đột biến hoặc ngẫu nhiên (10/12), tức tìm kiếm có tạo cấu trúc mới được chọn. Chi tiết ở `agent.md` §13.6.
- Mọi ứng viên zoo đi qua cùng ngân sách thời gian, nên thêm nhiều ứng viên làm các rung đầu ngắn hơn cho từng ứng viên.
- Mamba là bản quét tuần tự (độ dài tối đa 256), chưa so với `mamba_ssm`. Các khối từ paper chưa đối chiếu mã tác giả. Chưa có kiểu `graph`/`ids`, trọng số pretrained, autoencoder/GAN (cần vòng huấn luyện khác).
- Cost model vẫn chỉ đếm tham số (chưa có FLOPs/VRAM).

## [0.9.0] - 2026-11-10

Mốc M10 (xem `agent.md` §8, §13.5, §14.4): thư viện khối từ bốn repo GitHub, plugin phân loại và phân đoạn ảnh.

### Thêm
- `dlnet/blocks.py`: thư viện khối dùng chung (torch only): backbone (`lenet`, `allconv`, `vgg`, `resnet`, `densenet`, `mobilenetv2`), attention (`se`, `self_attention_2d`), decoder (`fcn`, `unet`, `deconv`), `MixtureOfExperts` (soft, top-k, product), `LatentFactor`, `SecondOrderInteraction`, `pairwise_log_loss`, `NeuralTree`/`SumTrunk`/`MeanPooling`. Mỗi khối ghi nguồn; liệt kê bằng `python -m dlnet blocks`. Được copy vào gói xuất ra (`blocks.py`).
- `THIRD_PARTY.md`: repo, commit, giấy phép và phần đã dùng/chưa dùng của rasbt/deeplearning-models (MIT), prescient-design/cortex (Apache-2.0), ylongqi/openrec (Apache-2.0), yamsgithub/modular_deep_learning (không có giấy phép, nên chỉ cài lại phương pháp, không sao chép mã).
- Plugin `image_segmentation` (`.npz` với `X` và mask `y`): họ `fcn`, `unet`, `deconvnet` với encoder `vgg`/`resnet`/`densenet`/`mobilenetv2` và tùy chọn attention; chia theo ảnh; metric mIoU; baseline pixel 5×5.
- Plugin `image_classification` (đổi tên từ `image`, tên cũ vẫn dùng được) có thêm các họ `lenet`, `allconv`, `vgg`, `densenet`, `mobilenetv2`.
- Dữ liệu bảng có thêm họ `moe_mlp` (mixture of experts) và `fm_mlp` (tương tác bậc hai kiểu factorisation machine). `design` sinh ứng viên từ 8 họ.

### Đổi
- Router không còn trả điểm 0,3 cho plugin ảnh/chuỗi khi hình dạng không khớp; nhập nhằng thì dùng `--plugin`.

### Hạn chế đã biết
- Không khối nào là bản tái hiện đã kiểm chứng của mô hình gốc; không có trọng số pretrained.
- Đo: digits (dữ liệu thật nhỏ) top-1 thắng baseline; phân đoạn chỉ đo trên dữ liệu tổng hợp rất dễ và **không** thắng baseline pixel 5×5 có ý nghĩa. Chưa có augmentation, Dice/focal loss, ảnh lớn.
- `NeuralTree`, `SumTrunk`, `MeanPooling`, `LatentFactor`, `pairwise_log_loss` có test nhưng chưa có plugin nào dùng (chưa có bài toán nhiều đầu vào hoặc gợi ý).

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

[Unreleased]: https://github.com/Duy-Huan/DLNet/compare/v0.24.0...HEAD
[0.24.0]: https://github.com/Duy-Huan/DLNet/releases/tag/v0.24.0
[0.23.0]: https://github.com/Duy-Huan/DLNet/releases/tag/v0.23.0
[0.22.0]: https://github.com/Duy-Huan/DLNet/releases/tag/v0.22.0
[0.21.0]: https://github.com/Duy-Huan/DLNet/releases/tag/v0.21.0
[0.20.0]: https://github.com/Duy-Huan/DLNet/releases/tag/v0.20.0
[0.19.0]: https://github.com/Duy-Huan/DLNet/releases/tag/v0.19.0
[0.18.0]: https://github.com/Duy-Huan/DLNet/releases/tag/v0.18.0
[0.17.0]: https://github.com/Duy-Huan/DLNet/releases/tag/v0.17.0
[0.16.0]: https://github.com/Duy-Huan/DLNet/releases/tag/v0.16.0
[0.15.0]: https://github.com/Duy-Huan/DLNet/releases/tag/v0.15.0
[0.14.0]: https://github.com/Duy-Huan/DLNet/releases/tag/v0.14.0
[0.13.0]: https://github.com/Duy-Huan/DLNet/releases/tag/v0.13.0
[0.12.0]: https://github.com/Duy-Huan/DLNet/releases/tag/v0.12.0
[0.11.0]: https://github.com/Duy-Huan/DLNet/releases/tag/v0.11.0
[0.10.0]: https://github.com/Duy-Huan/DLNet/releases/tag/v0.10.0
[0.9.0]: https://github.com/Duy-Huan/DLNet/releases/tag/v0.9.0
[0.8.0]: https://github.com/Duy-Huan/DLNet/releases/tag/v0.8.0
[0.7.0]: https://github.com/Duy-Huan/DLNet/releases/tag/v0.7.0
[0.6.0]: https://github.com/Duy-Huan/DLNet/releases/tag/v0.6.0
[0.5.0]: https://github.com/Duy-Huan/DLNet/releases/tag/v0.5.0
[0.4.0]: https://github.com/Duy-Huan/DLNet/releases/tag/v0.4.0
[0.3.0]: https://github.com/Duy-Huan/DLNet/releases/tag/v0.3.0
[0.2.0]: https://github.com/Duy-Huan/DLNet/releases/tag/v0.2.0
[0.1.0]: https://github.com/Duy-Huan/DLNet/releases/tag/v0.1.0
