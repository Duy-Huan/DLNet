# Changelog

Định dạng theo [Keep a Changelog](https://keepachangelog.com/vi/1.1.0/), phiên bản theo [SemVer](https://semver.org/lang/vi/) (trước 1.0: tăng minor khi thêm tính năng hoặc đổi hành vi, tăng patch khi sửa lỗi).

Quy tắc: mọi thay đổi người dùng nhìn thấy được phải thêm một dòng vào `[Unreleased]` trong cùng commit. Khi phát hành, đổi `[Unreleased]` thành `[x.y.z] - YYYY-MM-DD`, tăng version trong `pyproject.toml` và `dlnet/__init__.py`. Xem `agent.md` §11 và `skills/release/SKILL.md`.

## [Unreleased]

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

[Unreleased]: https://github.com/Duy-Huan/DLNet/compare/v0.11.0...HEAD
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
