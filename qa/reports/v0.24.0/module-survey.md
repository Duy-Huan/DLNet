# Khảo sát module cutting-edge cho kho zoo (sau v0.24.0)

Ngày: 2026-11-11. Nguồn: tìm kiếm paper và GitHub (link ở cuối). Mục tiêu: chọn các khối nên đưa vào `dlnet/zoo` để làm phong phú không gian tìm kiếm kiến trúc.

## Tiêu chí chọn

Một khối chỉ nên vào zoo khi đáp ứng cả bốn điều kiện:

1. **Có mã chính thức PyTorch**: để port đúng và so số học trên CI (như `verify_freq.py`, `verify_modern.py`).
2. **Viết lại được bằng torch thuần**: không cần kernel CUDA/Triton riêng, vì DLNet chạy được trên CPU.
3. **Hợp với dữ liệu nhỏ đến vừa** mà DLNet nhắm tới, không phải loại chỉ mạnh khi pretrain trên hàng tỷ mẫu.
4. **Lắp được vào ngữ pháp**: nhận và trả đúng kiểu tensor (`vec`, `seq`, `map`, `ser`, `vol`, `graph`, `ids`), thì Hyperband và tiến hóa tự dùng được.

Kho hiện có 100 op. Đã có: ResNet/ConvNeXt/MBConv/RepVGG/GhostNet, SE/CBAM/ECA/CoordAttn, Transformer/window attention, Mamba-1/2, PatchTST/iTransformer, TabM/PLR, GCN/GAT/SAGE/GIN, DLRM/FM, và từ v0.24 có GFNet/AFNO/FNO/FFC/FcaNet/FNet.

Mọi con số dưới đây là **tác giả tự báo cáo**, chưa ai đo lại độc lập; DLNet phải đo lại cùng ngân sách trước khi kết luận.

## Nhóm 1: toán tử dùng chung, giá trị cao, chi phí thấp (đề xuất v0.25)

Các khối này là "phép thay thế tại chỗ" (drop-in), áp được cho nhiều miền dữ liệu cùng lúc, nên nhân số kiến trúc ngữ pháp tạo được.

| Khối | Nguồn | Ý tưởng | Lắp vào DLNet | Mã chính thức |
|---|---|---|---|---|
| **Gated Attention** | Qiu et al., NeurIPS 2025 (oral; Qwen3-Next dùng) | cổng sigmoid theo từng head sau SDPA: thêm phi tuyến, attention thưa, hết "attention sink", ổn định với LR lớn | tùy chọn `gate` cho `self_attn`, `transformer_block`, `window_attn`, PatchTST/iTransformer | qiuzh20/gated_attention |
| **Differential Attention** | Ye et al., ICLR 2025 (oral, Microsoft) | attention = hiệu của hai bản đồ softmax, khử nhiễu, chú ý thưa hơn | op `diff_attn` / tùy chọn trong `transformer_block` | microsoft/unilm (Diff-Transformer) |
| **DyT (Dynamic Tanh)** | Zhu, He, LeCun, Liu, CVPR 2025 | thay LayerNorm bằng tanh(αx) học được; ngang hoặc hơn, nhanh hơn | thêm `norm="dyt"` cho mọi khối dùng `norm` | jiachenzhu/DyT |
| **2-D RoPE (RoPE-Mixed)** | Heo et al., ECCV 2024 (NAVER) | rotary position embedding 2 chiều cho ViT, tần số trộn trục học được; bền khi đổi độ phân giải | `pos` mode `rope2d` cho `vit`, `window_attn` | naver-ai/rope-vit |
| **WTConv** | Finder et al., ECCV 2024 | tích chập trên các mức wavelet: trường nhìn rất rộng, tham số tăng theo log(k); bền với nhiễu, thiên về hình dạng | op `wtconv` (map, có thể mở rộng `ser`/`vol`); **rất hợp HSI** và nối tiếp hướng miền tần số của v0.24 | BGU-CS-VIL/WTConv (wavelet Haar viết trực tiếp, không cần PyWavelets) |
| **MobileNetV4 UIB + Mobile MQA** | Qin et al., ECCV 2024 (Google) | Universal Inverted Bottleneck gộp IB/ConvNeXt/FFN/ExtraDW; MQA nhanh hơn MHSA > 39 % | op `uib` (thay dần `mbconv`), `mqa` | timm `mobilenetv4` (đối chiếu bằng `verify_vision_libs.py` sẵn có) |
| **LSNet (LS conv)** | Wang et al., CVPR 2025 | "nhìn rộng, tập trung hẹp": kernel lớn để cảm nhận, kernel nhỏ để gộp | op `ls_conv` (map) | jameslahm/lsnet |
| **StarNet (star op)** | Ma et al., CVPR 2024 | nhân phần tử hai nhánh (x₁ ⊙ x₂) ánh xạ ngầm lên không gian chiều cao; rất nhẹ | op `star_block` (map) | ma-xu/Rewrite-the-Stars |
| **EfficientViT ReLU linear attention** | Cai et al., ICCV 2023 (MIT Han lab) | attention tuyến tính ReLU + token đa tỉ lệ bằng conv nhỏ; nhanh cho phân đoạn độ phân giải cao | op `relu_linear_attn` (map, seq) | mit-han-lab/efficientvit |
| **SegNeXt MSCA** | Guo et al., NeurIPS 2022 | chú ý tích chập đa tỉ lệ (dải conv 1×k, k×1) thay self-attention | op `msca` (map), decoder phân đoạn nhẹ | Visual-Attention-Network/SegNeXt |

**Ước lượng công sức**: mỗi khối 50–150 dòng; cả nhóm khoảng một mốc. Đối chiếu số học: thêm job vào `.github/workflows/verify-modern.yml` như v0.24.

## Nhóm 2: họ theo miền dữ liệu (đề xuất v0.26)

### Ảnh siêu phổ (HSI), ưu tiên vì là trọng tâm của bạn

| Khối/họ | Nguồn | Ghi chú | Mã |
|---|---|---|---|
| **MambaHSI** | Li et al., arXiv 2501.04944 | spatial Mamba block ở mức pixel cho toàn ảnh + spectral block chia nhóm băng; DLNet đã có `mamba_block` torch thuần nên port được | li-yapeng/MambaHSI |
| **SSFTT** | Sun et al., TGRS 2022 | conv 3D/2D + tokenizer + transformer; baseline mạnh, rẻ, được so trong gần như mọi bài HSI 2024–2025 | zgr6010/HSISSFTT |
| **MambaHSI+** | TGRS 2025 | lan truyền trạng thái đa hướng, bản V2 nhẹ và chính xác hơn | RockAilab/MambaHSI_Plus |
| **FAHM** (Frequency-Aware Hierarchical Mamba) | 2025 | **Mamba + miền tần số**, khớp đúng hướng FFT bạn quan tâm | zhangxc0105/FAHM (cần kiểm tra repo có mã đầy đủ) |
| **MambaMoE** | Information Fusion 2025 | Mamba + mixture-of-experts phổ–không gian | YichuXu/MambaMoE |
| *Không đề xuất*: MorphFormer, GSC-ViT (không tìm thấy mã chính thức); HyperSMamba (mã "sẽ công bố") | | | |

### Bảng (tabular)

| Khối/họ | Nguồn | Ghi chú | Mã |
|---|---|---|---|
| **RealMLP** | Holzmüller et al., NeurIPS 2024 | MLP với nhiều kỹ thuật nhỏ (robust scaling, smooth clipping, PBLD embedding, front-scale, NTK init); **đứng đầu Elo TabArena** (1564, TabM 1541) | dholzmueller/pytabkit |
| **TabR / ModernNCA** | ICLR 2024 / ICLR 2025 | mô hình dựa truy hồi láng giềng trong tập train; cần đường huấn luyện riêng (truy vấn kNN mỗi batch) | yandex-research/tabular-dl-tabr |
| *Không làm được*: TabPFN v2/2.5 (Nature 2025) mạnh với dữ liệu nhỏ nhưng là **mô hình pretrained**, trọng số bị proxy chặn; và không phải khối kiến trúc | | | |

### Chuỗi thời gian

| Khối/họ | Nguồn | Ghi chú | Mã |
|---|---|---|---|
| **TimeMixer** | Wang et al., ICLR 2024 | phân rã mùa/xu hướng đa tỉ lệ, trộn bằng MLP; nhẹ | kwuking/TimeMixer |
| **TimeXer** | NeurIPS 2024 (thuml) | Transformer với biến ngoại sinh; xếp đầu bảng Time-Series-Library | thuml/TimeXer |
| **xLSTM-Mixer** | NeurIPS 2025 | khối xLSTM (mLSTM/sLSTM) trộn biến; báo SOTA 6/7 bộ dài hạn | mauricekraus/xLSTM-Mixer |
| Phân loại chuỗi: HIVE-COTE 2 / MultiRocket mạnh nhất trên UCR/UEA nhưng là phương pháp cổ điển: nên dùng làm **baseline tham chiếu** trong báo cáo, không phải khối zoo | | | |

### Đồ thị

| Khối/họ | Nguồn | Ghi chú | Mã |
|---|---|---|---|
| **GraphGPS + RWSE** | Rampášek et al., NeurIPS 2022; OpenGT 2025 | MPNN cục bộ + attention toàn cục + mã hóa vị trí random-walk; OpenGT 2025 thấy đây là lựa chọn ổn định nhất cho bài toán mức đồ thị | rampasek/GraphGPS |
| **Polynormer** | Deng et al., ICLR 2024 | graph transformer tuyến tính, attention cục bộ rồi toàn cục, biểu diễn đa thức | cornell-zhang/Polynormer |
| **SGFormer** | Wu et al., NeurIPS 2023 | một lớp attention toàn cục tuyến tính, không cần mã hóa vị trí | qitianwu/SGFormer |

### Recommender / CTR

| Khối/họ | Nguồn | Ghi chú | Mã |
|---|---|---|---|
| **DCNv3 / FCN** | 2024–2025 | mạng chéo mũ + tuyến tính; đầu bảng Criteo (AUC 0,8162, theo trang tổng hợp) | link trong paper (arXiv 2407.13349) |
| **FinalMLP** | AAAI 2023 (Huawei) | MLP hai luồng + feature gating + tổng hợp tương tác | reczoo/FuxiCTR |
| **Wukong** | Meta, ICML 2024 | chồng khối factorization machine, có luật mở rộng | (mã trong paper) |

## Nhóm 3: bộ trộn chuỗi thế hệ mới và KAN (đề xuất v0.27, cần đo kỹ)

| Khối | Nguồn | Vì sao để sau | Mã |
|---|---|---|---|
| **Gated DeltaNet** | Yang, Kautz, Hatamizadeh, ICLR 2025 | thay thế Mamba-2 bằng delta rule có cổng; mạnh ở chuỗi dài, nhưng bản nhanh cần kernel Triton (flash-linear-attention); torch thuần chạy được dạng hồi quy, chậm | NVlabs/GatedDeltaNet, fla-org/flash-linear-attention |
| **mLSTM (xLSTM)** | Beck et al., NeurIPS 2024 | tương tự; có dạng song song torch thuần | NX-AI/xlstm |
| **GR-KAN (KAT)** | Yang & Wang, ICLR 2025 | thay MLP bằng KAN hàm hữu tỉ theo nhóm; FlashKAT (AAAI) báo huấn luyện **chậm 123 lần** dù FLOPs ngang; trên bảng KAN chỉ ngang MLP | Adamdad/kat |
| **OverLoCK (ContMix)** | Lou & Yu, CVPR 2025 (oral) | kernel động trộn ngữ cảnh, 3 mạng con; mạnh trên ImageNet nhưng nặng, cấu trúc nhánh khó đưa vào ngữ pháp tuần tự | LMMMEng/OverLoCK |
| **MambaVision** | NVlabs, CVPR 2025 | lai Mamba + attention ở tầng sau; ghép được từ `mamba_block` + `transformer_block` sẵn có bằng một preset | NVlabs/MambaVision |

## Không đề xuất (và lý do)

- **Mô hình nền pretrained** (TabPFN v2.5, Chronos-2/TiRex, DINOv3, các ViT lớn): trọng số tải từ huggingface.co / download.pytorch.org bị proxy chặn trong môi trường phát triển; nếu môi trường mở mạng, đây là hướng **tinh chỉnh**, khác với tìm kiếm kiến trúc.
- **Vision-RWKV / RWKV-7**: cần kernel WKV CUDA; bản torch thuần rất chậm trên CPU.
- **ViT-5 (2026)**, CPUBone (2026): quá mới, chưa có kiểm chứng độc lập; xem lại sau.

## Đề xuất lộ trình

| Phiên bản | Nội dung | Số op thêm (ước) | Cách kiểm |
|---|---|---|---|
| **v0.25** | Nhóm 1: gated attention, diff attention, DyT, RoPE-2D, WTConv, UIB/MQA, LS conv, star block, ReLU linear attention, MSCA | ~10 | so số với mã gốc trên CI; đo trên CIFAR-10 3k, digits, Indian Pines (chia khối) |
| **v0.26** | Nhóm 2: MambaHSI, SSFTT, FAHM (HSI); RealMLP (bảng); TimeMixer, TimeXer (chuỗi); GraphGPS+RWSE, Polynormer (đồ thị); FinalMLP, DCNv3 (CTR) | ~10 họ | như trên + bộ benchmark sẵn có của từng plugin |
| **v0.27** | Nhóm 3: Gated DeltaNet, mLSTM, GR-KAN, preset MambaVision | ~4 | đo thời gian/epoch trên CPU trước khi đưa vào không gian mặc định |

Mỗi mốc giữ quy tắc của v0.23–v0.24: port theo mã chính thức (ghim commit), so số học trên GitHub Actions, đo cùng ngân sách 3 seed, ghi thắng/thua thật vào `agent.md`.

## Nguồn

- MambaVision: https://github.com/NVlabs/MambaVision ; https://openaccess.thecvf.com/content/CVPR2025/html/Hatamizadeh_MambaVision_A_Hybrid_Mamba-Transformer_Vision_Backbone_CVPR_2025_paper.html
- CPUBone: https://arxiv.org/html/2603.26425 ; ViT-5: https://arxiv.org/html/2602.08071
- MambaHSI+: https://github.com/RockAilab/MambaHSI_Plus ; FAHM: https://www.researchgate.net/publication/388808614_FAHM_Frequency-Aware_Hierarchical_Mamba_for_Hyperspectral_Image_Classification ; HyperSMamba: https://doi.org/10.3390/rs17122008 ; Awesome-Mamba: https://github.com/XiudingCai/Awesome-Mamba-Collection
- MambaHSI: https://arxiv.org/html/2501.04944v1 ; SSFTT & so sánh HSI: https://arxiv.org/pdf/2410.03171 ; SpectralFormer: https://github.com/danfenghong/IEEE_TGRS_SpectralFormer
- TimeMixer: https://github.com/kwuking/TimeMixer ; TimeXer: https://github.com/thuml/TimeXer ; xLSTM-Mixer: https://github.com/mauricekraus/xlstm-mixer ; Time-Series-Library: https://github.com/thuml/Time-Series-Library ; danh sách: https://github.com/TSCenter/awesome-time-series-papers
- Phân loại chuỗi: https://arxiv.org/html/2302.02515v2 ; https://www.alphaxiv.org/abs/2603.01348
- TabArena: https://papers.neurips.cc/paper_files/paper/2025/file/1697e3fb412da11dc9488249f9e7bbc9-Paper-Datasets_and_Benchmarks_Track.pdf ; TabPFN v2: https://arxiv.org/html/2502.17361v1 ; RealMLP: https://arxiv.org/abs/2407.04491 , https://pytabkit.readthedocs.io/ ; survey: https://github.com/LAMDA-Tabular/Tabular-Survey
- DyT: https://arxiv.org/abs/2503.10622 ; Differential Transformer: https://arxiv.org/abs/2410.05258 ; Gated Attention: https://github.com/qiuzh20/gated_attention , https://proceedings.neurips.cc/paper_files/paper/2025/hash/904e89bb4e632e75fb47f093b620b257-Abstract-Conference.html
- Gated DeltaNet: https://arxiv.org/abs/2412.06464 ; TFLA/xLSTM: https://arxiv.org/html/2503.14376v2
- MobileNetV4: https://arxiv.org/html/2404.10518v1 ; LSNet: https://arxiv.org/abs/2503.23135 ; OverLoCK: https://arxiv.org/abs/2502.20087v3 ; WTConv: https://github.com/BGU-CS-VIL/WTConv ; RoPE-ViT: https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/1584_ECCV_2024_paper.php ; Vision-RWKV: https://arxiv.org/pdf/2403.02308
- KAN: https://arxiv.org/pdf/2406.14529 , https://arxiv.org/abs/2411.14904v2 ; KAT: https://arxiv.org/html/2409.10594v1
- Đồ thị: Exphormer https://arxiv.org/pdf/2303.06147 ; OpenGT https://arxiv.org/html/2506.04765v1 ; PE benchmark https://arxiv.org/pdf/2411.12732v2 ; Polynormer https://www.github.com/cornell-zhang/Polynormer ; SGFormer https://arxiv.org/html/2306.10759v5
- Phân đoạn: SegNeXt https://arxiv.org/pdf/2209.08575 ; EfficientViT https://arxiv.org/pdf/2205.14756.pdf
- CTR: DCNv3 https://arxiv.org/html/2407.13349v1 ; ML-DCN https://arxiv.org/html/2602.09194 ; Criteo leaderboard https://hyper.ai/en/sota/tasks/click-through-rate-prediction/benchmark/click-through-rate-prediction-on-criteo

Chưa kiểm chứng trong khảo sát này: repo StarNet (`ma-xu/Rewrite-the-Stars`), DyT (`jiachenzhu/DyT`), Gated DeltaNet (`NVlabs/GatedDeltaNet`), GraphGPS (`rampasek/GraphGPS`), TabR, FuxiCTR, xlstm: tên repo theo hiểu biết của tôi, sẽ xác nhận khi port. Trạng thái "best paper" của Gated Attention có nguồn mâu thuẫn; chắc chắn là oral NeurIPS 2025.
