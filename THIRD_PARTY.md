# Nguồn bên thứ ba của kho module (`dlnet/zoo/`)

Công cụ dùng nội bộ nên **không xét giấy phép** khi lấy module (yêu cầu của chủ dự án). Cột giấy phép chỉ để ghi nhớ nếu sau này phân phối ra ngoài. Mã không được chép nguyên văn từng file: mỗi khối được viết lại thành module **có kiểu vào/ra, tham số hóa và độc lập kích thước đầu vào** để ghép được (`dlnet/zoo/core.py`). Liệt kê toàn bộ: `python -m dlnet zoo`.

| Nguồn | Commit đã đọc | Giấy phép (để nhớ) |
|---|---|---|
| [rasbt/deeplearning-models](https://github.com/rasbt/deeplearning-models) | `59beda3` | MIT |
| [prescient-design/cortex](https://github.com/prescient-design/cortex) | `d4f6bb7` | Apache-2.0 |
| [ylongqi/openrec](https://github.com/ylongqi/openrec) | `a00de23` | Apache-2.0 |
| [yamsgithub/modular_deep_learning](https://github.com/yamsgithub/modular_deep_learning) | `8e197b4` | không có file giấy phép |
| [pytorch/vision](https://github.com/pytorch/vision) (thêm ngoài bốn repo, vì là nguồn khối PyTorch giàu nhất) | `4260301` | BSD-3 |
| [state-spaces/mamba](https://github.com/state-spaces/mamba) (thêm cho khối SSM) | `e9594ce` | Apache-2.0 |
| [huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models) (timm, thư viện khối ảnh chính thống lớn nhất) | PyPI `timm==1.0.31` | Apache-2.0 |
| [qubvel-org/segmentation_models_pytorch](https://github.com/qubvel-org/segmentation_models_pytorch) (decoder phân đoạn) | PyPI `segmentation-models-pytorch==0.6.0` | MIT |
| [Project-MONAI/MONAI](https://github.com/Project-MONAI/MONAI) (ảnh y tế 2-D/3-D) | PyPI `monai==1.6.1` | Apache-2.0 |

## Đã port vào zoo

| Nguồn | Lớp gốc | Op trong zoo |
|---|---|---|
| rasbt | `BasicBlock`, `ResidualBlock` | `basic_block` |
| rasbt | `Bottleneck` | `bottleneck` (có groups và base width như ResNeXt/Wide) |
| rasbt | `_DenseLayer`, `_DenseBlock`, `_Transition` | `dense_block`, `transition` |
| rasbt | `NiN` | `nin_block` |
| rasbt | `FilterResponseNormalization` | `norm` (`frn`) |
| rasbt | `LeNet5`, `VGG16` | `lenet_stage`, `vgg_stage` |
| rasbt | `MLP`, `MultilayerPerceptron`, `MultiLayerPerceptron` | `mlp` |
| rasbt | `RNN` (LSTM), `TextCNN` | `rnn`, `textcnn` |
| rasbt | decoder của `ConvolutionalAutoencoder`; `AutoEncoder`, `ConvVariationalAutoencoder` | decoder `deconv`, `up_deconv`; họ `conv_ae`, `vae` (`vae_bottleneck`: reparameterization, KL) trên plugin `image_reconstruction` |
| rasbt | `DCGAN` (generator, discriminator) | `vec_to_map`, `up_deconv`, `conv_bn_act`; lệnh `dlnet gan` (huấn luyện đối kháng, xếp hạng theo MMD²) |
| rasbt | `GraphNet` (3 notebook: trung bình hàng xóm, cạnh học được chưa port, bộ lọc phổ) | `grid_graph_avg`, `grid_spectral_conv` (đồ thị lưới pixel với kernel Gaussian) |
| rasbt | `ResNet`, `AlexNet`, `ConvNet` (all-conv) | preset `resnet`, ghép từ `conv_bn_act` |
| cortex | `BidirectionalSelfAttention`, `CausalSelfAttention` | `attn_seq`, `transformer_block` (cờ `causal`) |
| cortex | `TransformerBlock`, `MLP` | `transformer_block`, `ffn` |
| cortex | `SinePosEncoder` | `pos` (`sine`) |
| cortex | `MeanPooling`, `WeightedMeanPooling` | `seq_pool` (`mean`, `attn`) |
| cortex | `Conv1dResidBlock` | `conv1d_resid` |
| openrec | `MLP`, `SecondOrderFeatureInteraction` (tf2) | `mlp`, `fm_interaction`, `pair_dots` |
| openrec | `LatentFactor`, `GMF`, DLRM (tf2) | `embed_fields`, `gmf`; họ `mf`, `ncf`, `fm`, `dlrm` trên plugin `recsys` |
| modular_deep_learning | mixture of experts: expectation, top-k; `escort`; expert/gate bất kỳ | combinator `moe` (`soft`, `topk`, `product`), activation `escort` |
| pytorch/vision | `BasicBlock`, `Bottleneck` (ResNeXt, Wide) | `basic_block`, `bottleneck` |
| pytorch/vision | `MBConv`, `FusedMBConv`, `mobilenetv3.InvertedResidual` | `mbconv`, `fused_mbconv` |
| pytorch/vision | `convnext.CNBlock`, `regnet.ResBottleneckBlock` | `convnext_block`, `regnet_block` |
| pytorch/vision | `shufflenetv2.InvertedResidual`, `squeezenet.Fire` | `shuffle_unit`, `fire` |
| pytorch/vision | `SqueezeExcitation`, `StochasticDepth`, `Conv2dNormActivation` | `se`, `res` (`drop_path`), `conv_bn_act` |
| pytorch/vision | Swin `PatchMerging`, `ShiftedWindowAttention` (relative position bias, mask cửa sổ dịch) | `patch_merge`, `window_attn` (trong residual pre-LayerNorm) |
| pytorch/vision | `googlenet.Inception`, `BasicConv2d` | `inception` (tỉ lệ kênh như inception 3a; `k5=3` như torchvision, `k5=5` như paper) |
| pytorch/vision | `ViT` | preset `vit` (`patchify` + `pos` + `transformer_block`) |
| smp | `UnetDecoder`, `UnetPlusPlusDecoder`, `FPNDecoder`, `DeepLabV3Decoder`, `DeepLabV3PlusDecoder` (ASPP, SeparableConv2d), `PSPDecoder`, `LinknetDecoder`, `UPerNetDecoder`, `SegformerDecoder`, `SCSEModule` | decoder `unet`, `unetpp`, `fpn`, `deeplabv3` (tên cũ `aspp`), `deeplabv3plus`, `pspnet`, `linknet`, `upernet`, `segformer` (`dlnet/zoo/dense.py`); `attention: scse` cho unet/unet++; op `scse` |
| timm | `CbamModule`, `EcaModule` | `cbam`, `eca` (`k=0`: kernel tự chọn theo số kênh như timm) |
| timm | `SelectiveKernel`, `SplitAttn`, `GlobalContext`, `CoordAttn`, `GlobalResponseNorm` | `sk_conv`, `split_attn`, `gc_block`, `coord_attn`, `grn` |
| timm | `ghostnet.GhostBottleneck`, `res2net.Bottle2neck`, khối `convmixer`, `byobnet.RepVggBlock` (có `fuse()` gộp ba nhánh thành một conv), `metaformer.MetaFormerBlock` + `Pooling` (PoolFormer v1) | `ghost_bottleneck`, `res2net_block`, `convmixer_block`, `repvgg_block`, `poolformer_block` |
| timm | `mlp_mixer.MixerBlock`, `PatchEmbed`, `DepthwiseSeparableConv`, `resnetv2.PreActBasic` | đối chiếu `mixer_block`, `patchify`, `dw_sep`, `preact_block` có sẵn |
| MONAI | `segresnet_block.ResBlock`, `dynunet_block.UnetResBlock` | `segres_block`, `unet_res_block` (1-D/2-D/3-D) |
| mamba | `Mamba` (`mamba_simple.py`), `selective_scan_ref` | `mamba_block` (quét tuần tự, giới hạn độ dài 256) |
| mamba | `Mamba2Simple`, `ssd_minimal_discrete` | `mamba2_block` (SSD theo khối, không giới hạn độ dài) |

Từ paper, viết từ trí nhớ (không có trong nguồn trên): GCN, GAT, GraphSAGE, GIN (đồ thị), CBAM, ECA, Inception, MobileNetV1 `dw_sep`, ResNet v2 `preact_block`, MLP-Mixer, GLU, DCN `cross_net`, FT-Transformer `feature_tokens`, U-Net.

### Họ mạnh 2022–2025 (v0.23, `dlnet/zoo/ops_modern.py`)

| Nguồn chính thức | Lớp gốc | Op / họ trong DLNet |
|---|---|---|
| `tabm` 0.0.3 (PyPI, tác giả Yandex Research), TabM arXiv:2410.24210 | `TabM` (`arch_type="tabm"`, `MLPBackboneBatchEnsemble`, `LinearBatchEnsemble`, `LinearEnsemble`) | op `tabm`, head `ensemble`, họ `tabm` |
| `rtdl_num_embeddings` 0.0.12 (PyPI), arXiv:2203.05556 | `PeriodicEmbeddings(lite=True)` | op `plr_embed`, họ `tabm_plr` |
| github.com/yuqinie98/PatchTST, arXiv:2211.14730 | `PatchTST_backbone` (RevIN, patching, `TSTiEncoder`, BatchNorm, residual attention) | op `patchtst`, họ `patchtst` |
| github.com/thuml/iTransformer, arXiv:2310.06625 | `Model` (`DataEmbedding_inverted`, `EncoderLayer`, `FullAttention`, LayerNorm) | op `itransformer`, họ `itransformer` |

Khác với gốc (có chủ ý): head do DLNet thêm (PatchTST/iTransformer: trung bình token rồi linear, thay cho head dự báo); TabM mặc định nhỏ hơn bài báo (k = 8–16, rộng 128–256 thay vì 32 và 512) cho vừa ngân sách CPU. Loss của TabM tính riêng cho từng thành viên rồi lấy trung bình, dự đoán là trung bình xác suất, đúng như bài báo.

**Đối chiếu số học** (`scripts/verify_modern.py`, chép trọng số từ mã gốc sang theo tên, so đầu ra ở chế độ eval; chạy trên GitHub Actions bởi `.github/workflows/verify-modern.yml`, [lượt chạy](https://github.com/Duy-Huan/DLNet/actions/runs/43025944848)): **7/7 khớp**.

| Khối | Mã gốc (ghim) | Sai khác lớn nhất |
|---|---|---|
| TabM backbone (k=8 w=64 d=3; k=4 w=32 d=2) | `tabm` 0.0.3 `TabM` | 0 (trùng từng bit) |
| PLR(lite) | `rtdl_num_embeddings` 0.0.12 `PeriodicEmbeddings(lite=True)` | 0 |
| PatchTST (2 cấu hình) | yuqinie98/PatchTST `204c21e`, `PatchTST_backbone` | 0 |
| iTransformer (2 cấu hình) | thuml/iTransformer `c2426e6`, `Model` | 4,8e-7 |

### Miền tần số, FFT/iFFT (v0.24, `dlnet/zoo/ops_freq.py`)

| Nguồn chính thức (commit ghim) | Lớp gốc | Op / họ trong DLNet |
|---|---|---|
| github.com/raoyongming/GFNet `2de48b6`, arXiv:2107.00645 | `gfnet.py`: `GlobalFilter`, `Block`, `Mlp` | op `global_filter` (1/2/3-D, cả `seq`), `gfnet_block`; họ HSI `gfnet2d` |
| github.com/NVlabs/AFNO-transformer `941e924`, arXiv:2111.13587 | `afno/afno2d.py` `AFNO2D`; `classification/afnonet.py` `Block` | op `afno`, `afno_block`; họ HSI `afno2d` |
| github.com/pkumivision/FFC `39dcc99`, NeurIPS 2020 | `model_zoo/ffc.py` `FourierUnit`, `SpectralTransform`, `FFC`, `FFC_BN_ACT`; `ffc_resnet.py` `BasicBlock` | op `ffc_block`; họ HSI `ffc2d` |
| github.com/cfzd/FcaNet `aa5fb63`, arXiv:2012.11879 | `model/layer.py` `MultiSpectralAttentionLayer`, `MultiSpectralDCTLayer`, `get_freq_indices` | op `fca`; họ HSI `fcanet2d` |
| FNO, Li et al. arXiv:2010.08895 (`fourier_2d.py` gốc của zongyi-li không còn trong repo, nay là neuraloperator dùng tensorly) | `SpectralConv1d/2d/3d`, lớp FNO2d | op `spectral_conv`, `fno_block`; họ HSI `fno2d` |
| FNet, Lee-Thorp et al. arXiv:2105.03824 (mã chính thức JAX) | lớp trộn Fourier + encoder post-LN | op `fnet_block` (`seq`) |
| DLNet (không phải port) | GFNet lọc dọc **trục phổ**; hệ số Fourier của phổ từng pixel | op `spectral_gf`, `spectral_rfft`; họ HSI `fourier_ss`, `spec_fft2d` |

Khác với gốc (có chủ ý): mọi khối nhận tensor channels-first của zoo và tự chuyển sang channels-last khi bản gốc dùng (GFNet, AFNO); FFC chỉ có stride 1 và nhận một map (kênh cuối là phần toàn cục); `torch.rfft(normalized=True)` đã bị bỏ khỏi torch nên dùng `torch.fft.rfft2(norm="ortho")` (bằng nhau). FFT chạy ở float32 rồi đổi lại kiểu.

**Đối chiếu số học** (`scripts/verify_freq.py`, chép trọng số và thống kê BatchNorm ngẫu nhiên theo tên, so đầu ra eval; job CI `verify-freq`, [lượt chạy](https://github.com/Duy-Huan/DLNet/actions/runs/43027248014)): **11/11 trùng từng bit** (sai khác 0): GFNet Block ×2, AFNO2D ×3, FFC BasicBlock ×3 (có và không có LFU, tỉ lệ 0,5 và 0,25), FcaNet ×3 (top16, low4, bot8; 5×5, 7×7, 9×9). FNO và FNet không có mã PyTorch chính thức chạy được: kiểm bằng đáp án đóng trong `tests/test_freq.py` (lọc toàn cục = tích chập vòng tính trực tiếp, góc tần số FNO, FNet = phần thực FFT 2 chiều của numpy, cơ sở DCT trực chuẩn).

### v1.0: khối và họ 2023–2025 (`ops_next.py`, `ops_hsi.py`, `ops_domain.py`, `ops_graph2.py`, `ops_seq2.py`)

Mỗi khối viết lại bằng torch thuần theo mã chính thức (commit ghim trong `.github/workflows/verify-modern.yml`), giữ tên tham số để chép trọng số theo tên.

| Nhóm | Nguồn chính thức (commit) | Lớp gốc | Op / họ DLNet |
|---|---|---|---|
| Attention | qiuzh20/gated_attention `f4c2a5f` (NeurIPS 2025) | `Qwen3Attention`, `Qwen3DecoderLayer` (cổng headwise / elementwise) | `gated_attn_block` |
| Attention | microsoft/unilm Diff-Transformer `31c5b90` (ICLR 2025) | `MultiheadDiffAttn` | `diff_attn_block` |
| Chuẩn hóa | jiachenzhu/DyT `aab5dde` (CVPR 2025) | `DynamicTanh` | `dyt` |
| Vị trí | naver-ai/rope-vit `48d8df5` (ECCV 2024) | `RoPEAttention`, `compute_axial_cis`, `compute_mixed_cis` | `rope_vit_block` (axial / mixed) |
| Tích chập | BGU-CS-VIL/WTConv `4a84da2` (ECCV 2024) | `WTConv2d` (db1), `WTConvNeXtBlock` | `wtconv`, `wtconvnext_block` |
| Tích chập | timm 1.0.31 (MobileNetV4, ECCV 2024) | `UniversalInvertedResidual`, `MobileAttention`, `MultiQueryAttention2d` | `uib`, `mobile_mqa` |
| Tích chập | jameslahm/lsnet `cbe737c` (CVPR 2025) | `Block`, `LSConv`, `LKP`, `SKA` (Triton → unfold) | `lsnet_block` |
| Tích chập | ma-xu/Rewrite-the-Stars `c999eb5` (CVPR 2024) | `starnet.Block` | `star_block` |
| Attention tuyến tính | mit-han-lab/efficientvit `de7d773` (ICCV 2023) | `EfficientViTBlock`, `LiteMLA`, `MBConv` | `efficientvit_block` |
| Attention tích chập | Visual-Attention-Network/SegNeXt `c87bcae` (NeurIPS 2022) | `mscan.Block`, `SpatialAttention`, `AttentionModule` | `msca`, `msca_block` |
| HSI | li-yapeng/MambaHSI `a705284` (TGRS 2024) | `BothMamba`, `SpaMamba`, `SpeMamba` | `mambahsi_block`, họ `mambahsi` |
| HSI | RockAilab/MambaHSI_Plus `5ef6282` (TGRS 2025) | `BothMamba`, `ECALayer`, `Spa/SpeMambaProcessor` | `mambahsi_plus_block`, họ `mambahsi_plus` |
| HSI | zgr6010/HSI_SSFTT `994f5d7` (TGRS 2022) | `SSFTTnet` (bỏ lớp cuối) | `ssftt`, họ `ssftt` |
| HSI | zhangxc0105/FAHM `6a3049e` (2025) | `FAHM`, `FGEM`, `RotaryEmbedding` | `fahm`, họ `fahm` |
| Bảng | dholzmueller/pytabkit `d232eec` (NeurIPS 2024) | RealMLP-TD: `PLREmbeddingsLayerCosBias`, `ScaleLayer`, `WeightLayer` (ntk), `BiasLayer`, `ParametricActivationLayer` | `realmlp`, họ `realmlp` |
| Chuỗi | kwuking/TimeMixer `e246105` (ICLR 2024) | `Model.classification`, `PastDecomposableMixing` | `timemixer`, họ `timemixer` |
| Chuỗi | thuml/TimeXer `7601190` (NeurIPS 2024) | `Model.forecast_multi`, `EnEmbedding`, `EncoderLayer` | `timexer`, họ `timexer` |
| Chuỗi | NVlabs/GatedDeltaNet `b53d6d3` (ICLR 2025) | `GatedDeltaNet` + `recurrent_gated_delta_rule_ref` | `gated_deltanet_block`, họ `gated_deltanet` |
| Chuỗi | NX-AI/xlstm `ab22ead` (NeurIPS 2024) | `mLSTMLayer`, `mLSTMCell`, `parallel_stabilized_simple` | `mlstm_block`, họ `xlstm` |
| Đồ thị | rampasek/GraphGPS `2801570` (NeurIPS 2022) | `GPSLayer` (GIN + Transformer), `get_rw_landing_probs` | `gps_layer`, `rwse`, `graph_lin`, họ `gps_net` |
| Đồ thị | cornell-zhang/Polynormer `fc8c276` (ICLR 2024) | `Polynormer`, `GlobalAttn` (GATConv của PyG viết dạng dense) | `polynormer`, họ `polynormer_net` |
| Đồ thị | qitianwu/SGFormer `87d36a9` (NeurIPS 2023) | `SGFormer`, `TransConv`, `models.GCN` | `sgformer`, họ `sgformer_net` |
| CTR | reczoo/FuxiCTR `183da1c` | `FinalMLP` (AAAI 2023), `WuKongLayer` (ICML 2024) | `finalmlp`, `wukong`, họ `finalmlp`, `wukong` |
| CTR | salmon1802/FCN `8eaa864` (DCNv3) | `ExponentialCrossNetwork`, `LinearCrossNetwork`, `MultiHeadFeatureEmbedding` | `dcnv3`, họ `dcnv3` |
| KAN | Adamdad/kat `d254de7` + rational_kat_cu `181bae8` (ICLR 2025) | `KAN`, `KAT_Group` | `gr_kan`, `kat_block`, họ ảnh `kat` |
| Ảnh | NVlabs/MambaVision `7860a50` (CVPR 2025) | `MambaVisionMixer`, `Block` | `mambavision_block`, họ ảnh `mambavision` |

Khác với gốc (có chủ ý, ghi trong docstring từng lớp):
- Kernel CUDA/Triton được thay bằng dạng tham chiếu mà chính repo cung cấp. Cụ thể: SKA của LSNet viết lại bằng unfold; Gated DeltaNet dùng `recurrent_gated_delta_rule_ref`; mLSTM dùng `parallel_stabilized_simple`; Mamba dùng quét tuần tự.
- MambaHSI gốc quét mọi pixel của cả cảnh thành một chuỗi. Trên patch, DLNet quét từng patch.
- Head (lớp cuối) của bản gốc được thay bằng head của DLNet: SSFTT, FAHM, TimeXer (FlattenHead), DCNv3 (logit, Tri-BCE), FinalMLP (đầu ra `out` đặc trưng), WuKong (MLP), Polynormer/SGFormer.
- RealMLP không port hệ số learning rate riêng từng nhóm tham số, khởi tạo phụ thuộc dữ liệu, và lịch dropout/weight decay.
- KAT dùng mẫu số 1 + Σ|b_i|·|x|^i như kernel Triton và paper. Đường torch/CPU của repo dùng x^i có dấu, khác kernel khi x < 0.
- Giữ nguyên các điểm lạ của mã gốc để tính đúng cùng một hàm:
  - MambaVision cộng bias của `dt_proj` hai lần;
  - FAHM cho phép lớp ẩn 0 đơn vị;
  - SSFTT chia attention cho dim^0,5.
- Tham số khai báo nhưng không dùng trong forward thì bỏ:
  - `bn4` của FAHM và BatchNorm của `Channel_Merge`;
  - `out_cross_layer` của TimeMixer và nhánh mùa của block cuối;
  - `D` của Gated DeltaNet thì giữ nhưng không huấn luyện.

**Đối chiếu số học** (5 job CI: `verify-next`, `verify-hsi`, `verify-domain`, `verify-seq2`, cùng `verify-freq`/`verify` cũ; chép trọng số và thống kê BatchNorm ngẫu nhiên theo tên, so đầu ra eval):
- `verify-next` **32/32**:
  - Trùng từng bit: UIB ×4, DyT ×2, Diff Attention ×4, RoPE ×4, WTConv ×3, LSNet ×2, StarNet ×2, EfficientViT ×3, SegNeXt ×2.
  - Bộ lọc Haar khớp pywt db1.
  - Gated Attention ×3 lệch ≤ 4,8e-7; Mobile MQA ×2 lệch ≤ 3,9e-7.
- `verify-hsi` **8/8**, trùng từng bit: MambaHSI ×2, MambaHSI+ ×2, SSFTT ×2, FAHM ×2.
- `verify-domain` **19/19**: RealMLP ×2, TimeMixer ×2, RWSE, GPSLayer ×2, FinalMLP ×2, WuKong ×2, DCNv3 ×2 trùng từng bit; TimeXer ×2 ≤ 8,3e-7; Polynormer ≤ 3,6e-7; SGFormer ≤ 1,2e-7.
- `verify-seq2` **9/9** (run 12): Gated DeltaNet: phép truy hồi trùng từng bit, lớp ×2 ≤ 2,4e-7; mLSTM ×2 trùng từng bit; GR-KAN ×2 ≤ 7,2e-7 (đầu vào x ≥ 0, vì đường torch của repo dùng lũy thừa có dấu ở mẫu số còn kernel Triton dùng |x|, DLNet theo kernel); MambaVision mixer ×2 trùng từng bit (bias của dt cộng hai lần như bản gốc).

**Chưa port**: OverLoCK (CVPR 2025) cần kernel NATTEN và kiến trúc ba mạng con truyền ngữ cảnh cho nhau, không lắp vào ngữ pháp tuần tự; xLSTM-Mixer cần sLSTM (chưa port, có mLSTM); TabR/ModernNCA cần truy hồi kNN trên tập train trong mỗi batch (vòng huấn luyện khác); mô hình pretrained (proxy chặn trọng số).

### v1.4: làm đầy kho (`dlnet/zoo/ops_more.py`)

Rà kho 137 module so với timm, PyTorch Geometric, Time-Series-Library, FuxiCTR và các paper nền; 21 module còn thiếu được viết lại bằng torch thuần theo mã chính thức (commit / phiên bản ghim), đối chiếu số học bằng `scripts/verify_more.py` (job `verify-more`).

| Module của zoo | Nguồn chính thức (ghim) | Paper |
|---|---|---|
| `dlinear` | thuml/Time-Series-Library@4e938a1 `models/DLinear.py` (`encoder`) | Zeng et al., AAAI 2023 |
| `tsmixer_block` | Time-Series-Library `models/TSMixer.py` `ResBlock` | Chen et al., TMLR 2023 |
| `tide_block` | Time-Series-Library `models/TiDE.py` `ResBlock` | Das et al., TMLR 2023 |
| `gatv2` | torch_geometric 2.6.1 `GATv2Conv` (viết lại trên ma trận kề dày) | Brody et al., ICLR 2022 |
| `gcnii` | torch_geometric `GCN2Conv` (trọng số dùng chung) + `examples/gcn2_cora.py` | Chen et al., ICML 2020 |
| `appnp` | torch_geometric `APPNP` sau một MLP hai lớp | Gasteiger et al., ICLR 2019 |
| `sgc` | torch_geometric `SGConv` | Wu et al., ICML 2019 |
| `cin` | reczoo/FuxiCTR@183da1c `CompressedInteractionNet` (bỏ lớp ra) | xDeepFM, Lian et al., KDD 2018 |
| `autoint_attn` | FuxiCTR `model_zoo/AutoInt` `MultiHeadSelfAttention` | Song et al., CIKM 2019 |
| `senet_fields`, `bilinear_fields` | FuxiCTR `SqueezeExcitation`, `BilinearInteractionV2` (so thêm với bản lặp từng cặp `BilinearInteraction`) | FiBiNET, Huang et al., RecSys 2019 |
| `isab`, `pma` | juho-lee/set_transformer@73432c6 `modules.py` | Lee et al., ICML 2019 |
| `attn_pool` | timm 1.0.31 `AttentionPoolLatent` | Beyer et al., arXiv:2205.01580 |
| `inception_next_block` | timm 1.0.31 `inception_next.MetaNeXtBlock` + `InceptionDWConv2d` | Yu et al., CVPR 2024 |
| `focal_block` | timm 1.0.31 `focalnet.FocalNetBlock` + `FocalModulation` | Yang et al., NeurIPS 2022 |
| `non_local` | timm 1.0.31 `NonLocalAttn` | Wang et al., CVPR 2018 |
| `pvt_block` | timm 1.0.31 `pvt_v2.Block` (attention giảm không gian, MLP có depthwise 3x3) | Wang et al., CVMJ 2022 |
| `blur_pool` | timm 1.0.31 `BlurPool2d` | Zhang, ICML 2019 |
| `simam` | ZjjConan/SimAM@ae98f43 `simam_module` | Yang et al., ICML 2021 |
| `gem` | filipradenovic/cnnimageretrieval-pytorch@1e66a41 `GeM` | Radenovic et al., TPAMI 2018 |

**Kết quả đối chiếu** (GitHub Actions `verify-more`, run 43351297903, torch CPU, trọng số chép từ module gốc, chế độ eval): 53/53 phép so khớp. Sai khác lớn nhất bằng 0 (trùng từng bit) với DLinear, TSMixer, TiDE, CIN, AutoInt, FiBiNET (cả bản V2 và bản lặp từng cặp), Set Transformer, và mọi khối timm, SimAM, GeM; các lớp đồ thị viết lại trên ma trận kề dày lệch ≤ 3,6e-7 so với PyG trên danh sách cạnh (GATv2 ≤ 3,6e-7; GCNII, APPNP, SGC ≤ 1,2e-7), là sai số làm tròn float32 do thứ tự cộng khác nhau.

Không port: dropout / drop-path trong các khối (dùng `res` có `drop_path` của zoo), `use_post_norm` của FocalNet, `linear_attn` của PVTv2, cạnh có thuộc tính (`edge_dim`) và `share_weights` của GATv2, `weight2` (không dùng chung) của GCNII, lớp ra `fc` của CIN.

### v1.8: lớp cơ bản và khối hiệu quả (`ops_prim.py`, `ops_lib.py`, `ops_v18.py`)

**Lớp cơ bản** (`dlnet/zoo/ops_prim.py`, tag `primitive`, không được tìm kiếm chọn): bọc chính lớp `torch.nn` hoặc hàm `torch` cùng tên với cùng tham số. Gồm conv / conv transpose (tham số riêng từng trục, padding_mode), BatchNorm / LayerNorm / GroupNorm / InstanceNorm / RMSNorm / LRN, khoảng 25 activation, pooling (max / avg / LP / adaptive), upsample, pixel shuffle, pad, embedding, LSTM / GRU / RNN, MultiheadAttention, scaled dot-product attention, và các phép tensor (reshape, permute, transpose, slice, select, chunk, split, reduce, phép toán với hằng, roll, flip, fft, tham số học được, hằng số). Không có mã bên thứ ba ngoài torch.

| Module DLNet | Nguồn | Giấy phép |
|---|---|---|
| `window_attention`, `patch_merge_pad` (`ops_lib.py`) | torchvision 0.29 `models/swin_transformer.py`: `shifted_window_attention`, `ShiftedWindowAttention` / `V2`, `_patch_merging_pad` | BSD-3-Clause |
| `pconv_block` | timm 1.0.31 `models/fasternet.py` `MLPBlock` + `Partial_conv3` (FasterNet, arXiv:2303.03667) | Apache-2.0 |
| `gated_cnn_block` | timm 1.0.31 `models/mambaout.py` `GatedConvBlock` (MambaOut, arXiv:2405.07992) | Apache-2.0 |
| `mobilevit_block`, `mobilevitv2_block` | timm 1.0.31 `models/mobilevit.py` `MobileVitBlock`, `MobileVitV2Block`, `LinearSelfAttention` (arXiv:2110.02178, 2206.02680) | Apache-2.0 |
| `sdta_block` | timm 1.0.31 `models/edgenext.py` `SplitTransposeBlock`, `CrossCovarianceAttn`, `PositionalEncodingFourier` (EdgeNeXt, arXiv:2206.10589) | Apache-2.0 |

Kiểm chứng: `scripts/verify_v18.py` (job CI `verify-v18`) chép trọng số timm (đã nhiễu) sang khối DLNet theo thứ tự, so đầu ra: **14/14 cấu hình trùng từng bit** (sai khác 0), kể cả kích thước lẻ buộc MobileViT nội suy. `window_attention` được kiểm qua việc dựng lại Swin v1/v2 của torchvision (sai khác 0, `scripts/bench_import.py`).

## Chưa port và lý do

| Nguồn | Lớp / phần chưa có | Lý do |
|---|---|---|
| rasbt | `CoralLayer` | cần tác vụ hồi quy thứ hạng (ordinal) |
| rasbt | `ConditionalVariationalAutoencoder`, `GraphNet` bản cạnh học được (`gnn-basic-edge-1`), `GAN` MLP | CVAE cần đầu vào điều kiện (nhãn) cùng ảnh; bản cạnh học được chưa làm; GAN dùng bản DCGAN tích chập |
| openrec | các recommender BPR, WRMF, UCML, YouTube, RNN và loss cặp (`PairwiseLogLoss`) | cần huấn luyện theo cặp dương/âm; zoo chỉ có hồi quy/phân loại điểm (pointwise). Đã có: `LatentFactor`, GMF, DLRM, FM |
| rasbt | `DistilBERTClass` | bọc mô hình HuggingFace pretrained |
| cortex | `root`/`branch`/`leaf`/`trunk` nhiều tác vụ, mask padding, `MaskLayerNorm1d`, `PoolingSelfAttention` | zoo xử lý đầu vào độ dài cố định, một tác vụ; cây nhiều đầu ra mới có ở `dlnet/blocks.py` (bản cũ) |
| cortex | phần protein: tokenizer, BLOSUM, corruption/diffusion, acquisition | ngoài phạm vi |
| modular_deep_learning | EM, stochastic, expert-loss; optimizer hai nhiệt độ; attention trên hidden của experts | là biến thể huấn luyện, không phải module |
| pytorch/vision | Inception v3, GoogLeNet phụ trợ, MaxViT, MNASNet, LRASPP, deform conv, DropBlock | chưa làm; detection, video, optical flow, quantization, trọng số pretrained: ngoài phạm vi |
| thuml/Time-Series-Library | TimesNet (`TimesBlock`) | chọn chu kỳ bằng FFT **trên cả batch** rồi rẽ nhánh Python (`.cpu().numpy()`): đầu ra của một mẫu phụ thuộc các mẫu khác trong batch và không xuất được bằng `torch.export`; chưa có cách viết tĩnh giữ đúng hành vi gốc |
| timm | MobileViT, MaxViT/CoAtNet, EfficientFormer, EdgeNeXt, CSPNet, GhostNet SE (hard-sigmoid), HaloAttn, BottleneckAttn, LambdaLayer, trọng số pretrained | chưa làm (khối attention 2-D nặng hoặc cần nhiều tầng phụ trợ); trọng số pretrained ngoài phạm vi (DLNet huấn luyện từ đầu) |
| smp | `MAnet`, `PAN`, `DPT`, encoder dùng trọng số ImageNet | MAnet/PAN chưa làm; DPT cần encoder ViT nhiều tầng; encoder của DLNet là các stage của zoo |
| MONAI | UNETR, SwinUNETR, `AttentionUnet` (cổng attention hai đầu vào), mạng 3-D hoàn chỉnh | cổng attention cần hai tensor vào (chưa có trong ngữ pháp một-vào-một-ra); UNETR/SwinUNETR là mạng lớn |
| mamba | Mamba3, kernel CUDA/Triton, cache suy luận, `Block`/`MHA`/`MLP` của gói | cần GPU và kernel riêng; Mamba và Mamba-2 đã port bằng PyTorch thuần (Mamba chạy tuần tự, Mamba-2 theo khối) |

## Kiểm chứng

Hai phép kiểm, đều cần cài tạm repo gốc (không phải phụ thuộc của DLNet):

1. **Số tham số và shape** (`scripts/verify_zoo.py`, torchvision): 14/14 khối khớp; số đã ghim ở `tests/test_zoo.py::GOLDEN`.
2. **Số học thật** (`scripts/verify_numeric.py --mamba <dir> --cortex <dir>`): dựng khối gốc và khối của zoo cùng cấu hình, **chép trọng số** từ khối gốc sang, so đầu ra trên cùng đầu vào ở chế độ eval. **24/24 khối khớp** (sai khác lớn nhất 4,8e-7):

| Khối | Nguồn gốc | Sai khác lớn nhất |
|---|---|---|
| `selective_scan` | `mamba_ssm.selective_scan_ref` | 0 (trùng từng bit) |
| `mamba_block` (2 cấu hình) | `mamba_ssm.modules.mamba_simple.Mamba`, đường chạy chậm | 0 |
| `ssd_minimal` | `mamba_ssm.modules.ssd_minimal.ssd_minimal_discrete` (Mamba-2, Listing 1) | 0 |
| `mamba2_block` (2 cấu hình) | `mamba_ssm.modules.mamba2_simple.Mamba2Simple`, đường không fused; kernel Triton thay bằng SSD tối thiểu của chính repo | ≤ 2,4e-7 |
| `transformer_block` (3 cấu hình, có/không causal, có/không đổi độ rộng) | cortex `TransformerBlock` | ≤ 2,4e-7 |
| `conv1d_resid` (2 cấu hình, có dilation) | cortex `Conv1dResidBlock` | ≤ 2,4e-7 |
| `basic_block`, `bottleneck` (ResNeXt) | torchvision resnet | 0 |
| `convnext_block`, `mbconv`, `fused_mbconv`, `regnet_block`, `shuffle_unit`, `fire`, `se`, `conv_bn_act`, `patch_merge` | torchvision | 0 |
| `mbconv` kiểu MobileNetV3 | torchvision mobilenetv3 | 0 với BatchNorm eps mặc định (torchvision dùng 1e-3) |
| `transformer_block` kiểu ViT (`bias=1`) | torchvision `EncoderBlock` | ≤ 4,8e-7 |

Những điều phép kiểm này **phát hiện và đã sửa**: (a) MobileNetV3 đặt SE trên kênh đã mở rộng; (b) khối transformer của cortex không có bias và nhánh skip của nó cộng MLP vào **đầu vào của khối** chứ không phải `x + attention` như transformer chuẩn: zoo giữ bản chuẩn làm mặc định và có tùy chọn `cortex_skip=1` để tái hiện chính xác; (c) LayerNorm theo kênh của cortex tính phương sai bằng `E[x²]-E[x]²` kẹp ở eps, nên `conv1d_resid` dùng đúng công thức đó thay vì `nn.LayerNorm`.

3. **Thư viện ảnh** (`scripts/verify_vision_libs.py`, cần `timm==1.0.31`, `segmentation-models-pytorch==0.6.0`, `monai==1.6.1`, `torchvision==0.29.2` cài tạm): trọng số ngẫu nhiên (cả thống kê BatchNorm) chép từ khối gốc sang. **39/39 khớp**, sai khác lớn nhất 8,8e-6 (`gc_block`, sai số làm tròn của softmax), còn lại 0:

| Khối DLNet | Khối gốc |
|---|---|
| decoder `unet` (có/không scse), `unetpp`, `fpn`, `deeplabv3`, `deeplabv3plus` (output stride 16), `pspnet`, `linknet`, `upernet`, `segformer` | smp 0.6.0 cùng tên, đặc trưng 5 mức stride 2 |
| `cbam`, `eca` (k tự chọn, k=5), `coord_attn`, `gc_block`, `grn`, `sk_conv`, `split_attn` (stride 1, 2), `ghost_bottleneck` (2 cấu hình), `res2net_block` (2 cấu hình), `repvgg_block` (2 cấu hình), `poolformer_block`, `dw_sep`, `preact_block`, `mixer_block`, `patchify` | timm 1.0.31 |
| `inception`, `window_attn` (dịch 0 và 2) | torchvision 0.29.2 |
| `segres_block`, `unet_res_block` (2-D có stride, 3-D) | MONAI 1.6.1 |
| `scse` | smp |
| `glu` | `torch.nn.functional.glu` |
| `repvgg_block.fuse()` | chính nó trước khi gộp (sai khác 6,7e-6) |

Những điều phép kiểm M14 **phát hiện và đã sửa**: CBAM cũ có bias trong MLP và không có BatchNorm sau conv 7x7; ECA cũ không có kernel tự chọn; Inception cũ chia kênh tùy ý và dùng 5x5 (torchvision dùng 3x3); window attention cũ không có relative position bias và không có mask cho cửa sổ dịch; MLP-Mixer cũ tính độ rộng ẩn của nhánh token theo số token (timm theo số kênh) và eps LayerNorm khác; decoder U-Net, FPN, ASPP cũ là bản rút gọn viết từ trí nhớ (nay là bản smp). Số tham số các họ phân đoạn vì vậy đổi so với v0.12.

**Còn chưa đối chiếu**: DCN (`cross_net`), FT-Transformer (`feature_tokens`), các op đồ thị (`gcn`, `gat`, `sage`, `gin` viết từ paper), decoder `fcn` (kiểu FCN-8s). Mamba chỉ đối chiếu đường chạy chậm trên CPU, không có kernel CUDA.

## Các họ cũ đã chuyển thành công thức (M12)

`library.py`, `array_lib.py`, `hsi_lib.py` và `blocks.py` đã bị gỡ; mọi họ nay là công thức sinh spec zoo (`dlnet/zoo/families.py`). Số tham số của công thức so với bản cũ cùng cấu hình (width 16, depth 2): **22/33 họ trùng tuyệt đối**; còn lại lệch ≤ 10% do khác biệt đã biết (`ft_transformer`, `vit`, `ts_transformer`, `spec_transformer`: lớp transformer của zoo không có bias ở LayerNorm/attention theo cortex, −5% đến −7%; `moe_mlp` +10%: mỗi expert không còn head riêng mà một head chung đọc hỗn hợp; `tcn` −0,9%; `fcn` +0,8%; `convnext2d` +0,3%; `densenet`, `hybridsn` ≈ 0%). Các con số trùng được ghim ở `tests/test_cost.py::test_family_param_counts_match_the_earlier_libraries`.
