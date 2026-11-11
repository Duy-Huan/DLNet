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

**Đối chiếu số học** (`scripts/verify_modern.py`, chép trọng số từ mã gốc sang, so đầu ra ở chế độ eval): xem mục Kiểm chứng.

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
