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
| rasbt | decoder của `ConvolutionalAutoencoder` | decoder `deconv`, `up_deconv` |
| rasbt | `ResNet`, `AlexNet`, `ConvNet` (all-conv) | preset `resnet`, ghép từ `conv_bn_act` |
| cortex | `BidirectionalSelfAttention`, `CausalSelfAttention` | `attn_seq`, `transformer_block` (cờ `causal`) |
| cortex | `TransformerBlock`, `MLP` | `transformer_block`, `ffn` |
| cortex | `SinePosEncoder` | `pos` (`sine`) |
| cortex | `MeanPooling`, `WeightedMeanPooling` | `seq_pool` (`mean`, `attn`) |
| cortex | `Conv1dResidBlock` | `conv1d_resid` |
| openrec | `MLP`, `SecondOrderFeatureInteraction` (tf2) | `mlp`, `fm_interaction` (DLRM = `fm_interaction` + `mlp`) |
| modular_deep_learning | mixture of experts: expectation, top-k; `escort`; expert/gate bất kỳ | combinator `moe` (`soft`, `topk`, `product`), activation `escort` |
| pytorch/vision | `BasicBlock`, `Bottleneck` (ResNeXt, Wide) | `basic_block`, `bottleneck` |
| pytorch/vision | `MBConv`, `FusedMBConv`, `mobilenetv3.InvertedResidual` | `mbconv`, `fused_mbconv` |
| pytorch/vision | `convnext.CNBlock`, `regnet.ResBottleneckBlock` | `convnext_block`, `regnet_block` |
| pytorch/vision | `shufflenetv2.InvertedResidual`, `squeezenet.Fire` | `shuffle_unit`, `fire` |
| pytorch/vision | `SqueezeExcitation`, `StochasticDepth`, `Conv2dNormActivation` | `se`, `res` (`drop_path`), `conv_bn_act` |
| pytorch/vision | Swin `PatchMerging`, window attention (không có relative position bias) | `patch_merge`, `window_attn` |
| pytorch/vision | `ViT` | preset `vit` (`patchify` + `pos` + `transformer_block`) |
| pytorch/vision | `segmentation` ASPP, FPN, FCN | decoder `aspp`, `fpn`, `fcn` (rút gọn) |
| mamba | `Mamba` (`mamba_simple.py`), `selective_scan_ref` | `mamba_block` (quét tuần tự, giới hạn độ dài 256) |
| mamba | `Mamba2Simple`, `ssd_minimal_discrete` | `mamba2_block` (SSD theo khối, không giới hạn độ dài) |

Từ paper, viết từ trí nhớ (không có trong nguồn trên): CBAM, ECA, Inception, MobileNetV1 `dw_sep`, ResNet v2 `preact_block`, MLP-Mixer, GLU, DCN `cross_net`, FT-Transformer `feature_tokens`, U-Net.

## Chưa port và lý do

| Nguồn | Lớp / phần chưa có | Lý do |
|---|---|---|
| rasbt | `AutoEncoder`, `VariationalAutoencoder`, `ConditionalVAE`, `ConvVAE` | cần loss tái tạo/KL và vòng huấn luyện riêng; DLNet chỉ huấn luyện có nhãn |
| rasbt | `GAN`, `DCGAN`, `Flatten`, `Reshape1` | cần huấn luyện đối kháng |
| rasbt | `CoralLayer` | cần tác vụ hồi quy thứ hạng (ordinal) |
| rasbt | `GraphNet` | cần dữ liệu đồ thị; zoo chưa có kiểu `graph` |
| rasbt | `DistilBERTClass` | bọc mô hình HuggingFace pretrained |
| cortex | `root`/`branch`/`leaf`/`trunk` nhiều tác vụ, mask padding, `MaskLayerNorm1d`, `PoolingSelfAttention` | zoo xử lý đầu vào độ dài cố định, một tác vụ; cây nhiều đầu ra mới có ở `dlnet/blocks.py` (bản cũ) |
| cortex | phần protein: tokenizer, BLOSUM, corruption/diffusion, acquisition | ngoài phạm vi |
| openrec | `LatentFactor` và các recommender (BPR, WRMF, GMF, UCML, YouTube, RNN) | cần đầu vào là id; zoo chưa có kiểu `ids` (bản cũ ở `blocks.py`) |
| modular_deep_learning | EM, stochastic, expert-loss; optimizer hai nhiệt độ; attention trên hidden của experts | là biến thể huấn luyện, không phải module |
| pytorch/vision | Inception v3, GoogLeNet phụ trợ, MaxViT, MNASNet, LRASPP, deform conv, DropBlock | chưa làm; detection, video, optical flow, quantization, trọng số pretrained: ngoài phạm vi |
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

**Chưa đối chiếu được** (không có mã tham chiếu cài được ở đây): `window_attn` (torchvision có relative position bias, zoo không), ASPP/FPN/FCN (bản rút gọn), CBAM, ECA, Inception, MLP-Mixer, GLU, DCN, FT-Transformer, U-Net (viết từ paper), `dw_sep` (MobileNetV1). Mamba chỉ đối chiếu đường chạy chậm trên CPU, không có kernel CUDA.

## Các họ cũ đã chuyển thành công thức (M12)

`library.py`, `array_lib.py`, `hsi_lib.py` và `blocks.py` đã bị gỡ; mọi họ nay là công thức sinh spec zoo (`dlnet/zoo/families.py`). Số tham số của công thức so với bản cũ cùng cấu hình (width 16, depth 2): **22/33 họ trùng tuyệt đối**; còn lại lệch ≤ 10% do khác biệt đã biết (`ft_transformer`, `vit`, `ts_transformer`, `spec_transformer`: lớp transformer của zoo không có bias ở LayerNorm/attention theo cortex, −5% đến −7%; `moe_mlp` +10%: mỗi expert không còn head riêng mà một head chung đọc hỗn hợp; `tcn` −0,9%; `fcn` +0,8%; `convnext2d` +0,3%; `densenet`, `hybridsn` ≈ 0%). Các con số trùng được ghim ở `tests/test_cost.py::test_family_param_counts_match_the_earlier_libraries`.
