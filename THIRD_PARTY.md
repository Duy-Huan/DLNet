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
| mamba | Mamba2/SSD, Mamba3, kernel CUDA/Triton, cache suy luận | cần GPU và kernel riêng; bản port chạy tuần tự |

## Kiểm chứng

`PYTHONPATH=<torchvision> python scripts/verify_zoo.py` (torchvision chỉ cài tạm, không phải phụ thuộc) so số tham số và shape đầu ra của 14 khối đã port với khối gốc cùng cấu hình: **14/14 khớp** (BasicBlock, Bottleneck của ResNet-50, ResNeXt-50 32x4d và Wide-ResNet-50-2, ConvNeXt CNBlock, EfficientNet MBConv và FusedMBConv, MobileNetV3 InvertedResidual, RegNet, ShuffleNetV2 stride 1 và 2, SqueezeNet Fire, SqueezeExcitation, Conv2dNormActivation). Các con số đã ghim vào `tests/test_zoo.py::GOLDEN`. Lần kiểm đầu tiên phát hiện MobileNetV3 đặt SE trên kênh đã mở rộng (khác EfficientNet): đã sửa.

Giới hạn của phép kiểm: khớp số tham số và shape nghĩa là cấu trúc giống, không chứng minh phép tính giống (trọng số không được chép sang, chưa so đầu ra số học). Mamba được kiểm bằng hồi quy tay (`test_mamba_selective_scan_matches_a_manual_recurrence`), chưa so với `mamba_ssm`. Các khối từ paper chưa đối chiếu mã tác giả.

## `dlnet/blocks.py` (bản M10)

Thư viện khối đầu tiên (backbone/decoder/attention/MoE cho các họ cũ). Vẫn được các họ cũ của plugin ảnh dùng; mọi phần trùng đã có bản tổng quát hơn trong zoo. Sẽ gỡ khi các họ cũ chuyển hẳn sang preset của zoo.
