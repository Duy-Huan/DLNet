# Nguồn bên thứ ba của thư viện khối (`dlnet/blocks.py`)

Không file nào của các repo dưới đây được sao chép nguyên văn vào DLNet. Các khối được viết lại bằng PyTorch thuần (chỉ phụ thuộc `torch`), thu nhỏ cho dữ liệu nhỏ, và ghi nguồn trong `blocks.CATALOG` (xem bằng `python -m dlnet blocks`).

| Repo | Commit đã đọc | Giấy phép | Dùng gì trong DLNet | Cách dùng |
|---|---|---|---|---|
| [rasbt/deeplearning-models](https://github.com/rasbt/deeplearning-models) | `59beda3` | MIT, © 2019–2022 Sebastian Raschka | LeNet-5, all-convolutional net, VGG, ResNet basic block, DenseNet layer/transition, decoder deconvolution của autoencoder | Viết lại theo cấu trúc trong notebook; đổi: số tầng và độ rộng theo kích thước ảnh, thêm BatchNorm cho VGG và all-conv |
| [prescient-design/cortex](https://github.com/prescient-design/cortex) | `d4f6bb7` | Apache-2.0 | Mẫu ghép root → trunk → branch → leaf (`NeuralTree`), `SumTrunk`, mean pooling có mask | Viết lại không dùng hydra/omegaconf; bỏ phần protein, tokenizer, tác vụ ngôn ngữ |
| [ylongqi/openrec](https://github.com/ylongqi/openrec) | `a00de23` | Apache-2.0 | `LatentFactor`, `SecondOrderFeatureInteraction`, `PairwiseLogLoss` (module tf2) | Chuyển từ TensorFlow sang PyTorch |
| [yamsgithub/modular_deep_learning](https://github.com/yamsgithub/modular_deep_learning) | `8e197b4` | **Không có file giấy phép** | Ý tưởng mixture of experts: gate mềm (expectation), top-k | **Không sao chép mã.** Cài lại từ phương pháp; khác bản gốc: trộn logit thay vì trộn rồi softmax |

Các khối viết từ trí nhớ theo paper, chưa đối chiếu mã tác giả: MobileNetV2 (arXiv:1801.04381), squeeze-and-excitation (arXiv:1709.01507), FCN (arXiv:1411.4038), U-Net (arXiv:1505.04597), product of experts (Hinton 2002).

Chưa đưa vào và lý do:
- rasbt: GAN, autoencoder biến phân, RNN/NLP, GNN, ordinal regression, các notebook TensorFlow (ngoài phạm vi "khối xây mạng cho các plugin hiện có"); ResNet-50/101/152 bottleneck, AlexNet, MobileNetV3.
- cortex: toàn bộ phần protein (tokenizer, BLOSUM, acquisition, corruption/diffusion), khối Conv1d/Transformer theo chuỗi token.
- openrec: các recommender hoàn chỉnh (BPR, WRMF, GMF, DLRM, UCML) và toàn bộ nhánh `tf1`; DLNet chưa có plugin recommendation.
- modular_deep_learning: các biến thể huấn luyện (EM, stochastic, expert-loss), thí nghiệm và notebook.

Giấy phép MIT và Apache-2.0 yêu cầu giữ thông báo bản quyền khi phân phối lại mã nguồn gốc; bảng trên là phần ghi công. Nếu sau này sao chép nguyên văn file từ các repo Apache-2.0, phải kèm file LICENSE của họ và ghi rõ thay đổi.
