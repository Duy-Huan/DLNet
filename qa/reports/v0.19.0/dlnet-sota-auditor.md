# Kiểm định nhóm C (mức hiện đại) — DLNet 0.19.0

## Môi trường và dữ liệu

- Máy: 4 CPU, không GPU, dùng chung với 3 agent kiểm định khác chạy cùng lúc (load trung bình 4–5 trong phần lớn thời gian). Mọi lệnh `dlnet design` dùng `--threads 1 --no-cache`, ngân sách epoch; script tự viết đặt `torch.set_num_threads(1)`.
- Python, torch 2.15.0 (CPU), scikit-learn 1.9.1.
- Bộ dữ liệu thật đã dùng (thư mục `QA_DATA`): `fashion3k.npz` (ảnh xám 28×28, 10 lớp), `cifar10_3k.npz` (32×32×3, 10 lớp), `breast_cancer.csv`, `wine.csv` (bảng, cột đích `y`). Split 60/20/20 phân tầng, seed 0 (cùng split cho DLNet, mạng chuẩn và baseline cổ điển).
- Thư mục làm việc: `QA_WORK/sota/` (script `std_net.py`, `run_*.sh`, log `d_*.log`, `t_*.log`, kết quả `std_results.jsonl`, `std_ablation.jsonl`, `tab_top1_retrain.txt`, `top1_30ep_*.txt`, đầu ra `d_fashion3k/`, `d_cifar10_3k/`, `t_breast_cancer/`, `t_wine/`).
- Tổng thời gian: khoảng 2,5 giờ đồng hồ (vượt kế hoạch 60–90 phút: một lần chạy `fashion3k --budget 300ep` bị dừng sau 30 phút ở giai đoạn `[hparams]` và mất kết quả; các lần `design` ảnh tốn 25–36 phút cho 60–100 epoch, xem phát hiện Q3).
- Có công cụ tìm kiếm web: các nguồn SOTA ghi "(đã kiểm)" là đã tra trên web trong lần kiểm định này; ghi "(theo hiểu biết)" là chưa kiểm nguồn trực tuyến.

## Bảng chấm

| ID | Kết quả | Bằng chứng (lệnh, số đo, file:dòng) |
|---|---|---|
| C1 | MỘT PHẦN | Kho có họ hiện đại cho ảnh (ConvNeXt, EfficientNetV2 FusedMBConv, GRN của ConvNeXt-V2, RepVGG, ViT, MLP-Mixer, Mamba/Mamba-2 khối), phân đoạn (SegFormer, UPerNet, DeepLabv3+, UNet++), bảng (FT-Transformer), chuỗi (TCN, Transformer, Mamba preset). Thiếu họ mạnh 2023–2026 ở: bảng (TabM, mã hóa số PLR, TabPFN), chuỗi (PatchTST, iTransformer), HSI (Mamba HSI, SSFTT), đồ thị (graph transformer), recommender (DCNv2, SASRec/HSTU). `doc/zoo.md`; `dlnet/zoo/families.py:24-62`; `dlnet/zoo/presets.py:25-474`. Bảng chi tiết ở mục C1 dưới. |
| C2 | KHÔNG ĐẠT | `core.Trainer` (`dlnet/core.py:190-299`): AdamW, clip grad 5,0 (`:272`), CrossEntropy **không** label smoothing (`:222`), AMP bf16/fp16 chỉ trên CUDA. `train_full` (`:302-313`): cosine không warmup, giữ checkpoint val tốt nhất. Trong tìm kiếm (`search.run_search`, `dlnet/search.py:173-187`) `fit` gọi không scheduler → LR hằng. Không có augmentation nào (grep `flip|crop|randaug|mixup|cutmix|smoothing` trong `dlnet/` = 0 kết quả), không EMA, không mixup/cutmix, không warmup; stochastic depth chỉ có ở op `res` và chỉ được đột biến ngẫu nhiên 0/0,1 (`dlnet/zoo/space.py:342`), không preset nào bật. Ablation đo được: augmentation + label smoothing tăng test mạng chuẩn +2,3 điểm (Fashion) và **+17,3 điểm** (CIFAR) cùng 30 epoch. |
| C3 | MỘT PHẦN | Đo trên 4 bộ (2 ảnh, 2 bảng), bảng C3 dưới. Cùng số epoch mà top-1 đã dùng: DLNet ≥ mạng chuẩn trên 2/4 bộ (Fashion thắng, CIFAR ngang/hơn nhẹ; 2 bộ bảng thua nhẹ trên test, bão hòa). Ở 30 epoch (công thức xuất `train.py`): CIFAR DLNet thắng (test 0,609 vs 0,573), Fashion thua (0,800 vs 0,852). Top-1 do DLNet giao trên Fashion (4 epoch) **thua** RandomForest (test 0,779 vs 0,831). Đạt ngưỡng "≥ 2/3 bộ" chỉ ở một cách đếm và chênh trên bảng nằm trong nhiễu → MỘT PHẦN. |
| C4 | MỘT PHẦN | Có: SH, Hyperband, aging evolution, ZAP surrogate, zero-cost proxy (gồm ZiCo 2023), AI advisor (LLM xếp hạng/đề xuất spec, `doc/ai-advisor.md`, chưa đo). Thiếu: tìm kiếm dựa mô hình đa độ trung thực (BOHB, DEHB), dùng prior của chuyên gia trong lịch (PriorBand), ngoại suy đường học (LC-PFN, ifBO), ASHA bất đồng bộ cho nhiều GPU, LLM-guided NAS sinh mã có kiểm chứng (EvoPrompting, LLMatic). `doc/algorithms.md` §1–4. |
| C5 | KHÔNG ĐẠT | Không có đường fine-tune từ trọng số có sẵn: không `torch.hub`, `timm.create_model(pretrained=...)`, `weights=` hay nạp checkpoint ngoài (grep trong `dlnet/`; chỉ `load_state_dict` của chính checkpoint tốt nhất `dlnet/core.py:312`). Không TabPFN cho bảng. Chữ "pretrained" chỉ xuất hiện trong mô tả ZAP (`dlnet/surrogate.py:4,16`). Ghi rõ là khoảng trống lớn nhất đối với ảnh/HSI dữ liệu nhỏ. |
| C6 | MỘT PHẦN | Có: mixed precision bf16/fp16 trên CUDA (`dlnet/core.py:172-185`), `channels_last` trên GPU (`:208-211`), `RepVggBlock.fuse()` (`dlnet/zoo/ops_img.py:385`). Thiếu: gói xuất không gọi `fuse()` (grep `.fuse(` trong `dlnet/` = 0 lời gọi), không `torch.compile`, không xuất ONNX/TorchScript/`torch.export`, không lượng tử hóa (PTQ/QAT), không gộp Conv+BN khi suy luận. |

## C1 — Kho kiến trúc so với SOTA 2023–2026

| Miền | DLNet có (họ / preset / op) | Nhóm mạnh hiện nay | Thiếu / nhận xét |
|---|---|---|---|
| Ảnh phân loại | cnn, resnet(+SE, preact, ResNeXt, Wide), densenet, mobilenetv2, efficientnet (FusedMBConv), convnext (+GRN), regnet, repvgg, vit, mixer, mamba2 patch, sknet, ghostnet, res2net | ConvNeXt V2 (arXiv:2301.00808, theo hiểu biết), ViT/DeiT III với công thức mạnh, CNN nhỏ + công thức hiện đại (ResNet strikes back, arXiv:2110.00476, đã kiểm); với dữ liệu nhỏ: fine-tune backbone pretrained (DINOv2 arXiv:2304.07193, theo hiểu biết) | Kiến trúc đủ; khoảng trống là **công thức huấn luyện** (C2) và **pretrained** (C5), không phải kho khối |
| Phân đoạn | UNet, UNet++, FPN, DeepLabv3/v3+, PSPNet, LinkNet, UPerNet, SegFormer | Mask2Former (arXiv:2112.01527), SAM/SAM 2 (arXiv:2304.02643, 2408.00714), nnU-Net cho y tế (theo hiểu biết) | Thiếu decoder kiểu mask-transformer và mọi encoder pretrained |
| Chuỗi thời gian | tcn, gru, ts_transformer; preset mamba, textcnn, resnet1d | PatchTST (arXiv:2211.14730, ICLR 2023, đã kiểm), iTransformer (arXiv:2310.06625, ICLR 2024, đã kiểm) | Thiếu patching + channel independence (có op `patchify` cho `ser` nhưng chưa thành họ), thiếu token theo biến |
| Bảng | mlp, resmlp, tt/tr_resmlp, lowrank, ft_transformer, moe_mlp, fm_mlp, dcn | TabM (arXiv:2410.24210, ICLR 2025, đã kiểm), TabPFN v2 (Nature 2025, đã kiểm), embedding số PLR (arXiv:2203.05556, theo hiểu biết) | Thiếu TabM/BatchEnsemble, embedding số, TabPFN. Đo C3: trên 2 bộ bảng, top-1 không hơn MLP chuẩn hay hồi quy logistic |
| HSI | cnn1d/2d/3d, HybridSN, SSRN, resnet2d, convnext2d, spec_transformer, composed | SSFTT, SpectralFormer; Mamba cho HSI: SS-Mamba (arXiv:2404.18401), S2Mamba (arXiv:2404.18213), MambaHSI (arXiv:2501.04944) (đã kiểm) | Có op `mamba_block` nhưng chưa thành họ HSI; thiếu tokenizer phổ–không gian kiểu SSFTT |
| Đồ thị | GCN, GAT, GraphSAGE, GIN, graph_mlp | GraphGPS (arXiv:2205.12454, theo hiểu biết), GNN có positional encoding | Thiếu graph transformer / PE |
| Recommender | MF, NCF, FM, DLRM, cross_net (DCN v1) | DCNv2 (arXiv:2008.13535, theo hiểu biết), mô hình chuỗi SASRec, HSTU (arXiv:2402.17152, đã kiểm) | Thiếu mô hình tuần tự hành vi người dùng |

## C3 — So sánh đo được

Mạng chuẩn viết tay (`QA_WORK/sota/std_net.py`): ảnh = ResNet-18 bố cục 2-2-2-2 BasicBlock, rộng 16-32-64-128 (≈ 0,70 M tham số); random crop (pad 2 với 28 px, pad 4 với 32 px, reflect) + lật ngang; label smoothing 0,1; AdamW lr 2e-3, wd 0,05; warmup 2 epoch + cosine; batch 128; giữ checkpoint val tốt nhất (như `train_full`). Bảng = MLP 3×256 + BN + dropout 0,1, cùng công thức (batch 64). Split lấy bằng `array_data.build` / `core.prepare` seed 0.

Top-1 của DLNet: `dlnet design --budget {100ep Fashion, 60ep CIFAR (--per-family 1 --zoo 6 --evolve 0), 1500ep bảng} --threads 1 --no-cache --seed 0 --topk 1 --no-proxies`. Top-1 ảnh chỉ được huấn luyện **4 epoch** (`epochs_used: 4` trong `config.yaml`), top-1 bảng 30 epoch.

| Bộ | DLNet top-1 như giao (epoch) val / test | Mạng chuẩn cùng epoch val / test (TB seed) | DLNet `train.py` 30 ep val / test | Mạng chuẩn 30 ep val / test | Baseline cổ điển tốt nhất theo val: val / test |
|---|---|---|---|---|---|
| fashion3k | mobilenetv2 78 k: **0,802 / 0,779** (4 ep, 1 seed) | ResNet 0,755 / 0,751 (4 ep, 2 seed) | 0,857 / 0,800 (seed 0) | **0,875 / 0,852** (2 seed: 0,881/0,854; 0,869/0,849) | RandomForest 0,840 / 0,831 |
| cifar10_3k | cnn 289 k: 0,404 / 0,426 (4 ep) | ResNet 0,399 / 0,419 (4 ep, 2 seed) | **0,622 / 0,609** (seed 0) | 0,591 / 0,573 (2 seed: 0,589/0,574; 0,594/0,572) | HistGB 0,362 / 0,355 |
| breast_cancer | zoo_glu_lowrank_linear 3,6 k: 0,965 / 0,983 (30 ep) | MLP 0,959 / **0,977** (3 seed; test 0,991/0,957/0,983) | 0,965 / 0,962 (3 seed) | 0,959 / 0,977 (MLP, 30 ep, 3 seed) | LogReg 0,947 / **0,991** |
| wine | ft_transformer 17,5 k: 0,971 / 1,000 (30 ep) | MLP **1,000** / 0,974 (3 seed) | 1,000 / 0,966 (3 seed) | 1,000 / 0,974 (MLP, 30 ep, 3 seed) | RandomForest 1,000 / 1,000 |

Huấn luyện lại top-1 bảng bằng `train.py --epochs 30` với 3 seed (`tab_top1_retrain.txt`): breast_cancer val 0,965 (cả 3 seed) / test 0,983, 0,939, 0,965 (TB 0,962); wine val 1,000 / test 0,949, 0,974, 0,974 (TB 0,966). Mạng MLP chuẩn: test TB 0,977 (breast_cancer), 0,974 (wine). Chênh nằm trong nhiễu seed (std test ≈ 0,02) — không kết luận được thắng/thua trên bảng; cả hai đều không hơn hồi quy logistic trên breast_cancer.

Ablation công thức (mạng chuẩn, 30 ep, seed 0, `std_ablation.jsonl`): bỏ augmentation và label smoothing → Fashion val 0,869 / test 0,831 (so với 0,881 / 0,854), CIFAR val 0,430 / test 0,401 (so với 0,589 / 0,574). Tức là trên CIFAR con, ResNet không augmentation sụp 17 điểm vì quá khớp; DLNet hiện huấn luyện mọi ứng viên **không** augmentation.

Đọc kết quả (phân biệt đo và suy luận):
- Đo: ở số epoch mà DLNet thực sự dùng để chọn và báo cáo (4), top-1 ảnh ≥ ResNet chuẩn (Fashion +4,7 val / +2,8 test; CIFAR +0,5 / +0,7, trong nhiễu). Ở 30 epoch, kết quả chia đôi: CIFAR DLNet hơn 3 điểm test, Fashion thua 5 điểm test.
- Đo: với `--budget 100ep` trên Fashion, mô hình DLNet giao (0,779 test) thua RandomForest (0,831) và leaderboard nói đúng điều đó (`[verdict] beats_reference=False`). Chính số epoch quá ít (4) làm top-1 kém; `train.py` 30 epoch đưa nó lên 0,800, vẫn thua RF.
- Suy luận (chưa đo): thêm augmentation + label smoothing vào `Trainer` nhiều khả năng nâng top-1 DLNet trên ảnh màu nhiều điểm, theo ablation ở trên.
- Một seed cho cột DLNet ảnh và một split: không đủ để khái quát; cần ≥ 2 seed và ≥ 2 split.

## Phát hiện

### Nghiêm trọng
- **N1 (C2, C3). Không có data augmentation và label smoothing trong mọi công thức huấn luyện ảnh/HSI.** `dlnet/core.py:222` (CrossEntropyLoss mặc định), không có biến đổi dữ liệu trong `Trainer.fit` (`:251-282`). Ablation: thiếu chúng làm ResNet chuẩn mất 17,3 điểm test trên CIFAR-10 con và 2,3 điểm trên Fashion. Với dữ liệu nhỏ (đúng phạm vi DLNet nhắm tới), đây là sai lệch lớn nhất so với thực hành chuẩn và cũng làm thứ hạng kiến trúc thiên về mạng nhỏ ít quá khớp.

### Quan trọng
- **Q1 (C3). Với ngân sách epoch vừa phải, top-1 chỉ được huấn luyện 4 epoch.** `--budget 100ep` trên 76 ứng viên (Fashion) và `60ep` trên 14 ứng viên (CIFAR) đều dừng ở rung 1 (`[search] rung 1 (4 ep) … time budget reached`, `d_fashion3k.log`, `d_cifar10_3k.log`); `config.yaml` ghi `epochs_used: 4`. Leaderboard vì vậy so các mạng chưa hội tụ; top-1 Fashion thua RandomForest. Cần ước lượng số ứng viên theo ngân sách (ví dụ ≥ η ứng viên tới rung cuối) hoặc cảnh báo khi top-1 < 12 epoch.
- **Q2 (C5). Không có pretrained / foundation model.** Với 1 800 ảnh huấn luyện, fine-tune một backbone pretrained (ví dụ ResNet/ConvNeXt từ timm, DINOv2) thường vượt xa mọi mạng huấn luyện từ đầu (theo hiểu biết, chưa đo ở đây — máy không có trọng số và mạng có thể bị chặn). Với bảng ≤ 10 000 dòng, TabPFN v2 được báo cáo vượt các baseline đã tune (Nature 2025, đã kiểm).
- **Q3 (C4/vận hành). Giai đoạn `[hparams]` và đánh giá chiếm phần lớn thời gian đồng hồ.** `d_fashion3k` (`[budget]`): search 330 s/44 ep, evolution 552 s/36 ep, **hparams 1 242 s/20 ep**, tổng 2 174 s cho "100ep"; `d_cifar10_3k`: hparams 616 s. Ước lượng hparams còn cho val rất thấp (mobilenetv2: searched 0,802, estimated 0,175) và không lần nào được nhận trên 4 bộ đã chạy. Ngân sách epoch không bao được giờ thực (liên quan R7/A5/A9 của nhóm khác).
- **Q4 (C2). Tìm kiếm dùng LR hằng, công thức xuất dùng cosine không warmup.** `dlnet/search.py:183` gọi `fit(...)` không scheduler; `train_full` (`dlnet/core.py:305`) cosine. Xếp hạng trong tìm kiếm và công thức cuối khác nhau; không warmup, không EMA trọng số, không mixup/cutmix, stochastic depth không được preset nào dùng.
- **Q5 (C1). Bảng và chuỗi thời gian thiếu họ mạnh hiện nay** (TabM, PLR embedding, PatchTST, iTransformer). Đo: trên 2 bộ bảng, top-1 DLNet không hơn MLP chuẩn 3×256 hay hồi quy logistic.

### Nên có
- **K1 (C6).** Gói xuất không tự gộp RepVGG (`fuse()` có ở `dlnet/zoo/ops_img.py:385` nhưng không được gọi), không gộp Conv+BN, không xuất ONNX/TorchScript, không `torch.compile`, không lượng tử hóa.
- **K2 (C1).** HSI có op Mamba nhưng chưa có họ Mamba/SSFTT cho HSI; đồ thị thiếu graph transformer; recommender thiếu mô hình tuần tự.
- **K3 (C4).** Thiếu tối ưu đa độ trung thực dựa mô hình (BOHB/DEHB), prior chuyên gia trong lịch (PriorBand), ngoại suy đường học (LC-PFN) — các cách giúp không phải dừng ở rung 1 như Q1.

## Đề xuất nâng cấp

| Đề xuất | Vì sao (bằng chứng) | Nguồn (paper / mã chuẩn) | Công sức | Lợi ích kỳ vọng |
|---|---|---|---|---|
| Công thức huấn luyện theo miền: ảnh = random crop + flip (+ TrivialAugment), label smoothing 0,1, warmup tuyến tính, tùy chọn mixup/cutmix, EMA; HSI = lật/xoay patch; ghi vào `config.yaml` và `train.py` | N1: ablation −17,3 điểm CIFAR, −2,3 điểm Fashion khi thiếu | ResNet strikes back arXiv:2110.00476 (đã kiểm); timm `create_transform`; TrivialAugment arXiv:2103.10158 (theo hiểu biết) | M | +3…+15 điểm trên ảnh nhỏ (ước theo ablation); thứ hạng công bằng hơn cho mạng lớn |
| Dùng cùng scheduler (warmup + cosine theo rung) trong tìm kiếm và công thức xuất | Q4 | Li et al. Hyperband arXiv:1603.06560 (lịch theo tài nguyên) | S | Xếp hạng khớp hơn với huấn luyện đầy đủ (liên quan R6) |
| Chặn ngân sách: số ứng viên tối đa sao cho ≥ 1 ứng viên đi tới rung 30 ep; cảnh báo khi top-1 < 12 ep; cho phép bỏ `[hparams]` hoặc giới hạn theo giờ | Q1, Q3: top-1 4 ep thua RF; hparams 57 % thời gian Fashion | DEHB arXiv:2105.09821 (đã kiểm); PriorBand arXiv:2306.12370 (đã kiểm) | S–M | Mô hình giao ra hội tụ hơn, giờ thực đúng ngân sách |
| Đường fine-tune pretrained (timm/torchvision backbone, DINOv2) như một họ ứng viên ảnh/phân đoạn; TabPFN v2 là baseline/ứng viên cho bảng ≤ 10 k dòng | Q2, C5 KHÔNG ĐẠT | TabPFN v2 Nature 2025 (đã kiểm); DINOv2 arXiv:2304.07193 (theo hiểu biết); timm | M–L | Thường là bước nhảy lớn nhất trên dữ liệu nhỏ |
| Họ bảng TabM + embedding số PLR | Q5: top-1 bảng không hơn MLP chuẩn | TabM arXiv:2410.24210 ICLR 2025 (đã kiểm); Gorishniy et al. arXiv:2203.05556 (theo hiểu biết) | M | Họ DL bảng mạnh nhất hiện công bố |
| Họ chuỗi PatchTST, iTransformer | C1 | arXiv:2211.14730, arXiv:2310.06625 (đã kiểm) | M | Bắt kịp chuẩn chuỗi thời gian |
| Họ HSI Mamba (SS-Mamba/MambaHSI) và SSFTT, dùng `mamba_block` có sẵn | K2 | arXiv:2404.18401, 2501.04944 (đã kiểm) | M | Bắt kịp SOTA HSI 2024–2025 |
| Xuất triển khai: gọi `fuse()` RepVGG + gộp Conv-BN khi xuất, `torch.export`/ONNX, tùy chọn `torch.compile`, PTQ int8 | K1 | PyTorch `torch.onnx`, `torch.ao.quantization`; RepVGG arXiv:2101.03697 | S–M | Suy luận nhanh hơn, dễ triển khai |
| Đo AI advisor như một LLM-guided NAS, hoặc thêm vòng sinh mã có kiểm (kiểu EvoPrompting) | K3, C4 | EvoPrompting arXiv:2302.14838; LLMatic arXiv:2306.01102 (đã kiểm) | M | Mở rộng không gian ngoài kho cố định |

## Giới hạn của lần kiểm định này

- Chỉ CPU, không GPU; máy dùng chung nên số giây không dùng để so sánh tốc độ (chỉ dùng độ chính xác).
- Cột DLNet ảnh: 1 lần chạy `design` (seed 0) và 1 seed cho `train.py` 30 ep; mạng chuẩn ảnh 2 seed; bảng 3 seed. Một split duy nhất. Không đủ để khái quát; chênh trên bảng nằm trong nhiễu.
- Lần `design` CIFAR dùng cấu hình rút gọn (`--per-family 1 --zoo 6 --evolve 0 --budget 60ep`) vì thời gian; lần Fashion `--budget 300ep` bị dừng sau 30 phút ở giai đoạn `[hparams]` (không có kết quả). Không đo HSI, chuỗi thời gian, phân đoạn, đồ thị, recommender cho C3.
- Mạng chuẩn chưa được tune (một bộ lr/wd); ResNet-18 rộng 16 có thể chưa tối ưu cho CIFAR ở 30 epoch (DLNet thắng ở đó). Không đo pretrained / TabPFN (C5) vì không có trọng số sẵn trên máy.
- Nguồn ghi "theo hiểu biết" chưa được kiểm trực tuyến trong lần này.
