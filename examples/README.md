# Ví dụ QA/QC: DLNet chạy trọn quy trình trên dữ liệu chuẩn

Mỗi thư mục `<lĩnh vực>/<bộ>/` là một lần dùng DLNet như người dùng thật:

- `input/`: dữ liệu chuẩn, file gốc của nhà phát hành (HSI) hoặc tập con cố định của nó; nguồn và sha256 ở `SOURCE.md`.
- `output/`: nguyên những gì `dlnet design` ghi ra: `leaderboard.md`, và ba gói `top1_*`, `top2_*`, `top3_*`.
  Trong mỗi gói, `model.py` là **chính mạng được sinh ra, PyTorch thuần** (không import DLNet); gói top-1 đã được
  train đầy đủ nên có thêm `model.pt` (trọng số) và `model.pt2` (đồ thị `torch.export`).
- `qc/`: log từng bước, `QC.md`, `qc.json`, `predictions.npy` và hình (bản đồ lớp HSI, ảnh chồng nhãn, ma trận nhầm lẫn).
- `output/baselines.ipynb`: notebook Jupyter / **Google Colab** để huấn luyện lại, so sánh và dùng các baseline của bộ đó
  (trên Colab: *File > Open notebook > GitHub*, hoặc tải file lên; ô Setup tự clone repo này và vào đúng thư mục).
  Mỗi gói có thêm `architecture.svg`: sơ đồ khối kiểu paper của mạng.

Chạy lại: `python examples/prepare_data.py --cache <thư mục tải>` rồi `python examples/run_example.py <bộ>|all`
(mặc định `--budget 10m --epochs 30 --threads 2`, CPU). Năm bước được kiểm: (1) design, (2) pytest của từng gói,
(3) train đầy đủ gói top-1, (4) predict + export, (5) chấm lại `predictions.npy` trên tập TEST của gói và so với
số `train.py` in ra, kiểm các tập không trùng nhau.

Mạng dựng lại từ paper (thẻ trong `papers/`, lệnh `dlnet paper build`) nằm ở [`papers/`](papers/): `model.py`, sơ đồ khối,
`report.md` ghi mức trung thành so với mã gốc của tác giả (StarNet-S1 và SSFTT: mức B, trùng từng bit).

So sánh với mạng từ paper trong cùng một lượt chạy (`dlnet design ... --papers papers/ssftt.yaml --paper-recipe`):
[`hsi/indian_pines_papers/output/leaderboard.md`](hsi/indian_pines_papers/output/leaderboard.md), mục *Paper baselines*
(Indian Pines, patch 13, PCA 30: SSFTT mức B đạt OA test 0,9772 theo giao thức chung và 0,9810 theo công thức tác giả;
top-1 của DLNet 0,9828 với 40 nghìn tham số so với 153 nghìn; chia pixel nên các số đều cao hơn thực tế).

## Kết quả

Số đo là của ngân sách nhỏ trên CPU (10 phút tìm kiếm, 30 epoch): dùng để kiểm quy trình và hình dung đầu ra,
không phải số tốt nhất đạt được trên các bộ này. Chia pixel của HSI cho OA cao hơn thực tế (patch train/test chồng nhau).

| Bộ | Đặc trưng dữ liệu | 5 bước | Top-1 (tham số) | Chỉ số test | Baseline cổ điển tốt nhất (val) | Thắng baseline |
|---|---|---|---|---|---|---|
| [`hsi/indian_pines`](hsi/indian_pines/qc/QC.md) | AVIRIS, 145x145, 200 băng (0,4-2,5 µm), 20 m/pixel, 16 lớp nông nghiệp, mất cân bằng; chia pixel | 5/5 | `top1_zoo_gfnet_block_spectral_gf` (57,584) | accuracy 0.9848 | random_forest_spectral_spatial 0.9600 | có |
| [`hsi/indian_pines_block`](hsi/indian_pines_block/qc/QC.md) | cùng cảnh, chia theo KHỐI không chồng lấn (số trung thực hơn chia pixel) | 5/5 | `top1_zoo_afno_block` (37,104) | accuracy 0.5037 | random_forest_spectral_spatial 0.2890 | có |
| [`hsi/pavia_university`](hsi/pavia_university/qc/QC.md) | ROSIS, 610x340, 103 băng (0,43-0,86 µm, chỉ VNIR), 1,3 m/pixel, 9 lớp đô thị; 2 % nhãn để train | 5/5 | `top1_spec_fft2d` (65,257) | accuracy 0.9928 | random_forest_spectral_spatial 0.9510 | có |
| [`hsi/salinas`](hsi/salinas/qc/QC.md) | AVIRIS, 512x217, 204 băng, 3,7 m/pixel, 16 lớp cây trồng; 2 % nhãn để train | 5/5 | `top1_afno2d` (89,744) | accuracy 0.9941 | random_forest_spectral_spatial 0.9769 | có |
| [`hsi/ksc`](hsi/ksc/qc/QC.md) | AVIRIS, 512x614, 176 băng, 18 m/pixel, 13 lớp đầm lầy (Kennedy Space Center) | 5/5 | `top1_mambahsi` (24,145) | accuracy 0.9410 | random_forest_spectral_spatial 0.9673 | không |
| [`hsi/botswana`](hsi/botswana/qc/QC.md) | Hyperion (vệ tinh), 1476x256, 145 băng, 30 m/pixel, 14 lớp (Okavango) | 5/5 | `top1_spec_fft2d` (28,302) | accuracy 0.9919 | random_forest_spectral_spatial 0.9908 | có |
| [`image_classification/mnist`](image_classification/mnist/qc/QC.md) | chữ số viết tay, xám 28x28, 10 lớp (600 ảnh/lớp) | 5/5 | `top1_cnn` (72,666) | accuracy 0.9883 | hist_gradient_boosting 0.9325 | có |
| [`image_classification/fashion_mnist`](image_classification/fashion_mnist/qc/QC.md) | quần áo, xám 28x28, 10 lớp (600 ảnh/lớp) | 5/5 | `top1_zoo_unet_res_block` (171,562) | accuracy 0.8750 | hist_gradient_boosting 0.8292 | có |
| [`image_classification/cifar10`](image_classification/cifar10/qc/QC.md) | ảnh tự nhiên màu 32x32, 10 lớp (600 ảnh/lớp) | 5/5 | `top1_cnn` (72,954) | accuracy 0.6475 | hist_gradient_boosting 0.4033 | có |
| [`image_classification/imagenette`](image_classification/imagenette/qc/QC.md) | 10 lớp ImageNet, màu, thu về 64x64 (300 ảnh/lớp) | 5/5 | `top1_vgg` (26,682) | accuracy 0.6067 | random_forest 0.4183 | có |
| [`image_classification/oxford_pets_breeds`](image_classification/oxford_pets_breeds/qc/QC.md) | 37 giống chó mèo (phân loại tinh), màu 64x64 (100 ảnh/giống) | 5/5 | `top1_zoo_gfnet_block_patchify_pos` (482,549) | accuracy 0.1662 | random_forest 0.1243 | không |
| [`image_segmentation/isbi2012_em`](image_segmentation/isbi2012_em/qc/QC.md) | ảnh hiển vi điện tử xám 256x256, nhị phân màng/tế bào, 30 ảnh | 5/5 | `top1_deeplabv3plus` (9,950) | miou 0.3953 | pixel_random_forest_5x5 0.6555 | không |
| [`image_segmentation/oxford_pets_trimap`](image_segmentation/oxford_pets_trimap/qc/QC.md) | thú cưng/nền/viền, màu 96x96, 1480 ảnh | 5/5 | `top1_deconvnet` (22,323) | miou 0.5795 | pixel_random_forest_5x5 0.3560 | có |
| [`image_segmentation/camvid`](image_segmentation/camvid/qc/QC.md) | cảnh đường phố, màu 120x90, 11 lớp (SegNet), 701 ảnh, có vùng bỏ qua | 5/5 | `top1_fcn` (20,707) | miou 0.3532 | pixel_random_forest_5x5 0.2662 | không |



## Lỗi của DLNet mà các lượt chạy này tìm ra (đã sửa, có regression test)

1. Huấn luyện các họ Mamba / Gated DeltaNet chậm bất thường: backward O(L²) của vòng quét (v1.0.0).
2. `--budget 90s` chạy hơn 12 phút: ứng viên chậm được định giá theo FLOPs (v1.0.0).
3. `model.py` không chứa mạng mà chỉ gọi thư mục `zoo/`: nay là mã PyTorch thuần của chính mạng (v1.0.0).
4. `model.pt2` kèm cả tập dữ liệu (74 MB cho mô hình 73 nghìn tham số) (v1.0.1).
5. Phân đoạn ảnh lớn làm hỏng lượt chạy hoặc hết bộ nhớ: mọi ứng viên bị dựng sẵn (v1.0.2).
6. 37 lớp của Oxford-IIIT Pet bị hiểu là bài hồi quy, không cảnh báo (v1.0.2).

Chi tiết: `agent.md` §9.10. Điều các lượt chạy cho thấy mà CHƯA sửa: với 10 phút CPU, ảnh từ 64x64 trở lên
chỉ kịp huấn luyện 1-2 ứng viên (số gói xuất ra ít hơn 3, DLNet có cảnh báo), và kết quả khi đó có thể thua
baseline cổ điển (cột cuối); `predict.py` ghi nhãn kiểu int64 (file lớn hơn cần thiết).
