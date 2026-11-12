# Ví dụ QA/QC: DLNet chạy trọn quy trình trên dữ liệu chuẩn

Mỗi thư mục `<lĩnh vực>/<bộ>/` là một lần dùng DLNet như người dùng thật:

- `input/`: dữ liệu chuẩn, file gốc của nhà phát hành (HSI) hoặc tập con cố định của nó; nguồn và sha256 ở `SOURCE.md`.
- `output/`: nguyên những gì `dlnet design` ghi ra: `leaderboard.md`, và ba gói `top1_*`, `top2_*`, `top3_*`.
  Trong mỗi gói, `model.py` là **chính mạng được sinh ra, PyTorch thuần** (không import DLNet); gói top-1 đã được
  train đầy đủ nên có thêm `model.pt` (trọng số) và `model.pt2` (đồ thị `torch.export`).
- `qc/`: log từng bước, `QC.md`, `qc.json`, `predictions.npy` và hình (bản đồ lớp HSI, ảnh chồng nhãn, ma trận nhầm lẫn).

Chạy lại: `python examples/prepare_data.py --cache <thư mục tải>` rồi `python examples/run_example.py <bộ>|all`
(mặc định `--budget 10m --epochs 30 --threads 2`, CPU). Năm bước được kiểm: (1) design, (2) pytest của từng gói,
(3) train đầy đủ gói top-1, (4) predict + export, (5) chấm lại `predictions.npy` trên tập TEST của gói và so với
số `train.py` in ra, kiểm các tập không trùng nhau.

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

Chưa chạy xong: `image_classification/mnist`, `image_classification/fashion_mnist`, `image_classification/cifar10`, `image_classification/imagenette`, `image_classification/oxford_pets_breeds`, `image_segmentation/isbi2012_em`, `image_segmentation/oxford_pets_trimap`, `image_segmentation/camvid`.
