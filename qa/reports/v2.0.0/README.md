# v2.0.0: kiểm định plugin `hsi_sr` (siêu phân giải ảnh siêu phổ) trên cảnh thật

Máy: CPU 4 luồng, không GPU. Mỗi lần `design`: `--budget 10m --topk 3`, còn lại mặc định (`--sr-patch 32`, `--degrade bicubic`, `--zoo 12`). Một cube mỗi lần, nên chia theo tile (test = pixel chưa thấy của cùng cảnh). PSNR = MPSNR (trung bình theo băng, đỉnh 1), SAM tính bằng độ.

## Kết quả tìm kiếm (10 phút, 30 epoch)

| Cảnh, hệ số | Tile train / val / test | Top-1 | Test PSNR / SSIM / SAM / ERGAS | Bicubic (test) | Verdict |
|---|---|---|---|---|---|
| Salinas 512×217×204, ×2 | 58 / 19 / 19 | `vdsr` 322k | 48,837 / 0,987 / 0,863 / 2,267 | 48,837 / 0,987 / 0,863 / 2,267 | không thắng (bằng) |
| Salinas, ×4 | 58 / 19 / 19 | `sspsr` 1,09M | 43,549 / 0,961 / 1,472 / 2,031 | 43,279 / 0,959 / 1,486 / 2,094 | +0,27 dB, dưới nhiễu 1,41 dB |
| Indian Pines 145×145×200, ×2 (`--sr-patch 16`) | 49 / 16 / 16 | `srcnn` 635k | 45,397 / 0,951 / 1,601 / 1,567 | 45,391 / 0,951 / 1,600 / 1,566 | không thắng (bằng) |

Nội suy bilinear và nearest kém bicubic 0,7–2,6 dB ở mọi lần chạy.

## Huấn luyện đầy đủ gói top-1 (`train.py --epochs 100`)

| Cảnh, hệ số | Mạng | Test PSNR | Bicubic | Chênh |
|---|---|---|---|---|
| Salinas ×4 | `sspsr` | 43,829 (SAM 1,433, ERGAS 1,968) | 43,279 | **+0,55 dB** |
| Salinas ×2 | `vdsr` | 48,837 | 48,837 | 0 (checkpoint tốt nhất theo validation là điểm xuất phát) |

## Khởi tạo head bằng 0

Lần chạy đầu (conv cuối của head khởi tạo ngẫu nhiên) cho top-1 **dưới** bicubic 0,1–0,3 dB ở cả ba cảnh. Đo riêng trên Salinas ×2 (EDSR rộng 32, sâu 4, lr 1e-3, test):

| Epoch | 10 | 20 | 30 | 40 | 50 | 60 |
|---|---|---|---|---|---|---|
| ngẫu nhiên | −8,04 | −4,72 | −2,11 | −1,46 | −1,17 | −0,89 |
| bằng 0 | −0,22 | +0,15 | +0,30 | +0,44 | +0,53 | +0,59 |

Từ v2.0.0 head `srN` khởi tạo bằng 0: mạng chưa huấn luyện chính là nội suy bicubic.

## Kết luận

- Luồng chạy trọn vẹn trên dữ liệu thật: `design` → leaderboard → gói `train.py` / `predict.py` / notebook.
- Với một cảnh và 10 phút CPU, bicubic là đối thủ rất khó: mạng chỉ bằng hoặc hơn vài phần mười dB. Nhiễu lấy mẫu với 16–19 tile validation là ~1,4 dB, nên verdict không xác nhận được mức hơn nhỏ như vậy.
- Muốn mạng hơn bicubic rõ: nhiều cảnh (`hr` dạng (N, H, W, bands); từ 5 cảnh DLNet chia theo cảnh và có nhiều tile hơn), ngân sách lớn hơn hoặc GPU, và huấn luyện đầy đủ gói top-1 (100+ epoch).
