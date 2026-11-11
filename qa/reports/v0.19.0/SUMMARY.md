# Kiểm định chất lượng DLNet 0.19.0 — báo cáo tổng hợp

Đội 4 agent kiểm định theo [`qa/CHECKLIST.md`](../../CHECKLIST.md). Báo cáo chi tiết:
[độ tin cậy](dlnet-reliability-auditor.md) ·
[mức hiện đại](dlnet-sota-auditor.md) ·
[tính khả thi](dlnet-feasibility-auditor.md) ·
[thuật toán](dlnet-algorithm-auditor.md).

**Điều kiện đo.**
- Máy 4 CPU, không GPU, dùng chung giữa 4 agent; mọi lệnh chạy `--threads 1` với ngân sách epoch (chủ yếu 60–400 epoch).
- Dữ liệu thật: MNIST, Fashion-MNIST, CIFAR-10 (tập con 3k), digits, breast_cancer, wine, diabetes, Indian Pines.
- Phân đoạn, đồ thị và recommender dùng dữ liệu tổng hợp.
- Kết luận chỉ áp dụng cho **ngân sách nhỏ trên CPU**; ngân sách lớn và GPU chưa được đo.

**Điều phối viên đã tự chạy lại để xác nhận** các phát hiện nghiêm trọng:
- Hyperband sập qua CLI chuẩn.
- Xếp hạng cuối trộn các mô hình huấn luyện số epoch khác nhau (`cli.py`, `results = results + res`).
- Không có cô lập lỗi ứng viên: `search.py` không bắt ngoại lệ khi huấn luyện.
- Vòng huấn luyện không có augmentation, label smoothing, warmup hay EMA.
- Nhãn `best-small` sai (`search.py:347`).
- Gói xuất ra không lưu trọng số.

## 1. Bảng chấm

| Nhóm | ĐẠT | MỘT PHẦN | KHÔNG ĐẠT |
|---|---|---|---|
| R. Độ tin cậy | R3 | R1, R2, R5, R7, R8, R10 | R4, R6, R9, R11 |
| C. Mức hiện đại | — | C1, C3, C4, C6 | C2, C5 |
| F. Tính khả thi | F2, F8 | F1, F3, F5, F6, F7 | F4 |
| A. Thuật toán | A6, A7, A8 | A1, A3 | A2, A4, A5, A9 |
| **Tổng (34)** | **6** | **17** | **11** |

**Đánh giá chung:**
- **Phần làm tốt:**
  - DLNet chạy được trên cả 8 loại dữ liệu.
  - Kết quả tái lập được.
  - Không thấy rò rỉ trong các bước fit.
  - Kết luận "không thắng baseline" là trung thực.
  - MCP và advisor hoạt động.
  - Các công thức thuật toán đúng với paper.
- **Điểm yếu:** tầng "ra quyết định" chưa đáng tin ở ngân sách thực tế. Thứ hạng gần như là nhiễu, top-1 thường chỉ được huấn luyện 4 epoch, và nhiều cơ chế "thông minh" chưa cho lợi ích đo được: chọn hạt giống, tiến hóa, ước lượng hyperparameter.
- **Kiến trúc** đủ hiện đại cho ảnh. **Công thức huấn luyện** thì lạc hậu, và chưa có mô hình pretrained.

## 2. Những phần chưa tốt

### Nghiêm trọng: kết quả sai hoặc sập

| # | Vấn đề | Bằng chứng |
|---|---|---|
| S1 | `--scheduler hyperband` sập (`KeyError: 'val0'`) khi đồng hồ ngân sách chạm đúng hạn: một ngoặc mới vẫn được mở rồi không huấn luyện ứng viên nào | `search.py:275` dùng `>` thay vì `>=`, lỗi ở `search.py:305`. Agent gặp ở 5/5 lần; điều phối viên tái hiện qua CLI chuẩn với `mnist14 --budget 150ep --seeds 16 --scheduler hyperband --evolve 0` |
| S2 | Một ứng viên lỗi làm sập cả lần chạy; ứng viên cho val `NaN` làm hỏng phép sắp xếp và đẩy sai top-1 | Ngoại lệ trong `fit` → thoát mã 1, không có leaderboard. Trên diabetes, một ứng viên NaN đứng hạng 2, và top-1 (−54,47) kém hạng 3 (−52,12) |
| S3 | Xếp hạng cuối trộn mô hình 4 epoch với mô hình 12 epoch (thế hệ đầu với tiến hóa); mặt Pareto cũng vậy; leaderboard không có cột epoch | `cli.py` nối `results + res`; gặp trên fashion3k, mnist14, digits |
| S4 | "+/- 0.0" khi chỉ có 1 lần chạy, nhưng không cảnh báo. Nhóm "không phân biệt được" khi đó tách mọi khác biệt, kể cả chênh 1 mẫu val. Vai trò `most-stable` được gán dựa trên std 0 | Ở 150–400 epoch, mọi finalist chỉ có 1 lần chạy; `search.py:331` |
| S5 | Nhãn `best-small` có thể gán cho mô hình **lớn hơn** top-1 | `search.py:347`; gặp ở 3/10 leaderboard (mnist3k: 171k so với 72k tham số) |
| S6 | Cột số có lẫn chuỗi bị one-hot thành 133 cột mà không cảnh báo; test 0,54 thay vì khoảng 0,95 | `core.py:34` |

### Quan trọng: giảm chất lượng hoặc khả năng dùng

| # | Vấn đề | Bằng chứng |
|---|---|---|
| Q1 | **Thứ hạng gần như là nhiễu.** Top-1 nằm trong top-3 của cả hai seed khác chỉ ở 1/5 bộ. Spearman giữa xếp hạng lúc tìm kiếm và kết quả huấn luyện lại từ −0,53 đến 0,20 | R4, R6 |
| Q2 | **Ngân sách bị dàn mỏng.** Ở 150 epoch, không ứng viên nào chạm rung 30; top-1 có 4–12 epoch. Mặc định có 76 ứng viên ảnh, nhưng chỉ 8 được huấn luyện ở 80 epoch | A9, F6, C3 |
| Q3 | **Ước lượng hyperparameter không giúp mà tốn kém.** 0/15 đề xuất được nhận; giai đoạn này chiếm 39–57 % giờ thực. lr đề xuất cao gấp 10–20 lần lr đã tìm. LR range test và đo nhiễu gradient không bị tính vào ngân sách epoch (fashion3k 150ep chạy 1744 s) | A5, R7 |
| Q4 | **Chọn hạt giống không hơn ngẫu nhiên.** Prior viết tay có Spearman **âm** (−0,25 đến −0,37) nhưng vẫn có trọng số 1. Tổ hợp prior + ZAP mà `design` thật sự dùng (+0,54) kém ZAP đơn thuần (+0,66) | A2, A6, A7 |
| Q5 | **Tiến hóa thua trên test 5/5 cặp** (trung bình −0,019) ở ngân sách nhỏ; Hyperband thua SH (−0,015 test) | A3, A4 |
| Q6 | **Công thức huấn luyện lạc hậu**: không augmentation, label smoothing, warmup, EMA hay mixup. Ablation trên mạng chuẩn: bỏ augmentation và label smoothing làm test giảm 2,3 điểm (Fashion) và **17,3 điểm (CIFAR)** | C2 |
| Q7 | **Gói đầu ra chưa dùng được ngay**: không lưu trọng số, thống kê chuẩn hóa hay danh sách lớp; không có `predict.py`; `train.py` chỉ chạy khi đứng trong thư mục gói | F5 |
| Q8 | **Thông báo lỗi**: chỉ 8/16 lỗi thường gặp chỉ được cách sửa; 10/26 ca đầu vào xấu cho traceback hoặc thông báo sai hướng (`.npz` sai khóa bị chuyển nhầm sang plugin HSI) | F4, R8 |
| Q9 | **HSI `--split pixel` mặc định** có 99,6 % pixel val/test nằm trong patch của pixel train (rò rỉ không gian) mà không cảnh báo. Quy tắc "thắng" không có ngưỡng nhiễu trên test | R1, R2 |
| Q10 | **Không có pretrained hay foundation model** (timm/DINOv2, TabPFN) | C5 |

### Nên có
- Tài liệu cài đặt dẫn người dùng CPU tải torch CUDA nhiều GB, và ghi phiên bản Python không thống nhất (F1).
- 7 chỗ tài liệu lệch với mã (F3).
- Leaderboard không nói bao nhiêu ứng viên thực sự được huấn luyện (F7).
- Thiếu các họ mạnh 2023–2025: TabM/PLR cho bảng, PatchTST/iTransformer cho chuỗi, Mamba-HSI/SSFTT cho HSI, graph transformer, DCNv2/SASRec (C1).
- Chưa có ONNX, `torch.compile` hay lượng tử hóa; gói xuất không tự gộp RepVGG (C6).
- Hai sai lệch so với paper chưa ghi lại: lr của LR range test và rung cuối 30 so với 36 (A1).

## 3. Lộ trình nâng cấp đề xuất (thứ tự ưu tiên)

Công sức: S ≈ vài giờ, M ≈ 1–2 ngày, L ≈ nhiều ngày.

### Mốc 1 — Sửa kết quả sai (P0, mỗi mục S, regression test trước)
1. Sửa lỗi sập của Hyperband (S1); thêm test với ngân sách epoch chạm đúng hạn.
2. Cô lập lỗi ứng viên: bắt ngoại lệ và val không hữu hạn, loại ứng viên đó và ghi vào leaderboard (S2).
3. **Chỉ xếp hạng cùng số epoch**: đưa finalist lên cùng rung trước khi xếp, hoặc xếp theo nhóm epoch; thêm cột epoch (S3). Theo nguyên tắc của successive halving (arXiv:1502.07943).
4. Nhiễu trung thực khi chỉ có 1 lần chạy: ước lượng nhiễu nhị thức của accuracy trên tập val (`sqrt(p(1−p)/n_val)`) thay cho std 0; thêm cảnh báo; không gán `most-stable`; sửa `best-small` (so với top-1); verdict có ngưỡng nhiễu trên test (S4, S5, Q9).
5. Cảnh báo khi cột số bị one-hot do lẫn chuỗi; cảnh báo HSI chia pixel (S6, Q9).

### Mốc 2 — Dùng ngân sách cho đúng chỗ (P1, S–M)
6. **Pool ứng viên theo ngân sách**: chọn số ứng viên sao cho finalist chạm ít nhất rung 2–3. Báo "x/N ứng viên được huấn luyện, top-1 có k epoch" và cảnh báo khi k < 12.
7. Giai đoạn hyperparameter: tắt mặc định ở ngân sách nhỏ, chỉ nhận đề xuất khi có ≥ 2 seed, dùng lr ≈ cực tiểu/10 (Smith, arXiv:1506.01186), và tính cả thời gian ước lượng vào ngân sách.
8. Chọn hạt giống: bỏ prior viết tay khỏi Borda (hoặc học trọng số trên LODO); dùng ZAP + kNN (arXiv:2206.08476).
9. Tiến hóa và Hyperband: chỉ bật khi ngân sách đủ; về lâu dài thay bằng ASHA (arXiv:1810.05934).

### Mốc 3 — Mạng đầu ra tốt hơn (P1, M)
10. **Công thức huấn luyện hiện đại theo miền**:
    - Ảnh: random crop, flip, label smoothing, warmup + cosine, EMA, tùy chọn mixup/cutmix; với phân đoạn, augmentation hình học áp cùng lúc cho ảnh và mask.
    - Chuỗi: jitter, scaling.
    - Bảng: label smoothing hoặc dropout.

    Đo trước và sau bằng chính đội kiểm định (C3).
11. **Gói đầu ra dùng được ngay**: `model.pt`, `preprocess.json` (thống kê chuẩn hóa, lớp, cột), `predict.py`, `train.py` chạy từ mọi thư mục, gộp RepVGG/Conv-BN khi xuất, ONNX.
12. **Thông báo lỗi và kiểm tra đầu vào sớm** cho 16 + 26 ca mà đội đã liệt kê.

### Mốc 4 — Cutting edge (P2, M–L)
13. **Pretrained / foundation model**: fine-tune từ timm hoặc DINOv2 cho ảnh; TabPFN v2 (Nature 2025) làm baseline và ứng viên cho bảng nhỏ.
14. **Họ mới**: TabM + embedding PLR (arXiv:2410.24210); PatchTST (arXiv:2211.14730), iTransformer (arXiv:2310.06625); Mamba cho HSI (arXiv:2404.18401); graph transformer; DCNv2, SASRec.
15. **Tìm kiếm dựa mô hình, đa độ trung thực**: DEHB (arXiv:2105.09821) hoặc BOHB (arXiv:1807.01774), PriorBand (arXiv:2306.12370), ngoại suy đường học (LC-PFN), ASHA song song khi có GPU.
16. **Đo trên GPU** khi có máy: tốc độ, bf16 so với fp32, VRAM ước lượng so với thực tế.

Mỗi mốc kết thúc bằng một lần chạy lại đội kiểm định; tiêu chí xong là các ID tương ứng chuyển sang ĐẠT.
