# v1.5.0: xếp hạng các vòng successive halving bằng TSE

Câu hỏi: khi successive halving cắt ứng viên sau vòng đầu, xếp hạng bằng điểm validation (mặc định)
hay bằng tốc độ giảm loss huấn luyện (TSE, Ru et al., NeurIPS 2021, mã gốc rubinxin/TSE@5f55142)
thì giữ được mạng tốt hơn?

## 1. Tương quan sớm với điểm sau 6 epoch

`rung_signal_study.py` (dữ liệu thô) và `rung_signal_analyse.py` (phân tích), kết quả ở
`rung_signal_study.txt`. 7 lượt (mnist3k ×2, fashion3k ×2, cifar10_3k, textures, shapes), mỗi lượt
24 ứng viên lấy ngẫu nhiên từ kho. Mục tiêu: điểm validation sau epoch 6.

| Tín hiệu | Spearman | Giữ được mạng tốt nhất (giữ 1/3) | Regret khi giữ 1/3 |
|---|---|---|---|
| val sau 1 epoch | 0,44 | 3/7 | 0,0059 |
| TSE sau 1 epoch | 0,60 | 4/7 | 0,0099 |
| val sau 2 epoch | 0,70 | 5/7 | 0,0029 |
| val+TSE (Borda) sau 2 epoch | 0,72 | 5/7 | 0,0046 |
| proxy zero-cost tốt nhất (synflow) | 0,51 | 4/7 | 0,0066 |

TSE xếp **cả nhóm** đúng hơn val ở cùng số epoch. Nhưng val giữ **phần đầu** tốt hơn: regret thấp
nhất là val sau 2 epoch. Successive halving chỉ cần giữ đúng phần đầu.

## 2. Đầu-cuối: `dlnet design` với `--rung-score val` và `val+tse`

`scripts/bench_rungs.py`, kết quả ở `rung_score_bench.json`. Ngân sách 150 epoch, `--zoo 40`,
`--threads 4`, seed 0, CPU. Ba bộ dữ liệu, mỗi bộ một lần chạy cho mỗi cách.

| Dữ liệu | Cách | Top-1 | Val | Test |
|---|---|---|---|---|
| fashion3k | val | mobilenetv2 | 0,8538 | 0,8377 |
| fashion3k | val+tse | mobilenetv2 | 0,8538 | 0,8377 |
| cifar10_3k | val | mobilenetv2 | 0,6166 | 0,6077 |
| cifar10_3k | val+tse | mobilenetv2 | 0,6166 | 0,6077 |
| shapes | val | zoo_unet_res_block | 0,9969 | 0,9985 |
| shapes | val+tse | zoo_unet_res_block | 0,9969 | 0,9985 |

Hai cách cho cùng top-1 và cùng các mạng vào chung kết trên cả ba bộ. Với 150 epoch chỉ khoảng 10
ứng viên được huấn luyện, vòng đầu giữ 4, và mạng tốt nhất lọt qua theo cả hai cách xếp. Kế hoạch
ban đầu là 2 lần lặp và thêm cách `screen` (vòng sàng lọc 1 epoch). Đã rút xuống vì mỗi lần chạy mất
khoảng 15–23 phút trên CPU, cả bộ sẽ mất gần 7 giờ.

## Kết luận

- Không đo được lợi ích đầu-cuối. **Mặc định giữ nguyên `--rung-score val`, `--rungs 4,12,30`.**
- `tse` và `val+tse` vẫn có để dùng khi vòng đầu rất ngắn (1 epoch), lúc val còn nhiễu. Ở đó TSE xếp
  cả nhóm đúng hơn (0,60 so với 0,44), nhưng chưa có số đo đầu-cuối nào cho thấy top-1 tốt hơn.
- Hạn chế: dữ liệu ảnh nhỏ (3k mẫu), một seed, CPU. Chưa đo trên HSI, bảng, chuỗi thời gian, hay
  với ngân sách lớn hơn, khi nhiều ứng viên hơn phải cạnh tranh ở vòng đầu.
