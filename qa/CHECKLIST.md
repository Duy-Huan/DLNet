# Checklist kiểm định chất lượng DLNet

Dùng cho đội agent kiểm định (`qa/README.md`). Mỗi tiêu chí có **ID**, **câu hỏi**, **cách đo** và **ngưỡng đạt**. Mỗi agent chấm từng tiêu chí của nhóm mình là **ĐẠT / MỘT PHẦN / KHÔNG ĐẠT / CHƯA ĐO ĐƯỢC**, kèm bằng chứng: lệnh đã chạy, số đo, file và dòng mã. Không có bằng chứng thì ghi "CHƯA ĐO ĐƯỢC", không đoán.

Quy ước đo:
- Dùng ngân sách epoch (`--budget Nep`) và `--threads 1` để kết quả tái lập được và các agent không tranh CPU.
- Mỗi kết luận về chất lượng cần ít nhất 2 bộ dữ liệu; mỗi so sánh A/B cần ít nhất 2 seed.
- Dữ liệu thật dùng chung nằm ở thư mục ghi trong `qa/README.md`.

## R. Độ tin cậy (kết quả có đáng tin không)

| ID | Câu hỏi | Cách đo | Ngưỡng đạt |
|---|---|---|---|
| R1 | Có rò rỉ dữ liệu không? | Đọc mã `prepare` của mọi plugin: chuẩn hóa, PCA, impute, chia khối HSI chỉ fit trên train; kiểm thêm bằng phép thử, ví dụ đổi nhãn test thì val không đổi | Không có bước fit nào dùng val/test |
| R2 | Quy tắc kết luận "thắng baseline" có đúng không? | Đọc `export.verdict`; dựng ca biên (thắng val nhưng thua test, chênh trong nhiễu) | Chỉ "thắng" khi hơn quá nhiễu trên val **và** hơn trên test |
| R3 | Cùng lệnh có cho cùng kết quả không? | Chạy 2 lần `--budget Nep --threads 1` | Leaderboard giống hệt |
| R4 | Thứ hạng có ổn định qua seed không? | Cùng dữ liệu, `--seed 0/1/2`: so top-1 và tập top-3 | Top-1 thuộc top-3 của các seed khác ở ≥ 2/3 lần; ghi Kendall tau |
| R5 | Val có dự đoán được test không? | Spearman(val_mean, test) trên mọi finalist; hạng test của top-1 theo val | Spearman ≥ 0,5; top-1 theo val nằm trong top-3 theo test |
| R6 | Xếp hạng ngân sách ngắn có giữ khi huấn luyện đầy đủ không? | Huấn luyện lại top-K bằng `train.py` (ví dụ 30–60 epoch), so thứ tự với leaderboard | Spearman ≥ 0,5; top-1 vẫn trong top-2 |
| R7 | Ngân sách có được tôn trọng không? | Mục Budget của leaderboard, đo giờ thực | Giây ≤ 1,1 × ngân sách; epoch ≤ ngân sách |
| R8 | Đầu vào xấu có được xử lý rõ ràng không? | NaN, cột hằng, nhãn chuỗi, lớp chỉ 1–2 mẫu, mất cân bằng 1:50, dữ liệu rất nhỏ, sai shape, file sai định dạng | Báo lỗi dễ hiểu hoặc xử lý đúng; không có traceback khó hiểu, không treo |
| R9 | Một ứng viên lỗi có làm sập cả lần chạy không? | Ứng viên sinh loss NaN/inf hoặc lỗi khi chạy (ví dụ lr cực lớn qua `--extra-specs` hoặc monkeypatch) | Ứng viên đó bị loại, lần chạy vẫn xong |
| R10 | Gói xuất ra có tái tạo số liệu không? | `train.py --epochs <số epoch đã dùng>` so với val trong `config.yaml` | Lệch trong ±2 độ lệch chuẩn seed |
| R11 | Báo cáo thống kê có trung thực không? | Số seed, std, nhóm không phân biệt, cảnh báo trong leaderboard | Mỗi con số chính có std và số lần chạy |

## C. Mức hiện đại (cutting edge) của mạng đầu ra

| ID | Câu hỏi | Cách đo | Ngưỡng đạt |
|---|---|---|---|
| C1 | Kho kiến trúc có các họ hiện đại của từng miền không? | Đối chiếu `doc/zoo.md`, `zoo/presets.py`, `zoo/families.py` với danh sách SOTA 2023–2026 (có nguồn) cho ảnh, phân đoạn, chuỗi, bảng, đồ thị, recommender, HSI | Mỗi miền có ≥ 1 họ trong nhóm mạnh hiện nay, hoặc lý do rõ vì sao không |
| C2 | Công thức huấn luyện có hiện đại không? | Đọc `core.Trainer`, `train_full`: augmentation, label smoothing, warmup, EMA, mixup/cutmix, stochastic depth, gradient clipping, scheduler | Có các thành phần chuẩn của công thức hiện đại cho từng miền, hoặc ghi rõ cái nào thiếu |
| C3 | Top-1 của DLNet có cạnh tranh không? | Trên dữ liệu thật (MNIST/Fashion/CIFAR con, bảng, HSI): so top-1 với baseline cổ điển mạnh và với một mạng chuẩn viết tay (ví dụ ResNet-18 nhỏ, MLP chuẩn) **cùng số epoch** | Top-1 ≥ mạng chuẩn viết tay cùng ngân sách trên ≥ 2/3 bộ |
| C4 | Thuật toán tìm kiếm có theo kịp hiện nay không? | So SH, Hyperband, aging evolution, ZAP với BOHB, DEHB, ASHA, PriorBand, LLM-guided NAS, zero-shot NAS 2024+ | Liệt kê khoảng trống, có nguồn |
| C5 | Có tận dụng mô hình pretrained / foundation model không? | Kiểm tra fine-tune từ trọng số có sẵn (ImageNet, TabPFN, ...) | Có, hoặc ghi rõ là khoảng trống |
| C6 | Đầu ra có sẵn sàng triển khai hiện đại không? | Mixed precision, `torch.compile`, ONNX/TorchScript, lượng tử hóa, gộp RepVGG | Liệt kê có và thiếu |

## F. Tính khả thi khi người dùng thật dùng DLNet để đề xuất thiết kế

| ID | Câu hỏi | Cách đo | Ngưỡng đạt |
|---|---|---|---|
| F1 | Cài đặt từ đầu có dễ không? | Làm theo `doc/user-guide.md` §1 trong venv mới; `python -m dlnet selfcheck` | Thành công, không cần bước ngoài tài liệu |
| F2 | Mỗi loại bài toán chạy được theo đúng hướng dẫn không? | Chạy từng lệnh của `doc/user-guide.md` §4 trên dữ liệu thật hoặc tổng hợp | Mọi lệnh chạy được, kết quả đúng như mô tả |
| F3 | Tài liệu có khớp với mã không? | Đối chiếu tùy chọn, tên file đầu ra, khóa JSON trong tài liệu với hành vi thật | Không có sai lệch |
| F4 | Thông báo lỗi có chỉ được cách sửa không? | Gây 10 lỗi thường gặp (sai `--target`, sai key, thư mục `--out` có sẵn, ...) | ≥ 8/10 thông báo nói rõ cần làm gì |
| F5 | Gói đầu ra có dùng được ngay không? | Chạy test của gói, `train.py`, nạp mô hình trong mã khác; kiểm có đường suy luận (predict) và lưu/nạp trọng số không | Huấn luyện, lưu, nạp, dự đoán được mà không phải đọc mã DLNet |
| F6 | Chạy được với dữ liệu cỡ thật không? | Đo thời gian và RAM khi tăng cỡ (ví dụ 3k → 30k ảnh, 100k dòng bảng) | Không vượt RAM; thời gian tăng gần tuyến tính; giới hạn được ghi trong tài liệu |
| F7 | Người dùng có hiểu được kết quả không? | Đọc `leaderboard.md` như người mới: đủ để chọn mô hình và biết bước tiếp theo? | Có khuyến nghị rõ, giải thích được các cột |
| F8 | Tích hợp AI (advisor, MCP) có chạy không? | MCP server qua JSON-RPC; advisor với `--advisor-cmd` giả | Chạy đúng, lỗi được xử lý |

## A. Thuật toán (đúng nguồn, có hiệu quả đo được)

| ID | Câu hỏi | Cách đo | Ngưỡng đạt |
|---|---|---|---|
| A1 | Mỗi thuật toán có đúng với nguồn không? | Đối chiếu ngẫu nhiên ≥ 4 thuật toán trong `doc/algorithms.md` với paper hoặc mã chính thức | Không sai lệch, hoặc sai lệch đã được ghi lại |
| A2 | Chọn hạt giống có tốt hơn ngẫu nhiên không? | `--seeds N` với `--seed-order data` so với `random`, cùng ngân sách epoch, ≥ 2 bộ × 2 seed | Val của top-1 tốt hơn ngẫu nhiên ổn định (quá nhiễu) |
| A3 | Hyperband có hơn successive halving không? | `--scheduler hyperband` so với `sh`, cùng ngân sách epoch | Ghi kết quả; khuyến nghị mặc định dựa trên số đo |
| A4 | Tiến hóa có thêm giá trị không? | `--evolve 0` so với `--evolve 2`, cùng ngân sách | Ghi kết quả; tiến hóa nên thắng hoặc không thua |
| A5 | Ước lượng hyperparameter có giúp không? | Tỉ lệ đề xuất được nhận và mức cải thiện val trong `leaderboard.json` | Ghi tỉ lệ và lợi ích |
| A6 | Zero-cost proxy có dự đoán được không? | Mục proxies của leaderboard trên ≥ 3 bộ | Ghi Spearman; so với đếm tham số |
| A7 | Surrogate ZAP có tái lập được số đã công bố không? | `scripts/bench_surrogate.py` hoặc tập con | Khớp số trong `doc/algorithms.md` trong mức nhiễu |
| A8 | Hướng tối ưu (`--objective`) có cho đúng thứ tự không? | `params`, `pareto` trên dữ liệu thật: top-1 có nhỏ hơn mà vẫn trong ngưỡng chênh cho phép không | Đúng định nghĩa |
| A9 | Ngân sách có được phân bổ hiệu quả không? | Mục Budget: phần trăm thời gian không huấn luyện, số ứng viên được train đủ rung | Ghi số đo; chỉ ra giai đoạn lãng phí |

## Tổng hợp

Người điều phối (agent chính) gộp báo cáo của các agent thành `qa/reports/<phiên bản>/SUMMARY.md`:
1. Bảng chấm toàn bộ checklist.
2. Những phần chưa tốt, xếp theo mức độ (nghiêm trọng / quan trọng / nên có).
3. Tính năng và thuật toán cần nâng cấp, có nguồn tham khảo và ước lượng công sức.
4. Cập nhật lộ trình trong `agent.md` §18.
