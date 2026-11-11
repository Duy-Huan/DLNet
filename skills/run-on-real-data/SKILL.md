---
name: run-on-real-data
description: Chạy DLNet trên dữ liệu thật rồi kiểm tra và báo cáo kết quả trung thực.
triggers: experiment, dataset, budget, real data, run, report, Indian Pines, Pavia
---
# Chạy trên dữ liệu thật

1. Cài đặt: `pip install -e ".[dev,design]"` (cần torch, scikit-learn; HSI cần scipy). Môi trường có thể không có GPU: giữ budget nhỏ (3 đến 8 phút) và nói rõ khi báo cáo.
2. Dữ liệu tải về đặt ngoài repo (thư mục scratchpad hoặc `/tmp`), không commit dữ liệu thô. Kiểm tra kích thước và thống kê khớp công bố trước khi dùng (HSI: số pixel có nhãn, số lớp). File rỗng hoặc sai kích thước thì bỏ.
3. Chạy `python -m dlnet design --data ... --budget 5m --topk 3 --out <thư mục ngoài repo>`. Với HSI chạy cả hai chế độ `--split pixel` và `--split block`.
4. Đọc `leaderboard.md`, `profile.json`, `reference/results.json`. Đối chiếu val với test của từng ứng viên: nếu top1 theo val không phải top1 theo test, ghi nhận nhiễu ranking.
5. Kiểm tra thời gian thực so với budget: mục "Budget" trong `leaderboard.md` ghi giây và epoch của từng giai đoạn. Chạy nhiều lệnh cùng lúc thì đặt `--threads` sao cho tổng số luồng ≤ số CPU (`agent.md` §9.2).
6. Báo cáo theo skill `evaluate-honestly`: nêu số của mô hình sâu, của baseline cổ điển, verdict, và các hạn chế. Không khái quát từ một lần chạy.
7. Nếu phát hiện lỗi hoặc hạn chế mới, ghi vào `agent.md` §9.
