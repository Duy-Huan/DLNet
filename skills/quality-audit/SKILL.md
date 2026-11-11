---
name: quality-audit
description: Điều phối đội agent kiểm định chất lượng DLNet (độ tin cậy, mức hiện đại, tính khả thi, thuật toán) theo qa/CHECKLIST.md và gộp thành báo cáo cùng lộ trình nâng cấp.
triggers: quality, audit, QA, kiểm định, checklist, harness, agent team, cutting edge, SOTA, reliability, feasibility
---
# Kiểm định chất lượng bằng đội agent

Tài liệu: `qa/README.md` (vai trò, mẫu báo cáo), `qa/CHECKLIST.md` (tiêu chí), định nghĩa agent ở `.claude/agents/dlnet-*-auditor.md`.

1. Xác định phiên bản đang kiểm (`dlnet/__init__.py`), tạo `qa/reports/<phiên bản>/`.
2. Chuẩn bị `QA_DATA` (dữ liệu thật, ngoài repo) và `QA_WORK` (thư mục làm việc ngoài repo). Ghi danh sách dữ liệu vào lời giao việc.
3. Khởi động song song 4 agent. Nếu loại agent `dlnet-*-auditor` chưa được nạp trong phiên (định nghĩa mới thêm), dùng agent đa năng và yêu cầu nó đọc và làm theo file định nghĩa tương ứng. Lời giao việc ghi: phiên bản, `QA_DATA`, `QA_WORK/<tên agent>`, đường dẫn báo cáo, giới hạn thời gian.
4. Khi có báo cáo: đọc cả 4, **kiểm lại** mọi phát hiện "nghiêm trọng" bằng cách chạy lại lệnh bằng chứng; phát hiện không tái hiện được thì hạ mức và ghi lý do.
5. Viết `qa/reports/<phiên bản>/SUMMARY.md`: bảng chấm toàn checklist, phần chưa tốt theo mức độ, đề xuất nâng cấp có nguồn và công sức, thứ tự ưu tiên.
6. Cập nhật `agent.md` §18 (lộ trình) và ghi mốc kiểm định; báo cáo cho người dùng bằng tiếng Việt.
7. Phát hiện là lỗi thật thì sửa ở mốc sau theo skill `audit-and-regression` (regression test trước), không sửa trong lúc kiểm định.
