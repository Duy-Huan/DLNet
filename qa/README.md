# Đội agent kiểm định chất lượng DLNet

Một bộ khung kiểm định dùng lại được: 4 agent độc lập chấm DLNet theo [`CHECKLIST.md`](CHECKLIST.md), mỗi agent một nhóm tiêu chí. Một agent điều phối gộp kết quả thành báo cáo tổng hợp và lộ trình nâng cấp. Kết quả của mỗi lần kiểm định nằm ở `qa/reports/<phiên bản>/`.

| Agent (định nghĩa trong `.claude/agents/`) | Nhóm | Câu hỏi chính |
|---|---|---|
| `dlnet-reliability-auditor` | R1–R11 | Có tin được leaderboard không? (rò rỉ, tái lập, ổn định thứ hạng, val so với test, ngắn so với dài) |
| `dlnet-sota-auditor` | C1–C6 | Mạng đầu ra có hiện đại không? (kiến trúc, công thức huấn luyện, so với mạng chuẩn cùng ngân sách) |
| `dlnet-feasibility-auditor` | F1–F8 | Người dùng thật có đi được từ dữ liệu tới mô hình dùng được không? |
| `dlnet-algorithm-auditor` | A1–A9 | Thuật toán có đúng nguồn và có hiệu quả đo được không? |

## Cách chạy

Trong Claude Code, ở thư mục repo:

1. Chuẩn bị dữ liệu thật vào một thư mục (`QA_DATA`), ví dụ các bộ `.npz` ảnh (MNIST, Fashion-MNIST, CIFAR con), CSV bảng, ảnh siêu phổ `.mat`, và một thư mục làm việc (`QA_WORK`) nằm ngoài repo.
2. Yêu cầu: *"Chạy đội kiểm định chất lượng DLNet theo qa/README.md, dữ liệu ở <QA_DATA>, làm việc ở <QA_WORK>"*. Agent điều phối khởi động song song 4 agent ở trên, mỗi agent nhận hai đường dẫn này cùng phiên bản đang kiểm.
3. Mỗi agent ghi `qa/reports/<phiên bản>/<tên agent>.md`. Agent điều phối đọc cả 4 báo cáo, kiểm lại các phát hiện nghiêm trọng (chạy lại lệnh bằng chứng), rồi viết `SUMMARY.md` và cập nhật `agent.md` §18.

Quy trình chi tiết cho agent điều phối: skill [`skills/quality-audit`](../skills/quality-audit/SKILL.md).

## Mẫu báo cáo của mỗi agent

```markdown
# Kiểm định <nhóm> — DLNet <phiên bản>

## Môi trường và dữ liệu
(máy, số CPU, có GPU không, các bộ dữ liệu đã dùng, tổng thời gian)

## Bảng chấm
| ID | Kết quả (ĐẠT / MỘT PHẦN / KHÔNG ĐẠT / CHƯA ĐO ĐƯỢC) | Bằng chứng (lệnh, số đo, file:dòng) |

## Phát hiện
### Nghiêm trọng (kết quả sai hoặc gây hiểu nhầm, sập, mất dữ liệu)
### Quan trọng (làm giảm chất lượng hoặc khả năng dùng rõ rệt)
### Nên có

## Đề xuất nâng cấp
| Đề xuất | Vì sao (bằng chứng) | Nguồn (paper / mã chuẩn) | Công sức S/M/L | Lợi ích kỳ vọng |

## Giới hạn của lần kiểm định này
```

## Nguyên tắc

- Agent kiểm định **không sửa** mã DLNet; việc sửa thuộc về các mốc phát triển sau, dựa trên báo cáo.
- Ngân sách epoch và `--threads 1` để số đo tái lập được và các agent không tranh CPU.
- Không có bằng chứng thì không kết luận. Kiểm định chưa chạy trên GPU thì phải ghi rõ.
