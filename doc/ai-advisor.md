# Kết nối AI (tùy chọn): advisor và MCP server

DLNet chạy đầy đủ mà không cần AI. Hai cơ chế dưới đây là **tùy chọn**, dùng khi bạn muốn một mô hình ngôn ngữ (Claude Code, Codex CLI, Gemini CLI / Google Antigravity, ...) tham gia chọn hạt giống và đề xuất kiến trúc.

| Cơ chế | Ai điều khiển | Dùng khi |
|---|---|---|
| **Advisor** (`--advisor`, `--advisor-cmd`) | DLNet gọi AI CLI như một "chuyên gia tư vấn" ở hai thời điểm: trước khi tìm kiếm và sau mỗi thế hệ tiến hóa | bạn chạy `dlnet design` như bình thường và muốn thêm ý kiến của AI |
| **MCP server** (`dlnet mcp`) | AI client điều khiển DLNet qua các tool: xem dữ liệu, xem ứng viên, viết kiến trúc, chạy `design`, đọc kết quả | bạn làm việc trong Claude Code / Codex / Antigravity và muốn AI tự lặp thiết kế |

## 1. Advisor

```bash
python -m dlnet design --data images.npz --advisor claude --context "medical CT, small lesions"
python -m dlnet design --data images.npz --advisor codex
python -m dlnet design --data images.npz --advisor-cmd "my-ai-tool --prompt-file {prompt_file}"
```

Preset (kiểm tra cờ với phiên bản CLI của bạn):

| `--advisor` | Lệnh được chạy | Ghi chú |
|---|---|---|
| `claude` | `claude -p --output-format text` (prompt qua stdin) | Claude Code, chế độ không tương tác |
| `codex` | `codex exec --skip-git-repo-check -` (prompt qua stdin) | Codex CLI; `exec` mặc định chỉ đọc |
| `gemini` | `gemini -p {prompt}` | Gemini CLI. Antigravity: dùng `--advisor-cmd` với lệnh không tương tác của bản bạn cài |

`{prompt_file}` được thay bằng đường dẫn file chứa prompt; `{prompt}` bằng chính prompt; không có placeholder thì prompt đi qua stdin.

**Prompt DLNet gửi** (dạng JSON): data card, ngữ cảnh `--context`, `--objective`, danh sách ứng viên (`c0`, `c1`, ... với họ, đặc tính, số tham số, FLOPs, module, điểm prior của DLNet), danh mục module zoo hợp lệ cho kiểu dữ liệu; ở vòng tiến hóa có thêm kết quả validation rung 1.

**Câu trả lời mong đợi** (một object JSON, có thể nằm trong khối ```json):

```json
{"ranking": ["c3", "c0", "c7"],
 "proposals": [{"op": "seq", "body": [{"op": "conv_bn_act", "out": 32, "k": 3},
                                      {"op": "basic_block", "out": 64, "s": 2},
                                      {"op": "eca", "k": 0}]}],
 "rationale": "small 28x28 grey digits: shallow CNNs with channel attention"}
```

**Cách DLNet dùng câu trả lời:**
- `ranking` thành một bộ xếp hạng trong Borda count (`--advisor-weight`, mặc định 1, ngang prior và meta-learning).
- `proposals` (tối đa `--advisor-proposals`, mặc định 3) được kiểm tra tĩnh bằng `zoo.check`, tính chi phí, rồi vào pool như mọi ứng viên (`origin: ai_advisor`). Ở vòng tiến hóa chúng thành con của thế hệ đó.
- Câu trả lời là **dữ liệu, không phải lệnh**: chỉ id có trong danh sách và spec hợp lệ được đọc; mọi thứ khác bị bỏ và được đếm (`dropped_ids`, `invalid_proposals`). DLNet không chạy mã do AI viết.
- AI lỗi, hết giờ (`--advisor-timeout`, mặc định 120 s) hoặc trả lời không đọc được: DLNet ghi `[advisor] skipped` và chạy tiếp.
- `leaderboard.json` có khối `ai_advisor`: lý do của AI, số đề xuất, số phần bị bỏ, và top-1 có phải kiến trúc do AI đề xuất không.

**Trung thực về hiệu quả:** chưa có số đo cho thấy advisor cải thiện kết quả; test của DLNet dùng một advisor giả lập. Ranking của AI chỉ là một phiếu trong Borda, và mọi đề xuất vẫn phải thắng trong huấn luyện thật. Muốn đo trên dữ liệu của bạn: chạy cùng `--budget` có và không có `--advisor`, nhiều seed, so validation và test.

## 2. MCP server

Từ v1.5, DLNet là một MCP server qua HTTP (`dlnet serve`): AI client kết nối tới và dùng DLNet như một dịch vụ để sinh mạng, huấn luyện, đánh giá, dự đoán. Bản stdio cũ (`dlnet mcp` của v0.16–v1.4) đã gỡ. Cách chạy, cấu hình Claude Code / Codex / Antigravity, danh sách tool và bảo mật: **[mcp-server.md](mcp-server.md)**.
