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

`python -m dlnet mcp` chạy DLNet như một MCP server qua stdio (JSON-RPC 2.0, mỗi dòng một thông điệp; viết theo đặc tả MCP 2025-06-18, không cần SDK). Tool:

| Tool | Làm gì |
|---|---|
| `dlnet_profile` | nạp dữ liệu, trả plugin, hồ sơ, data card |
| `dlnet_candidates` | pool hạt giống DLNet sẽ tìm (đặc tính, kích thước, điểm prior, spec) |
| `dlnet_zoo` | danh mục module (kiểu đầu vào, tham số, nguồn) để viết kiến trúc |
| `dlnet_check_spec` | kiểm tra tĩnh một spec: số tham số, shape đầu ra hoặc lỗi |
| `dlnet_design` | chạy `dlnet design` (chặn tới hết `budget`), có thể kèm `extra_specs` do AI viết; `device`, `threads` như tùy chọn CLI |
| `dlnet_leaderboard` | đọc kết quả của một lần chạy |

Vòng làm việc gợi ý cho AI: `dlnet_profile` → `dlnet_candidates` → viết vài spec, kiểm bằng `dlnet_check_spec` → `dlnet_design` với `extra_specs` → `dlnet_leaderboard` → lặp lại.

### Cấu hình client

Thay `/path/to/DLNet` và Python bằng của bạn (Python của môi trường đã cài DLNet).

**Claude Code** (lệnh hoặc file `.mcp.json` ở gốc dự án):

```bash
claude mcp add --transport stdio dlnet -- python -m dlnet mcp
```

```json
{"mcpServers": {"dlnet": {"command": "python", "args": ["-m", "dlnet", "mcp"], "cwd": "/path/to/DLNet"}}}
```

**Codex CLI** (`~/.codex/config.toml`, hoặc `codex mcp add`; tên bảng dùng gạch dưới):

```toml
[mcp_servers.dlnet]
command = "python"
args = ["-m", "dlnet", "mcp"]
cwd = "/path/to/DLNet"
```

**Google Antigravity** (Agent panel → MCP Servers → Manage MCP Servers → View raw config, file `mcp_config.json`; vị trí file khác nhau giữa các bản, ví dụ `~/.gemini/antigravity/mcp_config.json`):

```json
{"mcpServers": {"dlnet": {"command": "python", "args": ["-m", "dlnet", "mcp"], "cwd": "/path/to/DLNet"}}}
```

**Gemini CLI** (`~/.gemini/settings.json`): cùng khối `mcpServers` như trên.

Lưu ý: `dlnet_design` chạy hết `budget` trước khi trả lời. Nhiều client có giới hạn thời gian cho một lần gọi tool, nên chọn `budget` nhỏ hơn giới hạn đó (hoặc tăng giới hạn trong client). Các cấu hình trên lấy từ tài liệu của từng client tại thời điểm viết; DLNet chỉ kiểm tra server bằng test JSON-RPC, chưa chạy với từng client thật.
