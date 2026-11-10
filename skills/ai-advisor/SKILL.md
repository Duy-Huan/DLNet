---
name: ai-advisor
description: Dùng, sửa hoặc đo tùy chọn AI của DLNet: advisor gọi AI CLI (Claude Code, Codex, Gemini/Antigravity) và MCP server `dlnet mcp`.
triggers: AI, advisor, MCP, claude, codex, antigravity, gemini, LLM, agent, tool
---
# AI advisor và MCP server (doc/ai-advisor.md)

1. **Tùy chọn, không bắt buộc**: mọi đường đi của `dlnet design` phải chạy được khi không có AI. Lỗi của AI (thiếu lệnh, hết giờ, trả lời hỏng) chỉ ghi `[advisor] skipped`.
2. **Câu trả lời của AI là dữ liệu**: chỉ đọc id có trong danh sách và spec zoo; mọi spec qua `space._candidate` (kiểm tĩnh + giới hạn tham số) rồi cost model. Không bao giờ `eval`/chạy mã do AI viết, không để AI chọn đường dẫn ghi file.
3. **Thêm preset CLI**: sửa `advisor.PRESETS`, ghi rõ cờ không tương tác của CLI đó và nguồn tài liệu trong `doc/ai-advisor.md`; nói rõ nếu chưa chạy thử với CLI thật.
4. **MCP server** (`dlnet/mcp_server.py`): JSON-RPC 2.0 một dòng một thông điệp; stdout chỉ chứa JSON-RPC (mọi `print` bị chuyển sang stderr). Thêm tool: thêm vào `TOOLS` (có `inputSchema`) và `HANDLERS`, thêm test trong `tests/test_advisor_mcp.py`.
5. **Test**: dùng advisor giả lập bằng script Python (không gọi AI thật trong CI); có test cho trả lời rác, id lạ, spec độc hại, lệnh không tồn tại.
6. **Đo trước khi khẳng định**: chạy cùng `--budget`, nhiều seed, có và không có `--advisor`; ghi số vào `agent.md` §16.6. Chưa đo thì tài liệu phải nói chưa đo.
