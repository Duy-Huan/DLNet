# Tài liệu DLNet

Phiên bản tài liệu: **0.20.0** (phải trùng `version` trong `pyproject.toml`; `python scripts/release.py check` kiểm tra điều này).

DLNet nhận một bộ dữ liệu (bảng, ảnh, mask phân đoạn, chuỗi thời gian, ảnh siêu phổ, đồ thị, tương tác người dùng–mục) và trả về **Top-K mạng PyTorch** đã được huấn luyện thử, xếp hạng trung thực so với baseline cổ điển, mỗi mạng là một gói Python tự chạy (`model.py`, `train.py`, `config.yaml`, test).

| Tài liệu | Nội dung | Dành cho |
|---|---|---|
| [user-guide.md](user-guide.md) | Cài đặt, định dạng dữ liệu, chạy từng loại bài toán, chọn hướng tối ưu, đọc kết quả, huấn luyện lại, xử lý sự cố | người dùng |
| [cli.md](cli.md) | Tham chiếu đầy đủ mọi lệnh và tùy chọn (sinh tự động từ mã) | người dùng |
| [architecture.md](architecture.md) | Kiến trúc: pipeline, plugin, kho module (zoo), tìm kiếm, xuất gói | người phát triển |
| [algorithms.md](algorithms.md) | Thuật toán chuẩn đã dùng (tìm kiếm, hướng tối ưu, chọn hạt giống, proxy, hyperparameter), nguồn và kết quả đo | người dùng nâng cao, người phát triển |
| [zoo.md](zoo.md) | Danh mục mọi module của zoo (sinh tự động) | người thiết kế kiến trúc |
| [ai-advisor.md](ai-advisor.md) | Tùy chọn kết nối AI: advisor (Claude Code, Codex, Gemini/Antigravity) và MCP server `dlnet mcp` | người dùng muốn AI tham gia |

Tài liệu liên quan ở gốc repo: [`CHANGELOG.md`](../CHANGELOG.md) (thay đổi theo phiên bản), [`THIRD_PARTY.md`](../THIRD_PARTY.md) (nguồn module và kết quả đối chiếu với mã gốc), [`agent.md`](../agent.md) (kế hoạch, quy tắc phát triển, số đo chi tiết), [`skills/`](../skills/README.md) (quy trình cho người/AI phát triển).

## Quy tắc cập nhật (tóm tắt, đầy đủ ở `agent.md` §17)

- Mỗi bản phát hành: sửa các trang chịu ảnh hưởng, đổi dòng "Phiên bản tài liệu" ở trên, chạy `python scripts/gen_docs.py` để sinh lại `cli.md` và `zoo.md`.
- `cli.md` và `zoo.md` không sửa tay; test `tests/test_docs.py` báo lỗi khi chúng lệch với mã.
