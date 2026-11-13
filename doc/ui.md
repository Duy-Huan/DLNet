# Giao diện web (`dlnet ui`, tùy chọn)

CLI vẫn là công cụ chính. Giao diện web chạy trên chính máy của bạn, đọc và ghi cùng thư mục làm việc với CLI. Nó chỉ giúp làm nhanh hơn ba việc: chọn tệp, xem kết quả, bật MCP server cho AI. Mọi thao tác trên giao diện đều hiện lệnh CLI tương ứng, chép ra chạy trong terminal hay VS Code là được.

```bash
dlnet ui                                    # mở http://127.0.0.1:8766 trong trình duyệt
dlnet ui --workspace ~/research/hsi-paper   # mở một workspace cụ thể
dlnet ui --port 9000 --no-browser
```

Lần đầu không có `--workspace` thì DLNet tạo `./dlnet-workspace`. Từ lần sau nó mở workspace dùng gần nhất (danh sách lưu ở `~/.dlnet/ui.json`).

## Workspace

Một thư mục gồm:

| Thư mục | Chứa |
|---|---|
| `data/` | dữ liệu: `.csv`, `.npz`, `.npy`, `.mat` |
| `papers/` | thẻ paper `.yaml` để so sánh |
| `runs/` | mỗi lần chạy `design` một thư mục |
| `jobs/` | trạng thái và log của các job (do giao diện và MCP server ghi) |

Một lần chạy khởi động trong terminal (`dlnet design ... --out runs/ten`) ngay trong workspace cũng hiện trên giao diện.

## Sáu màn hình

1. **Workspace**: danh sách workspace gần đây, duyệt thư mục trên máy, tạo workspace mới.
2. **Lần chạy**: mọi thư mục trong `runs/` với top-1, val (± độ lệch), test và trạng thái. Lần chạy đang chạy tự cập nhật mỗi vài giây.
3. **Tệp**:
   - Cây tệp của workspace, kéo thả tệp vào để tải lên: dữ liệu vào `data/`, `.yaml` vào `papers/`.
   - Bấm một tệp để xem DLNet nhận ra loại gì: ảnh, ảnh siêu phổ, bảng... kèm kích thước và số lớp. Tệp văn bản thì hiện nội dung.
4. **Chạy mới**:
   - Chọn đầu vào và ground truth (ảnh siêu phổ cần tệp `.mat` nhãn; DLNet tự đoán tệp có `gt` trong tên).
   - Đặt tên đầu ra, ngân sách (`5m`, `10m`, `1h`, `400ep` hoặc tự nhập) và số baseline, rồi tích các paper cần so sánh.
   - Lệnh `dlnet design` tương ứng hiện ở cuối trang.
   - **Ước lượng trước** chạy `dlnet plan` (bên dưới). **Chạy** khởi động một job như lệnh trong terminal.
5. **Chi tiết lần chạy**:
   - Một bảng gồm Top-K, baseline cổ điển tốt nhất và mỗi paper. Cột cuối của paper là Δ test so với top-1 cùng p-value đã hiệu chỉnh Holm.
   - Sơ đồ khối của top-1 và các tệp chính.
   - Tải cả lần chạy dạng `.zip`.
   - Lệnh đã chạy, bảng LaTeX (chép hoặc tải `results.tex`), log và nút dừng khi lần chạy còn chạy.
6. **AI**:
   - Bật và tắt MCP server (`dlnet serve`). Chọn chỉ máy này hay cả mạng LAN, cổng, số job cùng lúc, ngân sách tối đa, có cho tải dataset công khai không.
   - Token ngẫu nhiên tạo mới ở mỗi lần bật. Lệnh terminal tương đương hiện bên dưới.
   - Đoạn cấu hình cho Claude Code, Codex và Antigravity.
   - Nút **Thử kết nối**: giao diện tự đóng vai một client MCP (`initialize`, `tools/list`).
   - Danh sách 20 tool mà AI dùng được và nhật ký hoạt động. Chi tiết: [mcp-server.md](mcp-server.md).

## Ngôn ngữ và giao diện sáng/tối

- Nút `EN` / `VI` ở góc trên đổi ngôn ngữ. Mặc định theo ngôn ngữ của trình duyệt, lựa chọn được nhớ.
- Hai tệp `dlnet/ui/i18n/en.json` và `vi.json` phải **đồng bộ 100%**. `tests/test_ui.py` kiểm, CI chặn nếu lệch:
  - cùng một bộ khóa;
  - không chuỗi rỗng;
  - cùng các chỗ thay thế `{n}`;
  - mọi khóa mã dùng đều có, mọi khóa có đều được dùng;
  - không có tiếng Việt viết cứng trong `app.js`.
- Thêm chữ mới vào giao diện thì thêm khóa vào cả hai tệp.
- Nút thứ hai đổi giữa *Tự động* (theo hệ điều hành), *Sáng* và *Tối*. Màu chỉ khai báo bằng biến CSS, và test kiểm cả hai chế độ có đủ mọi biến.

## Ước lượng trước khi chạy: `dlnet plan`

Giao diện dùng lệnh này, nhưng nó chạy độc lập được:

```bash
dlnet plan --data data/cifar10.npz --budget 10m --papers papers/
dlnet plan ... --json          # JSON (giao diện và tool MCP dlnet_plan đọc dạng này)
```

**Cách làm** (mất 1–3 phút, không huấn luyện mạng nào tới hội tụ):
- Nạp dữ liệu như `design` và chạy baseline cổ điển (điểm cần vượt).
- Sinh ứng viên với cùng tùy chọn và giới hạn kích thước.
- Đo **một bước huấn luyện** của 8 ứng viên trải từ nhỏ tới lớn và của từng mạng paper. `design` cũng dùng chính phép đo này để giữ ngân sách thời gian.

**Kết quả báo ra:**
- Số ứng viên tới mỗi vòng successive halving.
- Thời gian tìm kiếm. Với ngân sách theo epoch, thời gian = số epoch × giây mỗi epoch đo được.
- Thời gian so sánh paper, tính thêm ngoài `--budget`.
- Thẻ paper nào sẽ bị bỏ, và lý do (khác kích thước đầu vào, lỗi).
- Điểm cổ điển cần vượt.

**Không dự đoán điểm của mạng.** Con số duy nhất về điểm là baseline cổ điển, đo thật, và con số paper tự báo trên cách chia dữ liệu của họ.

## Bảo mật

- Giao diện chỉ nghe trên `127.0.0.1`.
- Mỗi lần khởi động tạo một token ngẫu nhiên nhúng trong trang. Mọi lời gọi API phải gửi token, nên trang của site khác không gọi được.
- Header `Host` phải là máy này (chống DNS rebinding). Gọi từ origin khác bị từ chối.
- Tệp tải về hiện dạng văn bản, không bao giờ được chạy.
- Đường dẫn luôn bị giữ trong workspace. Riêng màn hình Workspace được duyệt thư mục trên máy để chọn workspace.

Vì vậy giao diện là công cụ một người dùng trên máy của mình. Đừng mở nó ra mạng; cho AI ở máy khác dùng DLNet thì dùng `dlnet serve` có token ([mcp-server.md](mcp-server.md)).
