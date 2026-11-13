# DLNet MCP server (`dlnet serve`)

DLNet chạy được như một **dịch vụ MCP qua HTTP**: bạn bảo AI (Claude Code, Codex CLI, Google Antigravity, hoặc bất kỳ MCP client nào) dùng DLNet để phân tích dữ liệu, sinh mạng deep learning, huấn luyện, đánh giá, dự đoán và đọc kết quả. Server chạy trên máy có dữ liệu / GPU; client ở cùng máy hoặc ở máy khác.

Từ v1.5 đây là MCP server duy nhất của DLNet (bản stdio của các phiên bản trước đã gỡ). Chỉ dùng thư viện chuẩn của Python, không cần cài MCP SDK.

## 1. Chạy server

```bash
# trên máy của bạn (chỉ máy này truy cập được, không cần token)
python -m dlnet serve --workspace ~/dlnet-workspace

# cho máy khác truy cập: bắt buộc có token
export DLNET_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
python -m dlnet serve --host 0.0.0.0 --port 8765 --workspace /data/dlnet-workspace
```

Server in ra địa chỉ MCP (`http://<host>:8765/mcp`) và lệnh kết nối mẫu. `dlnet mcp` là tên gọi khác của `dlnet serve`.

| Tùy chọn | Mặc định | Ý nghĩa |
|---|---|---|
| `--host`, `--port` | `127.0.0.1`, `8765` | địa chỉ lắng nghe. Khác loopback thì **phải** có token, nếu không server từ chối khởi động |
| `--workspace` | `dlnet-workspace` | thư mục làm việc: `data/` (dữ liệu, thẻ paper), `runs/` (kết quả), `jobs/` (trạng thái và log) |
| `--token`, `--token-file`, `DLNET_TOKEN`, `--generate-token` | không | bearer token client phải gửi |
| `--max-jobs` | 1 | số job chạy cùng lúc (các job khác xếp hàng). Tăng khi máy có nhiều GPU / CPU |
| `--max-budget` | `12h` | ngân sách thời gian lớn nhất một lần `design` được xin |
| `--wait` | 45 | số giây một tool chờ job trước khi trả `job_id` (xem mục 4) |
| `--allow-fetch` | tắt | bật tool `dlnet_fetch` (tải URL https công khai về `data/`) |
| `--no-sandbox` | tắt | chấp nhận đường dẫn ngoài workspace (chỉ khi mọi client đều đáng tin) |
| `--allow-origin`, `--allow-host`, `--public-url` | | client chạy trong trình duyệt; server sau reverse proxy |
| `--certfile`, `--keyfile` | | phục vụ https trực tiếp (hoặc đặt reverse proxy có TLS phía trước) |
| `--max-upload-gb` | 8 | giới hạn một lần tải tệp lên |

## 2. Kết nối client

Thay URL và token bằng của bạn. Với server chỉ chạy trên loopback không có token thì bỏ phần header.

**Claude Code**

```bash
claude mcp add --transport http dlnet http://127.0.0.1:8765/mcp --header "Authorization: Bearer $DLNET_TOKEN"
```

hoặc `.mcp.json` ở gốc dự án (biến môi trường được thay thế khi đọc):

```json
{"mcpServers": {"dlnet": {"type": "http", "url": "http://127.0.0.1:8765/mcp",
                          "headers": {"Authorization": "Bearer ${DLNET_TOKEN}"}}}}
```

**Codex CLI**

```bash
codex mcp add dlnet --url http://127.0.0.1:8765/mcp --bearer-token-env-var DLNET_TOKEN
```

hoặc `~/.codex/config.toml`:

```toml
[mcp_servers.dlnet]
url = "http://127.0.0.1:8765/mcp"
bearer_token_env_var = "DLNET_TOKEN"
```

**Google Antigravity** (Agent panel → MCP Servers → Manage MCP Servers → View raw config, tệp `mcp_config.json`):

```json
{"mcpServers": {"dlnet": {"serverUrl": "http://127.0.0.1:8765/mcp",
                          "headers": {"Authorization": "Bearer <token>"}}}}
```

**Client khác**: mọi client hỗ trợ transport "Streamable HTTP" đều dùng được; điểm cuối là `/mcp`, xác thực bằng header `Authorization: Bearer <token>`.

Mức độ kiểm chứng (trung thực): server được kiểm bằng **MCP Python SDK chính thức** làm client ở cả hai thế hệ giao thức (job CI `mcp-sdk`, `scripts/verify_mcp.py`). Cú pháp Claude Code lấy từ tài liệu của Claude Code, khóa cấu hình Codex lấy từ mã nguồn Codex (`codex-rs/config/src/mcp_types.rs`, `codex-rs/cli/src/mcp_cmd.rs`). Cấu hình Antigravity viết theo định dạng đã biết của nó, **chưa** đối chiếu được với tài liệu chính thức và chưa chạy với Antigravity thật.

## 3. Tool

Mọi đường dẫn là **tương đối so với workspace** (`data/iris.csv`, `runs/exp1/top1_cnn`). Đường dẫn thoát ra ngoài (`..`, tuyệt đối, symlink) bị từ chối.

| Tool | Làm gì |
|---|---|
| `dlnet_info` | phiên bản, thiết bị (GPU / CPU), cấu trúc workspace, giới hạn, thứ tự dùng tool |
| `dlnet_files`, `dlnet_read_file`, `dlnet_write_file` | liệt kê, đọc tệp văn bản (report.md, model.py, log), ghi tệp nhỏ (CSV, thẻ paper; ≤ 5 MB) |
| `dlnet_fetch` | tải một URL https công khai về `data/` (chỉ khi bật `--allow-fetch`; địa chỉ nội bộ bị từ chối) |
| `dlnet_profile` | nạp dữ liệu: plugin, hồ sơ, data card |
| `dlnet_candidates` | pool hạt giống DLNet sẽ tìm, kèm spec của từng ứng viên |
| `dlnet_zoo`, `dlnet_check_spec` | danh mục module; kiểm tra tĩnh một kiến trúc do AI viết (số tham số, shape, sơ đồ khối SVG) |
| `dlnet_design` | chạy `dlnet design`: tìm kiếm, huấn luyện, đánh giá Top-K; nhận `extra_specs` (kiến trúc của AI) và `papers` (thẻ paper) để so sánh cùng giao thức |
| `dlnet_plan` | (v1.7) ước lượng trước một lần `design` với cùng tham số: số ứng viên mỗi vòng, giây mỗi epoch, thời gian tìm kiếm và so sánh paper, điểm cổ điển cần vượt, thẻ paper nào sẽ bị bỏ; không dự đoán điểm |
| `dlnet_job`, `dlnet_jobs`, `dlnet_cancel` | trạng thái, log, kết quả của job; danh sách; hủy |
| `dlnet_leaderboard`, `dlnet_package` | kết quả một lần chạy; chi tiết một gói baseline |
| `dlnet_train` | huấn luyện đầy đủ một gói (`train.py` của gói), trả điểm validation và test |
| `dlnet_predict` | dự đoán trên dữ liệu mới bằng gói đã huấn luyện |
| `dlnet_paper` | kiểm tra / dựng gói từ thẻ paper |
| `dlnet_archive` | nén một lần chạy hoặc một gói thành zip để tải về |

Ngoài ra có ba prompt mẫu (`design_baselines`, `propose_architecture`, `compare_with_papers`) mà client hiển thị như lệnh có sẵn.

Lỗi do tham số sai (đường dẫn không có, tham số lạ, ngân sách quá giới hạn) được trả như **lỗi của tool** (`isError: true`) kèm cách sửa, để AI tự chỉnh và gọi lại.

## 4. Job chạy nền

`dlnet_design`, `dlnet_plan`, `dlnet_train`, `dlnet_predict`, `dlnet_paper` chạy như **job**: một tiến trình con, tối đa `--max-jobs` cái cùng lúc, còn lại xếp hàng. Tool chờ job tối đa `wait` giây:

- job xong trong lúc chờ: tool trả luôn kết quả (bảng xếp hạng, điểm số, ...);
- chưa xong: tool trả `job_id` và `status`; AI gọi `dlnet_job` (cũng có `wait`) cho tới khi xong.

Nhờ vậy một lần `design` 2 giờ không bao giờ vượt giới hạn thời gian gọi tool của client. Hãy đặt `wait` nhỏ hơn giới hạn đó (mặc định 45 giây). Khi client gửi progress token, server trả lời bằng luồng SSE có thông báo tiến độ (dòng log mới nhất) rồi mới tới kết quả.

Hủy một lần gọi (client hủy hoặc đóng kết nối) chỉ dừng việc **chờ**; job vẫn chạy và kết quả vẫn nằm trong workspace. Muốn dừng job, gọi `dlnet_cancel`. Server tắt giữa chừng thì job đang chạy được báo là `lost` ở lần khởi động sau.

## 5. Tệp lớn

Tool chỉ ghi được tệp ≤ 5 MB. Dữ liệu lớn tải lên / xuống bằng HTTP thường, cùng token:

```bash
curl -H "Authorization: Bearer $DLNET_TOKEN" -T train.npz http://host:8765/files/data/train.npz        # tải lên
curl -H "Authorization: Bearer $DLNET_TOKEN" -O http://host:8765/files/runs/exp1/top1_cnn.zip          # tải về
```

`GET /healthz` (không cần token) trả `{"status": "ok", "version": ...}` cho việc giám sát.

## 6. Bảo mật

- Mặc định chỉ lắng nghe `127.0.0.1`. Lắng nghe địa chỉ khác mà không có token: server từ chối khởi động.
- Token so sánh trong thời gian hằng; sai hoặc thiếu trả `401`.
- Header `Origin` được kiểm ở mọi yêu cầu (chống DNS rebinding): origin lạ trả `403`; khi chạy trên loopback, header `Host` lạ cũng trả `403`.
- Mọi đường dẫn bị giữ trong workspace; `jobs/` chỉ server ghi.
- Server chạy lệnh huấn luyện trên máy của bạn theo yêu cầu của client: **chỉ cấp token cho client bạn tin**, và dùng https (reverse proxy hoặc `--certfile`) khi đi qua mạng không tin cậy. Chưa có OAuth.

## 7. Giao thức

Viết theo đặc tả MCP, transport Streamable HTTP, hỗ trợ song song hai thế hệ:

| Thế hệ | Phiên bản | Cách hoạt động |
|---|---|---|
| Cũ (có phiên) | 2025-03-26, 2025-06-18, 2025-11-25 | bắt tay `initialize`, server cấp `MCP-Session-Id`; thiếu id trả 400, id lạ trả 404; `DELETE /mcp` kết thúc phiên |
| Mới (không phiên) | 2026-07-28 | không bắt tay; mỗi yêu cầu mang phiên bản trong `_meta` và các header `MCP-Protocol-Version`, `Mcp-Method`, `Mcp-Name` (phải khớp thân yêu cầu, sai trả 400 `HeaderMismatch`); `server/discover` công bố phiên bản và năng lực; mọi kết quả có `resultType` |

Phiên bản không hỗ trợ trả 400 kèm `UnsupportedProtocolVersionError` và danh sách phiên bản hỗ trợ. Chưa làm (đều là phần tùy chọn của đặc tả): luồng SSE nối lại được (`Last-Event-ID`), `subscriptions/listen` (danh sách tool không đổi khi server chạy), yêu cầu từ server về client (sampling, elicitation), OAuth.
