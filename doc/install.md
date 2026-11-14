# Cài DLNet như một phần mềm (Windows, macOS, Linux)

Từ v1.5, mỗi bản phát hành trên GitHub (trang **Releases**, tag `vX.Y.Z`) có sẵn gói cài cho ba hệ điều hành. Gói cài chứa sẵn Python, PyTorch (CPU) và mọi thư viện DLNet cần: **không cần cài Python**. Sau khi cài, lệnh `dlnet` dùng được trong terminal, giống `python -m dlnet`.

| Hệ điều hành | Gói cài | Bản chạy ngay (không cài) |
|---|---|---|
| Windows 10/11 x64 | `DLNet-<ver>-windows-x64-setup.exe` | `DLNet-<ver>-windows-x64.zip` |
| macOS 14+ Apple silicon (M1–M4) | `DLNet-<ver>-macos-arm64.pkg` | `DLNet-<ver>-macos-arm64.tar.gz` |
| Linux x64 (glibc ≥ 2.35: Ubuntu 22.04+, Debian 12+, ...) | `DLNet-<ver>-linux-x64.deb` | `DLNet-<ver>-linux-x64.tar.gz` |
| Linux ARM64 (từ v1.6; Raspberry Pi 4/5 với hệ điều hành 64-bit, AWS Graviton, CPU của Jetson; glibc ≥ 2.35) | `DLNet-<ver>-linux-arm64.deb` | `DLNet-<ver>-linux-arm64.tar.gz` |

Mỗi hệ điều hành có `SHA256SUMS-<os>-<arch>.txt` để kiểm tra tệp tải về.

## Cài đặt

**Windows**: chạy `DLNet-<ver>-windows-x64-setup.exe`. Không có quyền quản trị thì chọn "Install for me only". Bộ cài thêm `dlnet` vào PATH (mở terminal mới sau khi cài) và tạo hai mục trong Start menu: *DLNet MCP server* (chạy `dlnet serve`, workspace ở `Documents\dlnet-workspace`) và *DLNet command prompt*. Gỡ trong *Settings → Apps*. Windows SmartScreen có thể cảnh báo vì bộ cài chưa được ký số: chọn *More info → Run anyway*.

**macOS**: mở `DLNet-<ver>-macos-arm64.pkg`. Gói chưa được Apple công chứng (notarize), nên lần đầu macOS chặn: chuột phải vào tệp → *Open*, hoặc *System Settings → Privacy & Security → Open Anyway*. DLNet được cài vào `/Library/DLNet`, lệnh `dlnet` ở `/usr/local/bin`. Gỡ: `sudo rm -rf /Library/DLNet /usr/local/bin/dlnet && sudo pkgutil --forget io.github.dlnet`. Trên Apple silicon, PyTorch dùng được GPU qua MPS (`--device mps`).

**Linux**:

```bash
sudo apt install ./DLNet-<ver>-linux-x64.deb      # cài vào /opt/dlnet, lệnh /usr/bin/dlnet (ARM64: ...-linux-arm64.deb)
sudo apt remove dlnet                              # gỡ
```

Bản `.tar.gz` (mọi hệ điều hành): giải nén ở đâu cũng được rồi chạy `dlnet` (Windows: `dlnet.exe`) trong thư mục đó.

## Kiểm tra sau khi cài

```bash
dlnet --version
dlnet selfcheck            # chạy thử một lần design đầy đủ (khoảng 20-60 giây)
```

## Dùng

Mọi lệnh như trong [user-guide.md](user-guide.md), thay `python -m dlnet` bằng `dlnet`:

```bash
dlnet design --data data.csv --target y --budget 10m --out results
dlnet serve --workspace ~/dlnet-workspace          # MCP server cho Claude / Codex / Antigravity (mcp-server.md)
```

Script trong gói đầu ra (`train.py`, `predict.py`) cần Python và PyTorch. Máy chỉ có DLNet đã cài (không có Python) thì chạy chúng bằng chính chương trình DLNet:

```bash
dlnet run-script results/top1_cnn/train.py --data data.csv --target y
dlnet run-script results/top1_cnn/predict.py --data new.csv --out pred.csv
```

`model.py` của gói là PyTorch thuần, dùng được ở bất kỳ môi trường nào có `torch`.

## Chạy bằng container (Podman / Docker)

Từ v1.8.2, repo có `Containerfile`: một image Linux (Python 3.12, PyTorch CPU, DLNet) chạy được trên bất kỳ máy nào có **Podman** (Windows và macOS qua `podman machine`, Linux trực tiếp) hoặc Docker. Dữ liệu và kết quả nằm trong một thư mục của máy bạn, gắn vào container ở `/workspace`.

Build image (một lần, từ thư mục gốc của repo; khoảng 2 GB vì có PyTorch):

```bash
podman build -t dlnet -f Containerfile .
```

Chạy `design` trên dữ liệu trong thư mục `work` hiện tại (kết quả ra `work/runs/a`):

```bash
podman run --rm -v ./work:/workspace:Z dlnet design --data data.csv --target y --budget 10m --out runs/a
```

Mọi lệnh khác dùng giống hệt: `podman run --rm -v ./work:/workspace:Z dlnet <lệnh> ...` (`selfcheck`, `paper import`, `run-script <gói>/train.py`, ...).

Giao diện web (`dlnet ui`) và MCP server (`dlnet serve`):

```bash
podman run --rm -p 127.0.0.1:8766:8766 -v ./work:/workspace:Z dlnet ui --workspace /workspace
```

```bash
podman run --rm -p 127.0.0.1:8765:8765 -v ./work:/workspace:Z -e DLNET_TOKEN=<token> dlnet serve --host 0.0.0.0 --workspace /workspace
```

Lưu ý:
- mở UI ở `http://127.0.0.1:8766/` (trang in ra trong log). Cổng bên ngoài phải **giống** cổng bên trong (`-p 127.0.0.1:9000:9000` cùng `ui --port 9000`), vì UI chỉ trả lời request ghi đúng `127.0.0.1` / `localhost` và cổng của nó;
- trong image, UI nghe trên 0.0.0.0 (biến `DLNET_UI_BIND`, hoặc `dlnet ui --bind`) để cổng chuyển tiếp tới được. Luôn publish với `127.0.0.1:` phía trước để máy khác trong mạng không truy cập được;
- `serve` trên 0.0.0.0 bắt buộc có token (`DLNET_TOKEN`), không thì từ chối chạy;
- `:Z` cần trên Linux có SELinux (Fedora, RHEL); trên máy khác bỏ đi cũng được;
- Podman rootless ánh xạ root trong container thành chính user của bạn, nên các file ghi ra `work/` thuộc về bạn. Với Docker, thêm `--user "$(id -u):$(id -g)"`;
- image chỉ có PyTorch CPU. Muốn GPU, dùng gói cài hoặc `pip install` như mục dưới.

Kiểm thử image (cho người phát triển): `python scripts/test_container.py` build image rồi chạy `--version`, `selfcheck`, `design` → `train.py` → `predict.py` trên thư mục gắn vào, `serve` và `ui` qua cổng publish. Job CI `container` chạy script này bằng Podman trên mỗi commit.

## GPU NVIDIA

Gói cài dùng PyTorch **CPU** (bản CUDA nặng vài GB, vượt giới hạn tệp của GitHub Releases). Máy có GPU NVIDIA nên cài DLNet bằng pip trong một môi trường Python, với PyTorch CUDA:

```bash
./scripts/setup_env.sh --gpu                          # Linux / macOS (từ mã nguồn)
powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1 -Gpu   # Windows
# hoặc: pip install torch (bản CUDA theo pytorch.org) rồi pip install "dlnet[design] @ <wheel trong Release>"
```

## Quy trình phát hành (cho người phát triển)

Mỗi tag phát hành **bắt buộc** có đủ gói cài cho mọi nền tảng trong `scripts/release.py` (`PLATFORMS`: Windows x64, macOS arm64, Linux x64, Linux arm64) (agent.md §11). `release.yml` gọi `installers.yml`: trên runner Windows, macOS (arm64) và Linux, `scripts/build_installer.py` build chương trình bằng PyInstaller (`packaging/dlnet.spec`), **chạy thử chính chương trình đã đóng gói** (`--version`, `selfcheck`, một lần `design` rồi `run-script train.py` và `predict.py` của gói đầu ra, `serve` trả lời `/healthz` và MCP `initialize`), rồi mới tạo gói cài (Inno Setup trên Windows, `pkgbuild` trên macOS, `dpkg-deb` trên Linux). Sau đó `scripts/test_install.py` **cài thật gói vừa tạo** như người dùng (Windows `setup.exe /VERYSILENT /CURRENTUSER`, macOS `sudo installer -pkg`, Linux `sudo apt-get install ./*.deb`), kiểm `SHA256SUMS`, gọi `dlnet` từ PATH mới (`--version` khớp tên tệp, `selfcheck`, `serve` trả lời `/healthz`), rồi gỡ và kiểm chương trình cùng mục PATH đã bị xóa. Chỉ khi `python scripts/release.py assets` thấy đủ mọi tệp, tag và GitHub Release mới được tạo (kèm wheel và sdist). Một hệ điều hành build hoặc chạy thử thất bại thì phiên bản đó chưa được phát hành; sửa rồi đẩy lại, `release.yml` tự chạy lại.

Build thử trên máy của bạn (tạo gói của hệ điều hành đang dùng vào `dist/installers/`):

```bash
pip install ".[design]" "pyinstaller==6.22.0"     # torch CPU trước, nếu không muốn gói vài GB
python scripts/build_installer.py
```

Chưa có: chữ ký số (Windows Authenticode, Apple notarization), gói Homebrew / winget, Windows ARM. **macOS Intel không được hỗ trợ**: PyTorch ngừng phát hành bản cho macOS x86_64 từ 2.3, còn DLNet dùng PyTorch 2.15 và chưa được kiểm với PyTorch 2.2.
