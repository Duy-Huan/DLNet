# Cài DLNet như một phần mềm (Windows, macOS, Linux)

Từ v1.5, mỗi bản phát hành trên GitHub (trang **Releases**, tag `vX.Y.Z`) có sẵn gói cài cho ba hệ điều hành. Gói cài chứa sẵn Python, PyTorch (CPU) và mọi thư viện DLNet cần: **không cần cài Python**. Sau khi cài, lệnh `dlnet` dùng được trong terminal, giống `python -m dlnet`.

| Hệ điều hành | Gói cài | Bản chạy ngay (không cài) |
|---|---|---|
| Windows 10/11 x64 | `DLNet-<ver>-windows-x64-setup.exe` | `DLNet-<ver>-windows-x64.zip` |
| macOS 14+ Apple silicon (M1–M4) | `DLNet-<ver>-macos-arm64.pkg` | `DLNet-<ver>-macos-arm64.tar.gz` |
| Linux x64 (glibc ≥ 2.35: Ubuntu 22.04+, Debian 12+, ...) | `DLNet-<ver>-linux-x64.deb` | `DLNet-<ver>-linux-x64.tar.gz` |

Mỗi hệ điều hành có `SHA256SUMS-<os>-<arch>.txt` để kiểm tra tệp tải về.

## Cài đặt

**Windows**: chạy `DLNet-<ver>-windows-x64-setup.exe`. Không có quyền quản trị thì chọn "Install for me only". Bộ cài thêm `dlnet` vào PATH (mở terminal mới sau khi cài) và tạo hai mục trong Start menu: *DLNet MCP server* (chạy `dlnet serve`, workspace ở `Documents\dlnet-workspace`) và *DLNet command prompt*. Gỡ trong *Settings → Apps*. Windows SmartScreen có thể cảnh báo vì bộ cài chưa được ký số: chọn *More info → Run anyway*.

**macOS**: mở `DLNet-<ver>-macos-arm64.pkg`. Gói chưa được Apple công chứng (notarize), nên lần đầu macOS chặn: chuột phải vào tệp → *Open*, hoặc *System Settings → Privacy & Security → Open Anyway*. DLNet được cài vào `/Library/DLNet`, lệnh `dlnet` ở `/usr/local/bin`. Gỡ: `sudo rm -rf /Library/DLNet /usr/local/bin/dlnet && sudo pkgutil --forget io.github.dlnet`. Trên Apple silicon, PyTorch dùng được GPU qua MPS (`--device mps`).

**Linux**:

```bash
sudo apt install ./DLNet-<ver>-linux-x64.deb      # cài vào /opt/dlnet, lệnh /usr/bin/dlnet
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

## GPU NVIDIA

Gói cài dùng PyTorch **CPU** (bản CUDA nặng vài GB, vượt giới hạn tệp của GitHub Releases). Máy có GPU NVIDIA nên cài DLNet bằng pip trong một môi trường Python, với PyTorch CUDA:

```bash
./scripts/setup_env.sh --gpu                          # Linux / macOS (từ mã nguồn)
powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1 -Gpu   # Windows
# hoặc: pip install torch (bản CUDA theo pytorch.org) rồi pip install "dlnet[design] @ <wheel trong Release>"
```

## Quy trình phát hành (cho người phát triển)

Mỗi tag phát hành **bắt buộc** có đủ gói cài cho ba hệ điều hành (agent.md §11). `release.yml` gọi `installers.yml`: trên runner Windows, macOS (arm64) và Linux, `scripts/build_installer.py` build chương trình bằng PyInstaller (`packaging/dlnet.spec`), **chạy thử chính chương trình đã đóng gói** (`--version`, `selfcheck`, một lần `design` rồi `run-script train.py` và `predict.py` của gói đầu ra, `serve` trả lời `/healthz` và MCP `initialize`), rồi mới tạo gói cài (Inno Setup trên Windows, `pkgbuild` trên macOS, `dpkg-deb` trên Linux). Sau đó `scripts/test_install.py` **cài thật gói vừa tạo** như người dùng (Windows `setup.exe /VERYSILENT /CURRENTUSER`, macOS `sudo installer -pkg`, Linux `sudo apt-get install ./*.deb`), kiểm `SHA256SUMS`, gọi `dlnet` từ PATH mới (`--version` khớp tên tệp, `selfcheck`, `serve` trả lời `/healthz`), rồi gỡ và kiểm chương trình cùng mục PATH đã bị xóa. Chỉ khi `python scripts/release.py assets` thấy đủ mọi tệp, tag và GitHub Release mới được tạo (kèm wheel và sdist). Một hệ điều hành build hoặc chạy thử thất bại thì phiên bản đó chưa được phát hành; sửa rồi đẩy lại, `release.yml` tự chạy lại.

Build thử trên máy của bạn (tạo gói của hệ điều hành đang dùng vào `dist/installers/`):

```bash
pip install ".[design]" "pyinstaller==6.22.0"     # torch CPU trước, nếu không muốn gói vài GB
python scripts/build_installer.py
```

Chưa có: chữ ký số (Windows Authenticode, Apple notarization), macOS Intel, Linux ARM, gói Homebrew / winget.
