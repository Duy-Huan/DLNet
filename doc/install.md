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

Từ v1.8.2, repo có `Containerfile`: một image Linux gồm Python 3.12, PyTorch 2.15 (CPU) và DLNet, khoảng 1,4 GB. Image chạy trên mọi máy có **Podman**: Linux chạy trực tiếp, Windows và macOS qua máy ảo `podman machine`. Docker cũng dùng được. Dữ liệu và kết quả nằm trong một thư mục trên máy bạn, gắn vào container ở `/workspace`; container chạy xong là xoá (`--rm`), không mất gì.

### 1. Cài Podman

| Hệ điều hành | Cài | Chuẩn bị (một lần) |
|---|---|---|
| Windows 10/11 | Podman Desktop hoặc gói `podman` từ podman.io; cần WSL2 (`wsl --install`) | `podman machine init --cpus 4 --memory 8192` rồi `podman machine start` |
| macOS | Podman Desktop, hoặc `brew install podman` | `podman machine init --cpus 4 --memory 8192` rồi `podman machine start` |
| Linux | `sudo apt install podman` / `sudo dnf install podman` | không cần |

Trên Windows và macOS, container chạy trong máy ảo của Podman. Máy ảo mặc định chỉ có 2 GB RAM, không đủ để huấn luyện, nên hãy cho nó **ít nhất 8 GB**. Máy ảo đã có sẵn thì chỉnh lại:

```bash
podman machine stop
```

```bash
podman machine set --cpus 4 --memory 8192
```

```bash
podman machine start
```

Kiểm tra: `podman info` chạy không lỗi là xong.

### 2. Lấy mã và build image

```bash
git clone -b dev https://github.com/Duy-Huan/DLNet.git
```

```bash
cd DLNet
```

```bash
podman build -t dlnet -f Containerfile .
```

Lần đầu mất vài phút (tải PyTorch CPU). Thử ngay:

```bash
podman run --rm dlnet selfcheck
```

In ra `selfcheck OK` là image dùng được.

### 3. Chạy lệnh trên dữ liệu của bạn

Tạo một thư mục làm việc, ví dụ `work`, chép dữ liệu vào đó (`work/data.csv`), rồi gắn nó vào `/workspace`. **Mọi đường dẫn trong lệnh `dlnet` tính từ thư mục này**: `--data data.csv`, `--out runs/a` sẽ ghi ra `work/runs/a` trên máy bạn.

Linux (thêm `:Z` khi máy có SELinux: Fedora, RHEL):

```bash
podman run --rm -v ./work:/workspace:Z dlnet design --data data.csv --target y --budget 10m --out runs/a
```

macOS:

```bash
podman run --rm -v "$PWD/work":/workspace dlnet design --data data.csv --target y --budget 10m --out runs/a
```

Windows PowerShell:

```powershell
podman run --rm -v "${PWD}\work:/workspace" dlnet design --data data.csv --target y --budget 10m --out runs/a
```

Windows cmd: thay `${PWD}\work` bằng `%cd%\work`.

Mọi lệnh khác viết giống hệt, chỉ đổi phần sau `dlnet`, ví dụ `selfcheck`, `paper import`, `run-script runs/a/top1_.../train.py --data data.csv --target y`, `predict`.

Cho gọn, có thể đặt một lệnh tắt. Trên Linux / macOS, thêm dòng sau vào `~/.bashrc` hoặc `~/.zshrc`:

```bash
alias dlnet-c='podman run --rm -v "$PWD":/workspace:Z dlnet'
```

Sau đó, đứng trong thư mục dữ liệu, gõ `dlnet-c design --data data.csv --target y --out runs/a`. Trên PowerShell, thêm vào `$PROFILE`:

```powershell
function dlnet-c { podman run --rm -v "${PWD}:/workspace" dlnet @args }
```

### 4. Giao diện web và MCP server

Giao diện web (`dlnet ui`). Sau khi chạy, mở `http://127.0.0.1:8766/` trong trình duyệt; Ctrl+C để dừng.

```bash
podman run --rm -p 127.0.0.1:8766:8766 -v ./work:/workspace:Z dlnet ui --workspace /workspace
```

MCP server (`dlnet serve`) cho Claude Code / Codex / Antigravity. Token tự đặt, client gửi kèm:

```bash
podman run --rm -p 127.0.0.1:8765:8765 -v ./work:/workspace:Z -e DLNET_TOKEN=<token> dlnet serve --host 0.0.0.0 --workspace /workspace
```

Muốn chạy nền, thay `--rm` bằng `-d --name dlnet-ui`. Xem log bằng `podman logs dlnet-ui`, dừng bằng `podman stop dlnet-ui`.

Vì sao viết như vậy:
- cổng bên ngoài phải **giống** cổng bên trong: muốn dùng 9000 thì viết `-p 127.0.0.1:9000:9000` cùng `ui --port 9000`. UI chỉ trả lời request ghi đúng `127.0.0.1` / `localhost` và cổng của nó, để chặn trang web lạ gọi vào;
- trong image, UI nghe trên 0.0.0.0 (biến `DLNET_UI_BIND`, hoặc `dlnet ui --bind`) để cổng chuyển tiếp tới được. Luôn viết `127.0.0.1:` trước cổng để máy khác trong mạng không truy cập được;
- `serve` trên 0.0.0.0 bắt buộc có token (`DLNET_TOKEN`), không có thì từ chối chạy.

### 5. Cập nhật lên bản mới

```bash
git pull
```

```bash
podman build -t dlnet -f Containerfile .
```

```bash
podman image prune -f
```

Lệnh cuối xoá các image cũ không còn tên. Dữ liệu trong `work/` không bị ảnh hưởng.

### 6. Lỗi thường gặp

| Hiện tượng | Nguyên nhân và cách sửa |
|---|---|
| Lệnh dừng giữa chừng, báo `Killed` hoặc mã thoát 137 | Hết RAM. Trên Windows / macOS, tăng RAM của `podman machine` (mục 1); hoặc giảm `--budget`. |
| `Permission denied` khi ghi `/workspace` trên Linux | SELinux: thêm `:Z` sau đường dẫn gắn (`-v ./work:/workspace:Z`). |
| File trong `work/` thuộc về root (khi dùng Docker) | Thêm `--user "$(id -u):$(id -g)"` vào `docker run`. Podman rootless không bị việc này. |
| Trang UI báo `this interface only answers on 127.0.0.1 / localhost` | Mở bằng `127.0.0.1` / `localhost`, không dùng IP máy; cổng ngoài và trong phải giống nhau. |
| `address already in use` | Cổng đang bận: chọn cổng khác, cùng một số cho cả `-p` và `--port`. |
| `data.csv` không tìm thấy | Đường dẫn tính từ `/workspace`: file phải nằm trong thư mục đã gắn, ghi `--data data.csv` chứ không ghi đường dẫn của máy bạn. |
| Build lỗi khi tải gói (mạng công ty) | Podman tự chuyển biến `HTTP_PROXY` / `HTTPS_PROXY` của máy vào lúc build; kiểm tra các biến này đã đặt. |
| Muốn dùng GPU | Image chỉ có PyTorch CPU. Dùng gói cài hoặc `pip install` như mục **GPU NVIDIA** dưới đây. |

### 7. Kiểm thử image (cho người phát triển)

`python scripts/test_container.py` (cần Python 3 trên máy) build image rồi chạy thử như người dùng:
- `--version` và `selfcheck`;
- `design` → `train.py` → `predict.py` trên một thư mục gắn vào, kiểm tra file ghi ra thuộc về user;
- `serve` và `ui` qua cổng publish.

Thêm `--engine docker` để dùng Docker. Job CI `container` chạy script này bằng Podman trên mỗi commit.

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
