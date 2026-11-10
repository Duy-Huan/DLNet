---
name: setup-env
description: Tạo hoặc sửa môi trường Python tái lập (.venv) cho DLNet trên Linux/macOS/Windows, không commit venv.
triggers: venv, virtualenv, environment, môi trường, cài đặt, install, requirements, setup, selfcheck, torch
---
# Môi trường Python

**Không bao giờ commit `.venv`** (6 GB, `libtorch_cuda.so` 520 MB so với giới hạn 100 MB/file của GitHub; đường dẫn tuyệt đối và binary riêng từng hệ điều hành). Mục tiêu "pull về rồi chạy được" đạt bằng script:

1. Người dùng: `./scripts/setup_env.sh` (Linux/macOS) hoặc `scripts\setup_env.ps1` (Windows), `--gpu`/`-Gpu` cho torch CUDA, rồi kích hoạt `.venv` và chạy `dlnet selfcheck`.
2. Phụ thuộc khai báo một chỗ: `pyproject.toml` (`design` = chạy, `dev` = pytest/ruff/pyyaml). `requirements*.txt` chỉ trỏ vào các extras đó. Thêm thư viện: sửa extras, không sửa script.
3. torch cài trước từ chỉ mục CPU (`download.pytorch.org/whl/cpu`), lỗi mạng thì tự lùi về PyPI. macOS luôn dùng PyPI.
4. Sau khi sửa script hoặc phụ thuộc: chạy script trên bản sao sạch (chỉ file git theo dõi) rồi `dlnet selfcheck`; job CI `env-setup` kiểm chứng trên ba hệ điều hành, đọc kết quả trước khi báo "chạy được".
5. Sự cố hay gặp: thiếu `python3-venv` (Debian/Ubuntu: `sudo apt install python3-venv`); PowerShell chặn script (dùng `-ExecutionPolicy Bypass`); Python ngoài 3.10-3.13; mạng chặn pytorch.org (script tự lùi về PyPI, nặng hơn).
