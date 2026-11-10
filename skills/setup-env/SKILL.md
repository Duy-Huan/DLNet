---
name: setup-env
description: Tạo hoặc sửa môi trường Python tái lập (.venv) cho DLNet trên Linux/macOS/Windows, không commit venv.
triggers: venv, virtualenv, environment, môi trường, cài đặt, install, requirements, setup, selfcheck, torch
---
# Môi trường Python

**Không bao giờ commit `.venv`** (6 GB, `libtorch_cuda.so` 520 MB so với giới hạn 100 MB/file của GitHub; đường dẫn tuyệt đối và binary riêng từng hệ điều hành). Mục tiêu "pull về rồi chạy được" đạt bằng script:

0. Đường đơn giản (ưu tiên): Python 3.12 (`.python-version`) → `python -m venv .venv` → activate → `pip install -r requirements.txt` → `python -m dlnet selfcheck`. `requirements.txt` ghim `name==version` đã kiểm tra; đổi phiên bản thì cập nhật cả `requirements*.txt` lẫn bản đã chạy được trong CI (test `test_env_setup` bắt thiếu hoặc không ghim).
1. Tùy chọn một lệnh: `./scripts/setup_env.sh` (Linux/macOS) hoặc `scripts\setup_env.ps1` (Windows), `--gpu`/`-Gpu` cho torch CUDA, rồi kích hoạt `.venv` và chạy `dlnet selfcheck`.
2. Phụ thuộc: `pyproject.toml` (extras `design`, `dev`, không ghim) là danh sách tên; `requirements*.txt` là bản ghim cho Python 3.12. Thêm thư viện: thêm vào cả hai (test bắt lệch).
3. torch cài trước từ chỉ mục CPU (`download.pytorch.org/whl/cpu`), lỗi mạng thì tự lùi về PyPI. macOS luôn dùng PyPI.
4. Sau khi sửa script hoặc phụ thuộc: chạy script trên bản sao sạch (chỉ file git theo dõi) rồi `dlnet selfcheck`; job CI `env-setup` kiểm chứng trên ba hệ điều hành, đọc kết quả trước khi báo "chạy được".
5. Sự cố hay gặp: thiếu `python3-venv` (Debian/Ubuntu: `sudo apt install python3-venv`); PowerShell chặn script (dùng `-ExecutionPolicy Bypass`); Python ngoài 3.10-3.13; mạng chặn pytorch.org (script tự lùi về PyPI, nặng hơn).
