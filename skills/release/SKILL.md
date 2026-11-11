---
name: release
description: Phát hành phiên bản DLNet theo SemVer: cập nhật CHANGELOG, tăng version, để CI/CD tự merge dev vào main và tạo tag vX.Y.Z cùng GitHub Release.
triggers: release, version, tag, changelog, publish, semver, phát hành, bump
---
# Release

Chi tiết đầy đủ: `agent.md` §11.2. Chỉ làm trên `dev`; không tạo nhánh, không tạo tag bằng tay.

1. Mọi thay đổi người dùng thấy được đã có dòng trong `CHANGELOG.md` mục `[Unreleased]` (nếu thiếu, thêm trước).
2. Chọn số phiên bản: trước 1.0 thêm tính năng hoặc đổi hành vi = minor, sửa lỗi hoặc tài liệu = patch.
2b. `README.md` và tài liệu `doc/` đã cập nhật theo các thay đổi (`agent.md` §17 quy tắc 0); `python scripts/gen_docs.py --check` và `python scripts/release.py docs` qua.
3. Chạy `python scripts/release.py bump X.Y.Z` (sửa mọi chỗ ghi phiên bản, mở mục CHANGELOG, phiên bản tài liệu, sinh lại `doc/cli.md`, `doc/zoo.md`). Một commit `release: vX.Y.Z` gồm: `[Unreleased]` đổi thành `[X.Y.Z] - YYYY-MM-DD` kèm mục `[Unreleased]` rỗng mới; `version` trong `pyproject.toml`; `__version__` trong `dlnet/__init__.py`; link cuối `CHANGELOG.md`.
4. `python scripts/release.py check && ruff check . && ruff format --check . && python3 -m pytest -q -m "not e2e"`.
5. `git push origin dev` (xem skill `git-dev-main`). `promote.yml` tự merge vào `main`, `release.yml` tự tạo tag và Release.
6. Kiểm tra Actions và trang Releases: có tag `vX.Y.Z`, ghi chú đúng. Chưa kiểm tra thì nói là chưa kiểm tra.
7. Auto-merge bị trễ (PR còn mở): sau khi PR merge, chạy tay workflow `Release`.
8. Bản lỗi: không xóa hay đẩy lại tag; phát hành patch kế tiếp.
