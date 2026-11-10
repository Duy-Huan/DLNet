---
name: git-dev-main
description: Quy trình Git của DLNet: chỉ hai nhánh dev và main, CI xanh thì tự đưa dev lên main.
triggers: git, branch, dev, main, commit, push, CI, promote, merge
---
# Git dev → main

**Chỉ có hai nhánh: `dev` và `main`. Không tạo nhánh khác.**

1. Làm việc trên `dev`: `git checkout dev && git pull origin dev`.
2. Trước khi commit: `ruff check .` và `python3 -m pytest -q` (dùng `python3 -m pytest`, không dùng binary `pytest` vì lệch môi trường).
3. Commit rõ ràng (một thay đổi một commit), `git push -u origin dev`. Push lỗi do mạng thì thử lại tối đa 4 lần (2s, 4s, 8s, 16s).
4. `promote.yml` tự gọi CI, nếu xanh thì tạo PR `dev → main` và merge. Không push thẳng vào `main`.
5. Kiểm tra kết quả Actions trước khi nói "đã lên main". Chưa kiểm tra thì nói là chưa kiểm tra.
6. Không commit dữ liệu thô, khóa hoặc bí mật. Kết quả chạy lưu ngoài repo.
7. Thiết lập GitHub thủ công (quyền Actions, auto-merge, branch protection): `.github/README.md`.
