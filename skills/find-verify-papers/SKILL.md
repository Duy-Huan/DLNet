---
name: find-verify-papers
description: Tìm paper liên quan, tải vào ref/ và đối chiếu mã arXiv với tiêu đề trước khi trích dẫn.
triggers: paper, arXiv, ref, citation, literature, survey
---
# Tìm và kiểm chứng paper

1. Tìm bằng công cụ tìm kiếm paper hoặc WebSearch. Ưu tiên survey và bài gốc của phương pháp.
2. Tải PDF arXiv vào `ref/` với tên `<nhóm>_<tên>_<arXiv id>.pdf` (nhóm 01 đến 07, xem `ref/README.md`).
3. **Đối chiếu tiêu đề trong trang đầu PDF với mã arXiv** (đã từng nhầm id 2008.11300, đúng là 2008.05441). Sai thì xóa và tải lại.
4. Thêm một dòng vào bảng `ref/README.md` nếu là nhóm mới, và vào "Tài liệu tham khảo" trong `agent.md`.
5. Khi mô tả ý của paper, chỉ nêu điều đã đọc. Điều gì từ trí nhớ thì ghi "chưa đối chiếu".
