# skills/ — quy trình đã đúc kết cho DLNet

Mỗi skill là một thư mục `skills/<tên>/SKILL.md`. Frontmatter có `name`, `description`, `triggers` để tìm kiếm.

**Quy tắc (xem `agent.md` §15):** trước khi làm bất kỳ việc nào, đọc bảng dưới, chọn skill khớp nhất rồi làm theo. Nếu không skill nào khớp, làm việc đó, rồi thêm skill mới vào đây.

Tìm nhanh:

```bash
grep -ril "<từ khóa>" skills/*/SKILL.md      # ví dụ: plugin, split, leakage, branch
```

| Skill | Dùng khi | Từ khóa |
|---|---|---|
| [add-domain-plugin](add-domain-plugin/SKILL.md) | Thêm loại dữ liệu mới (image, timeseries, ...) | plugin, domain, detect, prepare, loader |
| [add-model-family](add-model-family/SKILL.md) | Thêm họ mô hình / module vào thư viện của một plugin | model, family, library, tensor, Tucker, TT, SPACE |
| [evaluate-honestly](evaluate-honestly/SKILL.md) | Chia dữ liệu, chọn baseline tham chiếu, kết luận "thắng/không thắng" | split, leakage, reference, verdict, acceptance |
| [run-on-real-data](run-on-real-data/SKILL.md) | Chạy thử DLNet trên dữ liệu thật, báo cáo kết quả | experiment, dataset, budget, report |
| [fix-ranking-noise](fix-ranking-noise/SKILL.md) | Thứ hạng Top-K không ổn định / val nhiễu | ranking, noise, cross-validation, seeds |
| [audit-and-regression](audit-and-regression/SKILL.md) | Audit mã, thêm regression test, sửa CI | audit, regression, test, bug, CI |
| [git-dev-main](git-dev-main/SKILL.md) | Commit, push, CI/CD, merge | git, branch, dev, main, CI, promote |
| [release](release/SKILL.md) | Phát hành phiên bản, cập nhật CHANGELOG, tag | release, version, tag, changelog, semver |
| [find-verify-papers](find-verify-papers/SKILL.md) | Tìm và tải paper vào `ref/` | paper, arXiv, ref |
