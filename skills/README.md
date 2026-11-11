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
| [quality-audit](quality-audit/SKILL.md) | Chạy đội agent kiểm định chất lượng (độ tin cậy, mức hiện đại, khả thi, thuật toán) theo `qa/CHECKLIST.md` | quality, audit, QA, kiểm định, checklist, agent team |
| [gpu-device](gpu-device/SKILL.md) | Viết/kiểm mã chạy trên GPU (CUDA, MPS) khi không có GPU; mixed precision | gpu, cuda, device, meta, amp, bf16 |
| [setup-env](setup-env/SKILL.md) | Tạo/sửa môi trường Python, cài đặt, requirements | venv, environment, install, requirements, selfcheck |
| [implement-standard-algorithm](implement-standard-algorithm/SKILL.md) | Thêm/sửa bộ ước lượng chất lượng mạng hoặc hyperparameter theo thuật toán chuẩn | proxy, hyperparameter, lr, batch, SynFlow, ZiCo, Hyperband |
| [find-verify-papers](find-verify-papers/SKILL.md) | Tìm và tải paper vào `ref/` | paper, arXiv, ref |
| [deep-unfolding](deep-unfolding/SKILL.md) | Thêm/sửa mạng unfolding cho bài toán nghịch, so với thuật toán lặp gốc | unfolding, LISTA, ISTA, inverse problem |
| [meta-recommender](meta-recommender/SKILL.md) | Dùng, đo, sửa recommender học từ kho kết quả | meta-learning, meta store, family order |
| [module-zoo](module-zoo/SKILL.md) | Thêm module vào kho, mở rộng ngữ pháp / đột biến, đối chiếu mã gốc, chạy tìm kiếm kiến trúc | zoo, op, grammar, mutation, evolve, backbone |
| [seed-selection](seed-selection/SKILL.md) | Phân tích dữ liệu ảnh, ngữ cảnh, chọn kiến trúc hạt giống, Hyperband, aging evolution | seed, datacard, context, hyperband, portfolio |
| [ai-advisor](ai-advisor/SKILL.md) | Tùy chọn AI: advisor (Claude Code, Codex, Gemini/Antigravity), MCP server `dlnet mcp` | AI, advisor, MCP, claude, codex |
