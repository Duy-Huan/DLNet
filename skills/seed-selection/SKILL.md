---
name: seed-selection
description: Phân tích đặc trưng dữ liệu ảnh (data card), dùng ngữ cảnh bài toán và các thuật toán chuẩn (Borda, portfolio Auto-sklearn 2.0, vote proxy, Hyperband, aging evolution) để chọn kiến trúc hạt giống; thêm/sửa quy tắc và đo xem chúng có giúp không.
triggers: seed, hạt giống, data card, datacard, meta-feature, context, ngữ cảnh, hyperband, aging evolution, regularized evolution, portfolio, borda, select architecture
---
# Chọn hạt giống theo dữ liệu (agent.md §16.5)

1. **Dùng**: `python -m dlnet design --data X.npz --context "medical CT, small lesions" --seeds 12 [--scheduler hyperband] [--meta-store runs.jsonl] [--seed-proxies]`. Log in `[datacard]` (đặc trưng), `[seeds]` (quy tắc đã kích hoạt, số giữ lại). `leaderboard.json` có khối `seed_selection` (data card, quy tắc, xếp hạng).
2. **Đặc trưng dữ liệu** ở `dlnet/datacard.py`: mỗi đặc trưng có công thức và nguồn trong docstring; cái nào từ trí nhớ ghi rõ. Thêm đặc trưng mới thì thêm test đáp án đóng vào `tests/test_seeds.py` (ví dụ nhiễu 1/f phải ra slope ≈ 2).
3. **Đặc tính kiến trúc** đọc từ module trong spec (`seeds.OP_TRAITS`). Op mới trong zoo: thêm dòng vào `OP_TRAITS`, nếu không selector không biết nó có ích khi nào.
4. **Quy tắc** (`seeds.trait_weights`, `seeds.CONTEXT`) là heuristic của DLNet, mỗi quy tắc in lý do. Không thêm quy tắc mà không chạy `scripts/bench_seeds.py` trước và sau; ghi số vào `agent.md` §16.5.
5. **Thuật toán chuẩn**: tổng hợp xếp hạng bằng Borda có trọng số; meta-learning: kNN (Feurer 2015) + portfolio tham lam (Auto-sklearn 2.0, Algorithm 1); proxy: vote đa số (Abdelfattah 2021) của synflow/naswot/snip, ZiCo theo mã chính thức; lịch huấn luyện: Hyperband (Li 2018, Algorithm 1, lệch Table 1 đã ghi); tiến hóa: aging evolution (Real 2019, Algorithm 1, tournament S). Sửa công thức thì theo skill `implement-standard-algorithm`.
6. **Đo**: `python scripts/bench_seeds.py --data a.npz b.npz --budget 120s --n 10 --reps 2`: so `all` / `random` / `data` / `data_hb` cùng ngân sách, kèm Spearman của prior và các proxy với validation rung 1. Selector chỉ được bật mặc định (`--seeds` > 0) khi `data` hơn `random` ổn định qua dữ liệu và seed.
