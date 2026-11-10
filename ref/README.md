# ref/ — Tài liệu tham khảo cho DLNet

PDF tải từ arXiv, tên file = `<nhóm>_<tên>_<arXiv id>.pdf`. Mã arXiv đã đối chiếu tiêu đề với trang đầu PDF.

| Nhóm | Chủ đề | Dùng cho phần nào của `agent.md` |
|---|---|---|
| 01 | Paper nền tảng (Modular Deep Learning) | Khung 4 trục: computation/routing/aggregation/training |
| 02 | AutoML & NAS (survey, DARTS, NAS-RL, Once-for-All, regularized evolution, Auto-sklearn 2.0 portfolio, zero-shot AutoML, Task2Vec) | Search & Refine, Recommender, chọn hạt giống (§16.5) |
| 03 | Ước lượng không cần/ít huấn luyện: zero-cost proxy, scaling law, intrinsic dimension | Estimator |
| 04 | Tensor decomposition cho mạng nơ-ron (CP, Tucker, TT, TR, TensorLy) | Tensor module, ước lượng rank |
| 05 | Deep unfolding / algorithm unrolling | Unfolded module |
| 06 | PEFT: LoRA, adapters, prefix/prompt tuning, Compacter, MAD-X, AdapterFusion, Hyper-X, task arithmetic | Computation / Aggregation module |
| 07 | MoE, hypernetwork, lottery ticket | Routing, sparse composition |
| `optimazeDL/` | Thuật toán chuẩn ước lượng chất lượng mạng (zero-cost proxy, WeightWatcher), tìm/ước lượng hyperparameter (LR range test, noise scale, Hyperband, BOHB, ...), benchmark NAS + mã chính thức | Estimator, Search, `dlnet/hparams.py` (agent.md §16; chỉ mục `optimazeDL/README.md`) |

Lưu ý: Kossaifi & Cichocki (Tensor Networks Part 1/2) là chuyên khảo dài; Elsken NAS survey là bản arXiv 1808.05377 (cùng nội dung bản JMLR).
