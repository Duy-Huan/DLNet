# agent.md — Kiến trúc dự kiến của DLNet

> Framework **tổng quát** tự động thiết kế mạng deep learning: đầu vào là **dữ liệu bất kỳ** (tabular, chuỗi thời gian, ảnh, ảnh siêu phổ HSI, tín hiệu, tensor…), DLNet **phân tích dữ liệu, tự ghép và so sánh nhiều kiến trúc**, rồi trả về **danh sách Top-K baseline PyTorch đã xếp hạng** (top1, top2, …) để người dùng mang đi huấn luyện đầy đủ. Mục tiêu: thay cho việc thử tay, tiết kiệm thời gian và tìm baseline cho độ chính xác cao nhất.
> Cơ sở lý thuyết: *Modular Deep Learning* (Pfeiffer, Ruder, Vulić, Ponti — arXiv:2302.11529).

---

## 1. Mục tiêu & phạm vi

| | Nội dung |
|---|---|
| **Input** | Dữ liệu + nhãn (+ loại tác vụ, hoặc `auto` để DLNet tự nhận) + ngân sách (thời gian GPU, VRAM, tham số) + `K` (số baseline muốn nhận). |
| **Output** | **Top-K baseline** xếp hạng, mỗi cái là một gói PyTorch độc lập (`model.py`, `config.yaml`, `train.py`, `report.md`, test) + `leaderboard` so sánh + công thức huấn luyện đầy đủ gợi ý (xem §12.3). |
| **Tính tổng quát** | Lõi (spec, registry, search, đánh giá, đóng gói) không phụ thuộc loại dữ liệu; mỗi loại dữ liệu là một **domain plugin** (§14). HSI là plugin đầu tiên, không phải phạm vi duy nhất. |
| **Giá trị cốt lõi** | **Tự động hóa việc thử tay**: một lệnh ra Top-K baseline kèm bằng chứng, tiết kiệm thời gian đề xuất baseline (§12). |
| **Cách làm** | (1) Phân tích dữ liệu → (2) ước lượng "độ phức tạp" & cấu trúc → (3) chọn họ kiến trúc + module → (4) tìm kiếm cấu hình trong không gian có ràng buộc → (5) đóng gói. |
| **Ngoài phạm vi** | Viết paper, đảm bảo tính mới khoa học; thu thập dữ liệu; NAS từ-đầu quy mô lớn; huấn luyện LLM. |

### Lưu ý quan trọng về paper
Paper 2302.11529 là **survey về học sâu dạng mô-đun**, *không* phải thuật toán AutoML hay tensor decomposition. Ta dùng nó làm **ngôn ngữ thiết kế chuẩn hóa**: mọi mạng được mô tả bằng bốn trục độc lập
**Computation · Routing · Aggregation · Training setting**. Phần *ước lượng* (profiling, scaling law, tensor rank) và *unfolding* là đóng góp của DLNet, ghép lên khung này.

---

## 2. Ánh xạ khung của paper → kiến trúc DLNet

Một mô-đun hóa hàm theo paper: `y = g_γ( { f_i(x) · r_ρ(x)_i } )`.

| Trục (paper) | Lựa chọn trong paper | Vai trò trong DLNet |
|---|---|---|
| **Computation `f`** | Parameter composition (LoRA, sparse/masks, intrinsic dim.) · Input composition (prompt/prefix tuning) · Function composition (adapter, Compacter) · Hypernetwork | `ModuleZoo`: thư viện module chuẩn hóa. **Thêm**: *Tensor-decomposed module* (CP/Tucker/TT/TR) và *Unfolded module*. |
| **Routing `r`** | Fixed · Learned (soft: MoE; hard: top-k, Routing Networks, PathNet) · Hypernetwork routing | `Router`: fixed khi có metadata tin cậy (domain/task/ngôn ngữ), learned khi không; có chống module-collapse. |
| **Aggregation `g`** | Parameter (weight merging, task arithmetic) · Representation (weighted avg, AdapterFusion) · Input (concat prompt) · Function (sequential/hierarchical) | `Aggregator`: chọn theo topology mạng. |
| **Training setting** | Multitask chung · Continual · Parameter-efficient transfer (frozen backbone) | `TrainingRecipe`: gắn với ràng buộc dữ liệu (ít nhãn → PEFT; nhiều domain → multitask/routing). |

Quy tắc chọn tắt rút ra từ paper (dùng làm *prior* cho bộ gợi ý, sẽ kiểm chứng thực nghiệm):
- Cần chi phí thấp nhất, thay thế "cắm vào" → **parameter composition** (LoRA, sparse).
- Cực kỳ tiết kiệm tham số, backbone lớn → **input composition**.
- Cần hiệu năng cao nhất, chấp nhận thêm tham số → **function composition** (adapter).
- Có metadata tin cậy → **fixed routing**; không có → **learned routing** + regularizer.

---

## 3. Kiến trúc tổng thể (pipeline)

```
  Data ──► [0 Plugin Router]  nhận loại dữ liệu → chọn domain plugin (§14)
                │                 (loader, split, metric, stem, preset, prior)
                ▼
          [1 Ingest & Profile] ──► DataProfile
                │
                ▼
          [2 Estimator]  (ước lượng độ phức tạp & cấu trúc)
     ┌──────────┬───────────────┬────────────────┬──────────────────┐
     ▼          ▼               ▼                ▼
 Capacity    Tensor-structure  Inverse-problem  Modularity est.
 (scaling)   (rank, unfolding) (có model vật lý?) (domain/task)
     └──────────┴───────────────┴────────────────┘
                │  ProblemSpec
                ▼
          [3 Candidate Generator]  prior + preset + tự ghép từ Library (§13)
                │  hàng chục–trăm ứng viên hợp lệ (đã qua cost model)
                ▼
          [4 Search & Rank]  proxy → short-train → multi-seed (§12.3)
                │  bảng xếp hạng ± std, nhóm "chưa phân biệt được"
                ▼
          [5 Compiler]  ArchSpec ─► nn.Module PyTorch
                │
                ▼
          [6 Packager]  Top-K gói baseline + leaderboard + report
```

### 3.1 Ingest & Profile (`dlnet/data`)
Trích `DataProfile`: modality, shape, kích thước mẫu N, số chiều/kênh, tỉ lệ nhiễu, mất cân bằng lớp, phụ thuộc thời gian/không gian (ACF, FFT phổ), thưa, missing, **metadata nhóm** (domain/task/ngôn ngữ) nếu có. Có cache + kiểm thử tính nhất quán.

### 3.2 Estimator (`dlnet/estimate`) — "ước lượng qua thuật toán tính toán"
Đây là phần người dùng yêu cầu: dùng **thuật toán đo được**, không đoán.

1. **Capacity / độ khó**: 
   - Probe nhanh: huấn luyện model nhỏ ở vài mức N (learning-curve) → ngoại suy scaling law `L(N)=a·N^{-b}+c` để biết dữ liệu có "đủ cho model lớn" không.
   - Intrinsic dimension (TwoNN / MLE), PCA/SVD spectrum → độ rộng hợp lý của lớp ẩn.
   - **Scaling exponent từ intrinsic dimension** (Sharma & Kaplan, `ref/03_scaling-law-data-manifold_2004.10802.pdf`): khi dữ liệu đủ nhiều, loss theo số tham số `L ∝ N^{-α}` với `α ≈ 4/d` (d = intrinsic dimension). Dùng như sau:
     1. Ước lượng `d` bằng TwoNN/MLE trên (embedding của) dữ liệu → `α̂ = 4/d`.
     2. Fit song song `α` thực nghiệm từ learning-curve; so sánh `α̂` với `α` đo được làm kiểm tra chéo.
     3. Từ `α` suy ra **lợi suất giảm dần**: tăng tham số ×k chỉ giảm loss ×`k^{-α}`; nếu `d` lớn (α nhỏ) thì ưu tiên mô hình nhỏ/nén hơn, nếu `d` nhỏ thì mở rộng mô hình có hiệu quả. Kết quả là **dung lượng khuyến nghị** `N*` (tham số) đưa vào `ProblemSpec` làm ngân sách mặc định cho Search.
     - Giới hạn: công thức là lý thuyết cho miền dữ liệu dồi dào và loss cross-entropy/MSE; với N nhỏ chỉ dùng làm prior, luôn xác thực bằng short-train.
   - Zero-cost proxy (grad-norm, SynFlow, NTK condition number) để xếp hạng sơ bộ.
2. **Cấu trúc tensor** (`dlnet/tensor`):
   - Biểu diễn dữ liệu như tensor bậc-k; tính **mode-n unfolding** và phổ singular từng mode → ước lượng **multilinear rank**.
   - Thử CP / Tucker / Tensor-Train / Tensor-Ring; chọn theo lỗi tái tạo vs. số tham số (ngưỡng năng lượng, hoặc BIC/MDL).
   - Kết luận: dữ liệu *low-rank rõ* → dùng lớp phân rã (compress/khởi tạo); *không* → giữ dense.
3. **Có mô hình thuận không?** (bài toán nghịch: khôi phục ảnh, nén cảm biến, sparse coding, tách nguồn, matrix/tensor completion…): nếu có toán tử `A` đã biết hoặc học được → ứng viên **Deep Unfolding** (§5).
4. **Modularity**: nếu có nhóm domain/task → kiểm tra phân bố giữa nhóm (MMD/độ lệch) để quyết định dùng module riêng + routing hay mạng chung.

Đầu ra `ProblemSpec` (JSON) gồm: modality, N, intrinsic-dim, ước lượng rank, cờ `has_forward_model`, cờ `has_groups`, ràng buộc tài nguyên.

### 3.3 Architecture Recommender (`dlnet/recommend`)
- **Giai đoạn 1 — Rule prior** (xác định, giải thích được): bảng luật từ §2 + modality (ảnh→CNN/ViT, chuỗi→TCN/Transformer/SSM, tabular→MLP/FT-Transformer/GBDT-like MLP...).
- **Giai đoạn 2 — Meta-learned** (sau khi có dữ liệu thực nghiệm): regress từ `ProblemSpec` → hiệu năng của từng `ArchSpec` (kho kết quả benchmark nội bộ, như meta-learning/AutoML warm-start).
- Trả về tập ứng viên `ArchSpec` kèm lý do (đầu vào của bước Search & Rank, không phải kết quả cuối).

### 3.4 Search & Refine (`dlnet/search`)
- Không gian **có cấu trúc, nhỏ** (4 trục × siêu tham số) thay vì NAS vô hạn.
- Thuật toán: Bayesian Optimization / Successive Halving / evolutionary; đánh giá bằng zero-cost proxy rồi short-train.
- Đa mục tiêu: accuracy, #params, FLOPs, latency → Pareto front; luôn tôn trọng ràng buộc cứng.
- Mọi lần chạy ghi seed, config, metric (tái lập).
- Kết quả cuối là **bảng xếp hạng Top-K** theo §12.3.

### 3.5 Compiler (`dlnet/compile`)
`ArchSpec` (YAML/dataclass) ⟶ `torch.nn.Module` qua registry. Mỗi module đăng ký `Computation/Router/Aggregator`. Có kiểm tra shape tĩnh và `torch.fx`/`torch.compile` smoke test.

### 3.6 Packager (`dlnet/package`)
Xuất thư mục kết quả; mỗi baseline độc lập, chỉ phụ thuộc PyTorch:
```
dlnet_out/
 ├─ leaderboard.md / leaderboard.json   # xếp hạng, metric ± std, params, FLOPs, thời gian
 ├─ profile.json                        # DataProfile + ProblemSpec
 ├─ reference/                          # kết quả bộ baseline tham chiếu (§12.2)
 ├─ top1_<name>/
 │   ├─ model.py        # nn.Module đã sinh, đọc được, có docstring
 │   ├─ config.yaml     # ArchSpec + hyperparams + seed
 │   ├─ train.py        # vòng huấn luyện (+ eval) với công thức gợi ý
 │   ├─ init.pt         # (tuỳ chọn) khởi tạo từ phân rã tensor / pretrained
 │   ├─ report.md       # vì sao chọn, điểm ± std, chi phí, rủi ro
 │   └─ tests/          # test shape, forward/backward, tái lập
 ├─ top2_<name>/ ...
 └─ topK_<name>/
```

---

## 4. Tensor decomposition trong framework

Hai chỗ dùng, tách bạch:
1. **Trên dữ liệu** (phân tích): rank ước lượng → quyết định kích thước/độ nén (§3.2).
2. **Trên trọng số** (module): thay lớp dense/conv bằng dạng phân rã.

| Phân rã | Dùng cho | Ghi chú |
|---|---|---|
| CP | Conv kernel, embedding | Rất gọn, nhưng huấn luyện nhạy |
| Tucker | Conv / lớp nhiều mode | Cân bằng nén–độ chính xác |
| Tensor-Train / Tensor-Ring | FC lớn, embedding, attention proj. | Nén mạnh, mở rộng tốt |
| LoRA (low-rank) | Parameter composition (paper) | Là trường hợp ma trận của phân rã |

Quan hệ với paper: phân rã hạng thấp chính là một dạng **parameter composition** (`θ ⊕ Δθ` với Δθ low-rank/tensor-factorized). DLNet tổng quát hóa LoRA → **Tensor-LoRA** như một `Computation` module.

### Tensor-LoRA: cơ sở và thiết kế
Tensor-LoRA = adapter/delta-weight được tham số hóa bằng cấu trúc tensor thay vì cặp ma trận `B·A`. Dựa trên ba nguồn trong `ref/`:

| Nguồn | Đóng góp được dùng |
|---|---|
| **Tensorizing Neural Networks** (Novikov, 1509.06569) | Reshape ma trận `W ∈ R^{M×N}` thành tensor bậc-d (`M=∏m_k`, `N=∏n_k`) rồi biểu diễn bằng **Tensor-Train**; số tham số giảm từ `MN` xuống `∑ r_{k-1} m_k n_k r_k`. Dùng cho lớp FC/projection lớn. |
| **Tensor Ring** (Zhao, 1606.05535) | Vòng lõi tensor, không cố định rank biên như TT → tham số hóa linh hoạt, bất biến theo hoán vị vòng; dùng khi TT bị nghẽn rank ở hai đầu. |
| **Compacter** (Mahabadi, 2106.04647) | Adapter hạng thấp dạng **tổng tích Kronecker** `W = ∑_i A_i ⊗ (u_i v_iᵀ)`, chia sẻ `A_i` giữa các lớp → cực ít tham số; là mẫu thiết kế để chia sẻ lõi tensor giữa nhiều adapter/module. |

Thiết kế module `TensorLoRA`:
- `Δθ = TT/TR/Kronecker(cores)`, `θ_eff = θ_frozen + (α/r)·Δθ`; backbone đóng băng (training setting = parameter-efficient transfer).
- Tham số điều khiển trong `ArchSpec`: `format ∈ {lora, tt, tr, kronecker}`, `factor_shape` (cách reshape), `rank` (hoặc danh sách rank), `share_cores` (kiểu Compacter).
- Chọn `format` bằng ước lượng ở §3.2: nếu trọng số/ma trận biến đổi có phổ singular suy giảm nhanh theo nhiều mode → TT/TR; ngược lại giữ LoRA (ma trận).
- Khởi tạo: phân rã trọng số pretrained (TT-SVD / TR-SVD) khi có, nếu không thì khởi tạo nhỏ để `Δθ ≈ 0`.
- Kiểm thử bắt buộc: `Δθ` tái tạo đúng shape; số tham số theo công thức; gradient chảy qua mọi lõi; với `rank` đủ lớn tái tạo được LoRA chuẩn.
- Rủi ro đã biết: nén CNN bằng TT/TR dễ mất độ chính xác, CP dễ bất ổn số học (xem `ref/04_stable-lowrank-cp-cnn_2008.05441.pdf`) → mặc định dùng TT/TR cho FC/projection, Tucker cho conv, CP chỉ khi có bước ổn định hóa.

### Kết quả đo M7 (`python scripts/bench_tensor.py`, scikit-learn digits, 3 seed, 40 epoch)
Đã có: `dlnet/tensor_lora.py` (`TensorLoRALinear` với `lora`/`tt`/`tr`/`kronecker`, `apply_tensor_lora`, `merge_tensor_lora`, `tt_svd`), `dlnet/tensor.py` (`hosvd`, `multilinear_rank`, `compress_report`), lớp `TRLinear` và họ `tr_resmlp` trong `dlnet/library.py`. Công thức LoRA (eq. 3), Compacter (eq. 5) và Tensor Ring (eq. 1) đã đối chiếu PDF trong `ref/`; TT-SVD và HOSVD viết từ trí nhớ, có test đáp án đóng.

**A. Nén trọng số** (ResMLP rộng 256, sâu 2, 10 lớp):

| Lớp ẩn | Tham số | Giảm | Test accuracy | Mất so với dense |
|---|---|---|---|---|
| dense (`resmlp`) | 283.914 | 0% | 0,9755 ± 0,0044 | 0 |
| `tt_resmlp` r=16 | 54.538 | 80,8% | 0,9774 ± 0,0034 | −0,0018 |
| `tt_resmlp` r=4 | 29.962 | 89,4% | 0,9719 ± 0,0046 | +0,0036 |
| `tr_resmlp` r=4 | 54.538 | 80,8% | 0,9764 ± 0,0051 | −0,0009 |
| `tr_resmlp` r=2 | 29.962 | 89,4% | 0,9755 ± 0,0022 | 0,0000 |
| `lowrank_mlp` r=16 | 27.658 | 90,3% | 0,9792 ± 0,0026 | −0,0036 |

Mọi chênh lệch nằm trong độ lệch chuẩn qua seed: nén 80–90% không làm giảm accuracy đo được trên bộ này. Low-rank thường (ma trận) đạt kết quả tương đương TT/TR, nên chưa có bằng chứng TT/TR hơn low-rank.

**B. Tensor-LoRA** (pretrain trên chữ số 0–4, thích nghi sang 5–9 với 100 nhãn, backbone đóng băng, head mới luôn được huấn luyện):

| Phương pháp | Tham số huấn luyện | Test accuracy trên 5–9 |
|---|---|---|
| huấn luyện từ đầu | 41.989 | 0,9508 ± 0,0118 |
| fine-tune toàn bộ | 41.989 | 0,9235 ± 0,0279 |
| chỉ head | 645 | 0,5647 ± 0,0498 |
| LoRA r=4 | 3.461 | 0,7978 ± 0,0648 |
| TT r=4 | 3.973 | 0,8142 ± 0,0491 |
| TR r=2 | 3.973 | 0,8342 ± 0,0670 |
| Kronecker n=2, r=1 | 1.373 | 0,6321 ± 0,0545 |

Kết luận trung thực: với backbone nhỏ pretrain trên 5 chữ số, pretrain **không giúp** (huấn luyện từ đầu tốt nhất) và mọi adapter kém fine-tune toàn bộ 9–30 điểm. TT/TR ngang LoRA ở cùng số tham số (chênh lệch trong mức nhiễu). Phép đo này chỉ xác nhận module chạy đúng; muốn đánh giá Tensor-LoRA cần backbone pretrained lớn, DLNet chưa có.

**C. HOSVD** (khối tổng hợp 48×48×40, rank Tucker (6,6,4) + nhiễu 5%): giữ 99% năng lượng mỗi mode cho đúng rank [6,6,4], lưu 0,95% số phần tử, sai số tương đối 0,050 (cận lý thuyết 0,081). HOSVD **chưa** được dùng làm tiền xử lý trong plugin HSI: nhân tử không gian phải fit trên cả ảnh, kể cả pixel test, trái quy tắc "chỉ fit trên train".

---

## 5. Deep Unfolding (khi cần)

Điều kiện kích hoạt: `has_forward_model = true` (bài toán nghịch/tối ưu lặp có cấu trúc).
- Chọn thuật toán lặp nền: ISTA/FISTA, ADMM, Proximal Gradient, AMP, Unrolled Gradient Descent, (với tensor) **iterative tensor completion/ALS**.
- Mỗi vòng lặp → một **tầng**; tham số (bước, ngưỡng, toán tử) trở thành **học được**, có thể chia sẻ hoặc riêng từng tầng.
- Số tầng `K` do Search chọn theo ngân sách tính toán.
- Trong khung paper: unfolding là **function composition tuần tự** (sequential aggregation); các tham số học được có thể sinh bằng hypernetwork.
- Lợi ích: ít tham số, diễn giải được, hội tụ nhanh; kèm kiểm tra so với thuật toán lặp gốc làm baseline.

### Kết quả đo M8 (`python scripts/bench_unfold.py`)
Đã có: `dlnet/unfold.py` (`ista`, `fista`, `LISTA` dạng weight-coupled LISTA-CP theo Chen et al., arXiv:1808.10038 phương trình 10, đã đối chiếu PDF; FISTA viết từ trí nhớ), lệnh `dlnet unfold --data problem.npz --layers 4,8,16` (file `.npz` chứa `A` m×n, `Y` N×m, `X` N×n). Mạng chưa huấn luyện **bằng đúng** K vòng ISTA (có test), nên mọi cải thiện là do học.

Giao thức: chia 60/20/20; ISTA và FISTA chạy **đúng K vòng** với λ tune trên validation (19 giá trị); LISTA huấn luyện end-to-end (Adam, MSE, cosine, giữ trọng số tốt nhất trên validation), 3 seed; "thắng" khi mean + std trên test thấp hơn solver cổ điển tốt hơn.

Dữ liệu tổng hợp m=64, n=128, 10% phần tử khác 0, SNR 30 dB, 3000/1000/1000 mẫu, 60 epoch (NMSE dB trên test, thấp hơn là tốt hơn):

| K | Tham số | LISTA | ISTA cùng K | FISTA cùng K | Lợi | Thắng cả hai |
|---|---|---|---|---|---|---|
| 4 | 32.772 | −9,39 | −2,79 | −3,14 | +6,25 dB | có |
| 8 | 65.544 | −11,43 | −4,11 | −5,68 | +5,75 dB | có |
| 16 | 131.088 | −13,39 | −5,99 | −10,05 | +3,34 dB | có |

Độ lệch chuẩn qua seed < 0,01 dB vì khởi tạo là tất định (W = A/L), seed chỉ đổi thứ tự batch. Giới hạn: chỉ một bài toán tổng hợp, một mức nhiễu; lợi thế giảm khi K tăng (FISTA đuổi kịp); solver cổ điển chạy đến hội tụ (K lớn) chưa được so. Chưa có ADMM/AMP, support selection (LISTA-CPSS), lịch huấn luyện theo tầng của paper, và toán tử A dạng hàm (blur, MRI).

---

## 6. Cấu trúc repo đề xuất

```
DLNet/
 ├─ agent.md  README.md  pyproject.toml
 ├─ dlnet/
 │   ├─ data/        # ingest, profile
 │   ├─ estimate/    # capacity, intrinsic-dim, proxies, modularity
 │   ├─ tensor/      # unfolding, CP/Tucker/TT/TR, rank selection
 │   ├─ plugins/     # tabular/ hsi/ image/ timeseries/ (loader, split, metric, stem, preset, prior)
 │   ├─ library/     # registry module + preset baseline nổi tiếng (xem §13)
 │   ├─ evaluate/    # split, bộ baseline tham chiếu, metric, báo cáo (xem §12)
 │   ├─ modules/     # computation/ routing/ aggregation/ unfolding/
 │   ├─ spec/        # ArchSpec, ProblemSpec (pydantic/dataclass)
 │   ├─ recommend/   # rules + meta-model
 │   ├─ search/      # BO, successive halving, Pareto
 │   ├─ compile/     # ArchSpec → nn.Module
 │   ├─ package/     # exporter + templates
 │   └─ cli.py       # `dlnet design --data X --task Y --budget Z`
 ├─ benchmarks/      # dataset chuẩn + baseline tham chiếu
 ├─ examples/
 └─ tests/
```

Công nghệ: Python ≥3.10, PyTorch, `tensorly` (+`tensorly-torch`) cho phân rã, `optuna` cho search, `pydantic` cho spec, `pytest`.

---

## 7. Hợp đồng dữ liệu cốt lõi (rút gọn)

```yaml
# ArchSpec
backbone: {family: transformer|cnn|mlp|ssm|unfolded, depth: 6, width: 256}
computation:
  type: parameter|input|function|tensor_decomposed|unfolded
  method: lora|adapter|prefix|tucker|tt|ista|admm
  rank: 16
routing:      {type: fixed|learned, mode: soft|hard, top_k: 2}
aggregation:  {level: parameter|representation|input|function, mode: avg|attention|sequential}
training:     {setting: multitask|continual|peft, lr: 3e-4, epochs: 30}
constraints:  {max_params: 5e6, max_latency_ms: 20}
```

---

## 8. Lộ trình triển khai

Nguyên tắc: lõi **tổng quát trước**, kiểm chứng end-to-end sớm trên dữ liệu nhỏ rẻ (tabular), rồi thêm plugin; HSI là plugin ưu tiên cao.

| Mốc | Nội dung | Tiêu chí hoàn thành |
|---|---|---|
| **M0** | Khung repo, `ArchSpec/ProblemSpec`, registry module, compiler, giao diện plugin (§14) | Sinh model từ YAML, test pass |
| **M1** | **End-to-end tổng quát**: plugin tabular + bộ ứng viên MLP/ResNet-MLP/FT-Transformer + proxy → short-train → **xuất Top-K + leaderboard** | `dlnet design --data x.csv --topk 3` ra 3 gói baseline chạy được |
| **M2** | **Plugin HSI**: loader, split chống rò rỉ, preset 1D/2D/3D-CNN + HybridSN-style + CNN phân rã Tucker-2, metric OA/AA/Kappa | **Xong (v0.2.0)**: chạy thật trên Indian Pines, Pavia University; xếp hạng trên 3 lần chia khối (`--cv-splits`, §9 mục 7) |
| **M3** | Giao thức đánh giá đầy đủ (§12.2–12.3): nhiều seed, nhóm "chưa phân biệt được", quy tắc chấp nhận | **Xong (v0.2.0)**: báo cáo tái lập, val và test đều có ± std qua seed và lần chia |
| **M4** | Library mở rộng + tự ghép theo grammar (§13): SSRN, Transformer, Mamba/SSM, ResNet/ConvNeXt block | **Xong phần lõi (v0.3.0)**: registry có kiểu + grammar trong `hsi_lib.py`, test sinh ≥20 ứng viên hợp lệ. Chưa có: Mamba/SSM, ViT/Swin |
| **M5** | Cost model + tiết kiệm thời gian: zero-cost proxy (theo §16.1), successive halving, cache, resume; **hyperparameter estimate (LR range test, noise scale) đã có, §16.2** | **Xong phần đo được (v0.4.0)**, xem §16.3: successive halving giảm 50–70% thời gian (dữ liệu tổng hợp), cùng top-1 trên tabular, lệch trong mức nhiễu trên HSI; proxy chỉ ghi thông tin vì chưa hơn đếm tham số; cache cho giai đoạn nhiều seed. Chưa có: FLOPs/VRAM, resume cho successive halving |
| **M6** | Estimator: intrinsic dim → dung lượng, rank tensor | **Xong phần đo được (v0.5.0)**, xem §16.4: hai bộ ước lượng (TwoNN, MLE) kèm độ lệch chuẩn qua mẫu con và cờ `stable`; đo trên 5 bộ biết đáp án và 5 bộ thật. Chưa có: bằng chứng `capacity_hint` dự đoán đúng độ rộng tốt nhất (nên chỉ ghi thông tin) |
| **M7** | Tensor: Tucker/HOSVD tiền xử lý, lớp TT/Tucker/TR, Tensor-LoRA | **Xong phần module và phép đo (v0.6.0)**, xem "Kết quả đo M7" ở §4: lớp TT/TR nén 81–89% tham số, mất ≤0,4 điểm accuracy (trong mức nhiễu) trên digits. Tensor-LoRA có đủ 4 format nhưng **chưa nối vào `design`** (DLNet chưa có backbone pretrained) và trong phép đo nhỏ chưa hơn huấn luyện từ đầu. HOSVD chưa dùng làm tiền xử lý HSI |
| **M8** | Deep unfolding (khi có mô hình thuận) | **Xong cho bài toán thưa tuyến tính (v0.7.0)**, xem "Kết quả đo M8" ở §5: lệnh `dlnet unfold`, LISTA-CP hơn ISTA và FISTA (λ đã tune) 3,3–6,3 dB ở cùng số vòng K trên dữ liệu tổng hợp. Chưa có: ADMM/AMP, support selection, toán tử A không tường minh (ảnh), dữ liệu thật |
| **M13** | Port nhóm còn thiếu: đồ thị, id/recommender, autoencoder/VAE, GAN, Mamba-2; cost model; hợp nhất họ cũ | **Xong (v0.11.0–v0.12.0)**, xem §13.7. Chưa có: CVAE, loss cặp (BPR), GAN có điều kiện/FID, đồ thị lớn (dense), graph cạnh học được |
| **M14** | Khối ảnh từ thư viện chính thống (timm, smp, MONAI, torchvision); đối chiếu số học các khối viết từ trí nhớ | **Xong (v0.13.0)**, xem §13.8 |
| **M25** | Mốc 4 sau kiểm định: TabM, PLR, PatchTST, iTransformer (pretrained bị chặn bởi mạng) | **v0.23.0**; đối chiếu số học 7/7 khớp trên CI (§9.7) |
| **M26** | Module miền tần số (FFT/iFFT): GFNet, AFNO, FNO, FFC, FcaNet, FNet; lọc Fourier dọc trục phổ cho HSI; 7 họ HSI mới | **v0.24.0**; 11/11 khối trùng từng bit với mã gốc (CI), đo trên Indian Pines (§9.8) |
| **M27** | v1.0: 33 op và 17 họ mới (2023–2025) theo khảo sát `qa/reports/v0.24.0/module-survey.md` (attention/chuẩn hóa/tích chập mới; HSI Mamba/Transformer; RealMLP; TimeMixer/TimeXer/xLSTM/Gated DeltaNet; GraphGPS/Polynormer/SGFormer; FinalMLP/DCNv3/WuKong; KAT; MambaVision) | **v1.0.0**; so số học với mã gốc trên CI (§9.9) |
| **M24** | Mốc 3 sau kiểm định: công thức huấn luyện hiện đại, gói có `model.pt`/`predict.py`/export, thông báo lỗi | **Xong (v0.22.0)**, xem §9.6 |
| **M23** | Mốc 2 sau kiểm định: phân bổ ngân sách (SH chia đều theo rung, chặn tiến hóa, chạy lại finalist, `--hparams auto`, bỏ prior tay) | **Xong (v0.21.0)**, xem §9.5 |
| **M22** | Mốc 1 sau kiểm định: sửa kết quả sai (Hyperband, ứng viên lỗi, xếp hạng cùng epoch, nhiễu một lần chạy, vai trò, verdict, cảnh báo) | **Xong (v0.20.0)**, xem §9.4 |
| **M21** | GPU: `--device`, mixed precision (`--amp`), `channels_last`, gói xuất ra dùng GPU (§18 P1 mục 1) | **Xong phần mã (v0.19.0), chưa đo trên GPU thật**, xem §9.3 |
| **M20** | Ổn định (§18 P0 mục 1–3): ngân sách giây được tôn trọng, `--threads`, JSON chuẩn | **Xong (v0.18.0)**, xem §9.2 |
| **M19** | Regression test toàn bộ dlnet (coverage 90% → 97%, CI có sàn 95%), ngân sách theo epoch tái lập được, sửa lỗi tìm thấy; lộ trình nâng cấp | **Xong (v0.17.0)**, xem §9.1, §18 |
| **M17** | Học ánh xạ đặc trưng dữ liệu → hạt giống: surrogate zero-shot ZAP, meta-dataset 14 bộ ảnh có sẵn | **Xong (v0.16.0)**, xem §16.6 |
| **M18** | Tùy chọn AI: advisor gọi Claude Code / Codex / Gemini CLI; MCP server `dlnet mcp` | **Xong (v0.16.0)**, xem `doc/ai-advisor.md`. Chưa đo hiệu quả với AI thật |
| **M16** | Hướng tối ưu (`--objective`: accuracy, params, flops, balanced, pareto); bộ tài liệu `doc/` và quy tắc cập nhật theo phiên bản | **Xong (v0.15.0)**, xem §17, `doc/` |
| **M15** | Phân tích dữ liệu ảnh (data card), ngữ cảnh, chọn hạt giống, Hyperband, aging evolution, portfolio, SNIP/ZiCo/vote | **Xong (v0.14.0)**, xem §16.5. Đo: prior **chưa** dự đoán được thứ hạng; mặc định giữ mọi ứng viên |
| **M11** | Refactor lõi thành kho module có kiểu `dlnet/zoo/` dùng chung mọi plugin + ngữ pháp ghép + tiến hóa kiến trúc; gom module từ các repo GitHub | **Xong phần lõi (v0.10.0)**: §13.6; 54 op từ 6 repo, 14/14 khối đối chiếu torchvision khớp; kết quả đo zoo so với họ cũ ở §13.6. Chưa có: chuyển hẳn các họ cũ thành preset (vẫn song song), kiểu `graph`/`ids`, trọng số pretrained |
| **M10** | Thư viện khối từ 4 repo GitHub (§13.5), plugin `image_classification` và `image_segmentation` | **Xong (v0.9.0)**: `dlnet/blocks.py` + `THIRD_PARTY.md`; kết quả đo ở §14.4. Chưa có: pretrained, augmentation, dữ liệu segmentation thật |
| **M9** | Plugin ảnh, chuỗi thời gian; recommender học từ kho kết quả (meta-learning) | **Có plugin và recommender (v0.8.0), tiêu chí 80% CHƯA đạt**, xem §14.3: plugin `image` (cnn, resnet, vit) và `timeseries` (tcn, gru, ts_transformer) chạy end-to-end trên dữ liệu tổng hợp; recommender kNN đoán đúng họ tốt nhất trong top-3 ở 7/12 dataset bảng (58%, ngẫu nhiên 51%). Chưa có: Mamba/SSM, ConvNeXt cho ảnh, dữ liệu ảnh/chuỗi thật, kho kết quả đủ lớn |

---

## 9. Rủi ro & quyết định cần chốt

1. **Phạm vi**: framework tổng quát theo plugin. Thứ tự plugin đề xuất: tabular (kiểm chứng lõi nhanh) → HSI (ưu tiên nghiên cứu của bạn) → ảnh → chuỗi thời gian. **Cần chốt** thứ tự này và dataset kiểm chứng.
2. **Mức tự động**: tự chạy toàn bộ pipeline trong ngân sách; huấn luyện đầy đủ do người dùng thực hiện trên Top-K (tùy chọn `--finalize` để DLNet train đầy đủ top1).
   - **Thứ hạng chỉ là ước lượng**: xếp hạng dựa trên huấn luyện ngân sách hạn chế; sau huấn luyện đầy đủ các hạng sát nhau có thể đổi thứ tự. Vì vậy trả Top-K chứ không chỉ một mô hình.
3. **Độ tin cậy của ước lượng**: proxy zero-cost không luôn tương quan với hiệu năng → luôn xác thực bằng short-train.
4. **Paper là survey**: không có thuật toán chọn kiến trúc sẵn; rule prior ở §2 là giả thuyết cần kiểm chứng bằng benchmark nội bộ.
5. **Unfolding** chỉ hợp lý khi có mô hình thuận; nếu không, bỏ qua (không ép dùng).
6. **Tái lập & giấy phép** dataset/pretrained.
7. **Xếp hạng nhiễu khi chia theo khối** (phát hiện từ chạy thật trên Indian Pines/Pavia University): tập validation chỉ gồm vài khối ảnh nên điểm validation lệch xa điểm test (vd. Pavia: top1 theo validation 0.963 nhưng test 0.859, thấp hơn hạng 3 ở test 0.886). Std theo seed không phản ánh nhiễu này. **Đã cài đặt (v0.2.0)**: finalist được huấn luyện trên nhiều lần chia khác nhau (`--cv-splits`, mặc định 3 khi chia khối; lần chạy i dùng seed i trên lần chia i), baseline tham chiếu lấy trung bình trên cùng các lần chia. Đây là chia lặp lại (repeated split), chưa phải k-fold phủ kín ảnh; successive halving vẫn chạy trên một lần chia.
8. **Chia khối làm mất nhiều dữ liệu và lớp**: bỏ 41–51% pixel có nhãn ở margin, một số lớp vắng trong train hoặc test. Cần chiến lược chọn khối cân bằng lớp hơn hoặc k-fold theo khối.

9. **Kết quả audit (regression test khóa từng lỗi, `tests/test_regression.py`)** — đã sửa: (a) hết ngân sách trước rung đầu → `IndexError`; (b) hết giờ giữa rung → ứng viên bị bỏ lại giữ điểm cũ và bị xếp hạng cùng ứng viên train nhiều epoch hơn; (c) `--out` bị `rmtree` toàn bộ (xóa cả file lạ); (d) impute thiếu dữ liệu bằng median của toàn bộ dòng (rò rỉ test sang train), giờ dùng median của train; (e) dòng thiếu nhãn không bị loại; (f) split rỗng (dữ liệu quá nhỏ, khối lớn hơn ảnh) làm `evaluate` sập với thông báo khó hiểu, giờ báo lỗi rõ; (g) không sinh được ứng viên nào → `IndexError`, giờ báo lỗi rõ; (h) lớp ≥3 mẫu có thể vắng trong val/test, giờ luôn có mặt trong mọi split. Chưa sửa: ngân sách thời gian vẫn chỉ kiểm tra giữa các ứng viên; các hạn chế ở mục 7–8.

---

### 9.1 Audit v0.17 (bao phủ toàn bộ dlnet) và các phát hiện
Đo bằng `coverage` (cả lệnh CLI trong tiến trình con): **90% → 97%**, 590+ test; CI có job `coverage` với sàn 95%.
- **Kết quả phụ thuộc tốc độ máy (đã sửa)**: mọi ngân sách là giây, nên 2 test e2e khẳng định chất lượng (chuỗi thời gian, phân đoạn) đỏ khi máy chậm (dưới coverage). Thêm ngân sách theo epoch `--budget 400ep` (`dlnet/clock.py`): cùng lệnh, cùng leaderboard trên mọi máy; test chất lượng chuyển sang epoch.
- **Crash (đã sửa)**: advisor lỗi ở lần gọi đầu nhưng trả lời ở vòng tiến hóa làm `design` crash (`None.setdefault`).
- **Sai lệch tài liệu/mã (đã sửa)**: trợ giúp CLI nói `autoencoder` nhưng alias không có.
- **Cảnh báo số học (đã sửa)**: dữ liệu hằng / trùng điểm cho 0/0 trong TwoNN và `pca_dims_for_95pct_energy` > số đặc trưng.
- **Còn mở** (đưa vào §18): ngân sách giây bị vượt (30 s chạy 35–138 s: baseline, data card, proxy không bị giới hạn; HistGradientBoosting 0,07 s lúc rảnh nhưng 11,6 s khi máy bận vì OpenMP tranh luồng); `NaN` được ghi vào `profile.json` (JSON không chuẩn); không dùng GPU; `gan_data.py` là bản sao của `gan_run.py` trong gói GAN (có test so khớp, vẫn dễ lệch).

### 9.2 Ổn định v0.18 (M20): đo trước, sửa, đo lại
Máy 4 CPU; dữ liệu ảnh 600 mẫu 16×16 và bảng 2000×20 (`--budget 30s`).
- **Treo do tranh luồng OpenMP (đã sửa)**: 4 lệnh `design --budget 30s` chạy cùng lúc trên v0.17 vẫn kẹt trong `HistGradientBoosting` sau **10 phút** (py-spy: `run_reference` → `TreePredictor.predict` / `TreeGrower`). Đo riêng: 4 tiến trình × 4 luồng không xong trong 300 s; 4 × 2 luồng 1,0–1,3 s; 4 × 1 luồng **0,2 s**. `OMP_WAIT_POLICY=PASSIVE` cũng gỡ treo (0,8–1,6 s) nhưng làm chậm khi máy rảnh (HistGradientBoosting 0,69 → 5,7 s, torch 0,57 → 1,0 s) nên không dùng. Sửa: `--threads` (`dlnet/threads.py`) giới hạn torch, OpenBLAS, cả hai OpenMP (threadpoolctl) và `n_jobs` của random forest cho cả lần chạy; mặc định = số CPU được cấp trừ số CPU tiến trình khác đang dùng (`/proc/loadavg`), tối thiểu 1. Sau sửa: 4 lệnh cùng lúc mất **30,2–31,3 s**.
- **Lần huấn luyện bắt đầu sát hạn (đã sửa)**: trước đây một lần chạy được bắt đầu nếu đồng hồ còn trước hạn, nên có thể kết thúc rất xa sau hạn (rung 30 epoch, giai đoạn nhiều seed, ước lượng hyperparameter 11,8 s khi phần còn lại ~6 s). Nay `search.Pace` dự báo chi phí (giây/epoch đo trên chính trainer; ứng viên chưa train thì theo FLOPs) và chỉ bắt đầu khi dự báo kết thúc trước hạn; lần chạy đầu tiên vẫn luôn chạy. Với `--budget 60ep` số epoch đã train ≤ 60 (trước: ≤ 96).
- **Báo cáo ngân sách**: `leaderboard.json` khối `budget` và mục "Budget" trong `leaderboard.md`: giây và epoch theo giai đoạn (load, reference, generate, proxies, seed_selection, search, evolution, hparams, export), số luồng, % vượt; cảnh báo khi vượt > 10%. Đo: ở 30 s, các bước trước tìm kiếm chiếm ~13 s (sinh + tính chi phí ứng viên ~5 s); đây là chỗ cần tối ưu tiếp.
- **JSON chuẩn (đã sửa)**: mẫu trùng nhau cho intrinsic dimension `NaN`, ghi ra `profile.json` thành token `NaN` (không phải JSON). Mọi file DLNet ghi đi qua `dlnet/jsonsafe.py` (`NaN`/`inf` → `null`, `allow_nan=False`).
- Test: `tests/test_stability.py` (3 test đỏ trên mã v0.17, xanh sau sửa).

### 9.3 GPU v0.19 (M21): kiểm được gì, chưa kiểm được gì
Môi trường phát triển và CI **không có GPU**, nên chưa có số đo tốc độ hay độ chính xác trên GPU thật. Thay vào đó:
- **Lỗi lệch thiết bị**: thiết bị `meta` từ chối tensor CPU trộn với tensor của nó giống CUDA (chỉ cho phép số vô hướng 0 chiều trên CPU). Mọi ứng viên của 8 plugin (441 kiến trúc: 63 ảnh, 73 phân đoạn, 46 tái tạo, 49 chuỗi, 55 đồ thị, 22 recommender, 70 HSI, 63 bảng) chạy `forward` ở cả chế độ train và eval trên `meta` mà không lỗi; `Trainer.fit` chạy trọn trên `meta` (dữ liệu được chép sang, optimizer chạy). Test: `tests/test_device.py` (lấy mẫu ngữ pháp zoo cho 6 kiểu tensor).
- **Mixed precision**: autocast bf16 và fp16 (kèm `GradScaler`) chạy trên CPU; trên dữ liệu thử cả ba độ chính xác đạt cùng validation (0,68). Trên CPU chậm hơn float32 (80 epoch: float32 45 s, bf16 64 s, fp16 315 s), nên `auto` giữ float32 trên CPU.
- **Không đổi trên CPU**: cùng lệnh với `--threads 2` cho leaderboard giống từng bit với v0.18.0 (ảnh và bảng).
- **Chưa làm**: lệnh `gan`, `unfold` vẫn chỉ chạy CPU; proxy, data card, surrogate chạy CPU (nhỏ); `torch.compile`; nhiều GPU. Ước lượng VRAM (`--max-vram`) vẫn chưa đối chiếu với GPU thật.
- Cần làm khi có GPU: đo tốc độ so với CPU trên vài bộ dữ liệu, so độ chính xác bf16 với float32 cùng ngân sách epoch, đối chiếu VRAM ước lượng với `torch.cuda.max_memory_allocated`.

### 9.4 Mốc 1 sau kiểm định (M22, v0.20): sửa kết quả sai
Theo `qa/reports/v0.19.0/SUMMARY.md` mục S1–S6, mỗi mục có test trong `tests/test_audit_fixes.py`:
- **S1** Hyperband sập khi ngân sách epoch hết đúng hạn: ngoặc mới chỉ mở khi `now < deadline`; ngoặc không huấn luyện được ai bị bỏ qua. Lệnh tái hiện (`mnist14 --budget 150ep --seeds 16 --scheduler hyperband --evolve 0`) nay chạy xong.
- **S2** Cô lập ứng viên lỗi: ngoại lệ, loss NaN/inf ở mọi batch của một epoch (`core.Trainer.fit` bỏ bước có loss không hữu hạn và báo `FloatingPointError` khi cả epoch như vậy), điểm validation không hữu hạn → ứng viên bị loại, ghi `failed_candidates`; chỉ dừng khi mọi ứng viên lỗi.
- **S3** Xếp hạng trộn số epoch: `search.equalize` huấn luyện tiếp mô hình ít epoch tới mức cao nhất nếu ngân sách cho phép; `search.final_order` xếp nhóm nhiều epoch trước; leaderboard có cột `epochs`, `runs`; mặt Pareto chỉ trên nhóm epoch cao nhất. Đo trên fashion14 150ep: 3 mô hình được nâng từ 4 lên 12 epoch, Top-3 cùng 12 epoch.
- **S4** Nhiễu một lần chạy: `search.sampling_noise` (accuracy: √(p(1−p)/n); RMSE: RMSE/√(2n); MSE: MSE·√(2/n)) làm sàn cho ± và cho nhóm; quy tắc nhận hyperparameter dùng cùng sàn; cảnh báo khi `runs` = 1.
- **S5** Vai trò: `best-small` chỉ khi nhỏ hơn top-1; `most-stable` chỉ khi mọi mô hình có ≥ 2 lần chạy.
- **S6/R1/R2** Cột số lẫn chuỗi được đọc thành số (giá trị lạ thành thiếu) kèm cảnh báo; cột chữ > 50 giá trị cảnh báo; `--target` sai báo rõ; HSI `--split pixel` cảnh báo rò rỉ không gian; verdict có ngưỡng nhiễu trên test và nhiễu cho hồi quy.
- Mục leaderboard **Warnings** gom các cảnh báo trên, cộng "top-1 mới được huấn luyện < 12 epoch".

### 9.5 Mốc 2 sau kiểm định (M23, v0.21): dùng ngân sách đúng chỗ
Thay đổi: mỗi rung của successive halving một phần ngân sách bằng nhau (Jamieson & Talwalkar, Algorithm 1), finalist bù từ rung trước khi rung cuối thiếu người; tiến hóa chỉ chạy khi con tới được 12 epoch; `more_runs` chạy lại finalist tốt nhất khi còn ngân sách; `--hparams auto` (chỉ finalist có ≥ 2 lần chạy), chi phí ước lượng tính vào ngân sách epoch, LR đề xuất = cực tiểu/10; prior viết tay trọng số 0 khi có surrogate.

**Đo A/B** v0.20.0 so với v0.21.0, cấu hình mặc định, `--budget 150ep --threads 1`, mnist14 / fashion14 / digits × seed 0, 1:
| | v0.20.0 | v0.21.0 |
|---|---|---|
| epoch của Top-3 | 12, 12, 12 | **30, 30, 30** (cả 6 lần) |
| test của top-1, chênh new − old | | +0,005, −0,022, +0,039, +0,010, −0,003, +0,003 (TB **+0,005**, sd 0,02; 4/6 dương) |

Kết luận trung thực: Top-K được huấn luyện sâu gấp 2,5 lần nên thứ hạng gần kết quả huấn luyện đầy đủ hơn; độ chính xác test **không khác biệt có ý nghĩa** ở ngân sách này. Phiên bản đầu (chưa bù finalist) chỉ xuất 1 mô hình: đã sửa trước khi phát hành. Với ngân sách 150 epoch, `more_runs` thường không còn chỗ (top-1 vẫn 1 lần chạy).

### 9.6 Mốc 3 sau kiểm định (M24, v0.22): mạng đầu ra tốt hơn, gói dùng được ngay
- **Công thức huấn luyện** (`--recipe modern`, mặc định): augmentation cho ảnh (dịch tối đa 1/8 cạnh; lật ngang cho ảnh màu và phân đoạn, mask lật cùng ảnh; một phép lấy mẫu affine theo số nguyên pixel trên thiết bị) và label smoothing 0,1; `train_full` warm-up 5 % rồi cosine. Chưa làm: EMA, mixup/cutmix, augmentation cho chuỗi thời gian.
- **Đo A/B** `--recipe plain` so với `modern`, mặc định, `--budget 150ep --threads 1`, test của top-1:

  | | seed 0 | seed 1 |
  |---|---|---|
  | CIFAR-10 3k (màu 32×32) | 0,479 → **0,584** (+0,105) | 0,487 → **0,587** (+0,100) |
  | fashion14 (xám 14×14) | 0,863 → 0,836 (−0,027) | 0,821 → 0,828 (+0,007) |
  | digits (8×8) | 0,997 → 1,000 | 1,000 → 0,995 |

  Kết luận: +10 điểm, ổn định, trên ảnh màu tự nhiên; trên ảnh xám rất nhỏ thì trung tính (trong nhiễu). Mặc định `modern` vì không thấy thiệt hại có ý nghĩa.
- **Gói xuất ra**: `train.py` chạy từ mọi thư mục, lưu `model.pt` + `preprocess.json` (mỗi hàm dựng dữ liệu ghi khối `preprocess`); `predict.py` + `inference.py` dự đoán file mới theo đơn vị nhãn gốc cho cả 8 loại dữ liệu (`tests/test_package_inference.py`); `--export model.pt2` (torch.export, không cần gói thêm) hoặc `.onnx` (cần onnx, onnxscript).
- **Thông báo lỗi**: router không đưa `.npz` dạng mẫu/nhãn sang HSI và liệt kê các mảng; file không tồn tại; `--max-*` sai định dạng; giới hạn loại hết ứng viên (nêu ứng viên nhỏ nhất); `--target` sai gợi ý cột gần đúng; leaderboard ghi số ứng viên thật sự được huấn luyện (`search_coverage`) và cảnh báo khi Top-K thiếu.
- Tài liệu: thứ tự cài torch CPU trước, Python 3.10–3.13, yêu cầu của từng loại gói (F1, F3).

### 9.7 Mốc 4 sau kiểm định (M25, v0.23): họ mạnh mới; phần bị chặn
- **Thêm** (`dlnet/zoo/ops_modern.py`, viết theo mã chính thức, giữ thứ tự tham số để đối chiếu): TabM (BatchEnsemble k thành viên, head `ensemble`: một lớp đầu ra và một loss cho mỗi thành viên, dự đoán = trung bình xác suất), PLR(lite), PatchTST (RevIN, patch, encoder độc lập kênh, BatchNorm, residual attention), iTransformer (mỗi biến là một token). Họ mới: bảng `tabm`, `tabm_plr`; chuỗi `patchtst`, `itransformer`. `core.members` / `proxies` / `hparams` xử lý đầu ra nhiều thành viên.
- **Đo** (cùng split, 60 epoch bảng / 30 epoch chuỗi, lr 1e-3, 3 seed, `scripts`-ngoài-repo `fam_ab.py`): test trung bình
  - breast_cancer: MLP 0,977, ResMLP 0,980, TabM 0,980, TabM+PLR 0,977 (trong nhiễu);
  - wine: MLP 0,957, TabM 0,974, **TabM+PLR 1,000** (3/3 seed);
  - diabetes (−RMSE): MLP −56,1, ResMLP −59,0, TabM −56,0, TabM+PLR −55,9 (trong nhiễu);
  - ts_freq (tổng hợp): TCN 1,00, ts_transformer cũ **0,37**, PatchTST 1,00, iTransformer 1,00.
  Kết luận: TabM không kém MLP, PLR giúp trên wine; với chuỗi, hai họ mới sửa được điểm yếu của Transformer cũ. Chỉ một bộ chuỗi (tổng hợp) nên chưa khái quát.
- **Đối chiếu số học với mã gốc** (`scripts/verify_modern.py`, chạy trên GitHub Actions vì môi trường phát triển không cho chạy mã tải từ ngoài; workflow `verify-modern.yml`): **7/7 khớp** sau bản phát hành v0.23.0 — TabM, PLR, PatchTST trùng từng bit, iTransformer sai khác ≤ 4,8e-7 ([lượt chạy](https://github.com/Duy-Huan/DLNet/actions/runs/43025944848)). Bảng: THIRD_PARTY.md.
- **Bị chặn bởi mạng (không làm)**: mô hình pretrained (timm/torchvision/DINOv2 weights, TabPFN) — download.pytorch.org, huggingface.co, dl.fbaipublicfiles.com trả 403 qua proxy. **Chưa làm**: tìm kiếm dựa mô hình (DEHB, PriorBand, LC-PFN); graph transformer, DCNv2, Mamba cho HSI.

### 9.8 Module miền tần số (M26, v0.24): FFT → phép học trên phổ → iFFT
- **Lý do**: nhiều bài HSI gần đây chuyển phổ/không gian sang miền tần số (FFT), xử lý ở đó rồi biến đổi ngược. Thêm các khối chuẩn của hướng này, viết theo mã chính thức, vào `dlnet/zoo/ops_freq.py`:
  - `global_filter`, `gfnet_block` (GFNet: lọc phức học được trên mọi tần số, tương đương tích chập vòng toàn cục; 1/2/3-D và `seq`);
  - `afno`, `afno_block` (AFNO: MLP phức chéo khối, soft-shrink);
  - `spectral_conv`, `fno_block` (FNO: giữ `modes` tần số thấp, trộn kênh bằng ma trận phức);
  - `ffc_block` (FFC: nhánh cục bộ 3×3 và nhánh toàn cục Fourier trao đổi thông tin);
  - `fca` (FcaNet: SE với pooling bằng các hệ số DCT);
  - `fnet_block` (FNet: phần thực FFT 2 chiều thay self-attention, cho `seq`);
  - HSI (DLNet, không phải port): `spectral_gf` (lọc GFNet dọc **trục phổ**, có residual), `spectral_rfft` (hệ số Fourier của phổ từng pixel làm kênh).
- **Họ HSI mới**: `gfnet2d`, `fno2d`, `afno2d`, `ffc2d`, `fcanet2d`, `spec_fft2d`, `fourier_ss`; khối không gian của mô hình ghép thêm `gfnet`, `fno`, `afno`, `ffc`, `fca`. Ngữ pháp tự dùng các op mới qua thẻ `block`/`attn`/`stem`.
- **Kiểm chứng**: `scripts/verify_freq.py` (job CI `verify-freq`, mã gốc ghim commit): GFNet, AFNO, FFC, FcaNet **11/11 trùng từng bit**. FNO/FNet không có mã PyTorch chính thức chạy được: `tests/test_freq.py` kiểm bằng đáp án đóng (lọc toàn cục = tích chập vòng tính trực tiếp; FNO giữ đúng các góc tần số; FNet = phần thực FFT 2-D của numpy; cơ sở DCT trực chuẩn), cùng autocast bf16 và `torch.export`.
- **Đo** (`scripts/bench_freq_hsi.py`): Indian Pines, cùng split mỗi seed, 3 seed, 40 epoch, rộng 64, sâu 2, AdamW 1e-3, công thức `train_full`. OA test trung bình ± std (AA):
  | Họ | pixel, PCA 30 | pixel, đủ 200 băng (`--pca 0`) | khối không gian, PCA 30 |
  |---|---|---|---|
  | `spec_fft2d` (mới) | 0,9871 ± 0,0044 (0,974) | **0,9871 ± 0,0031 (0,984)** | 0,484 ± 0,039 |
  | `ffc2d` (mới) | **0,9871 ± 0,0026** (0,951) | 0,9818 (0,868) | 0,497 ± 0,057 |
  | `fcanet2d` (mới) | 0,9870 (0,932) | 0,9810 (0,843) | 0,461 ± 0,087 |
  | `resnet2d` (cũ) | 0,9864 ± 0,0029 (**0,982**) | 0,9855 ± 0,0016 (0,969) | **0,517 ± 0,043** |
  | `cnn2d` (cũ) | 0,9819 | 0,9784 | 0,420 ± 0,076 |
  | `gfnet2d` (mới) | 0,9783 | 0,9731 | 0,513 ± 0,016 |
  | `fourier_ss` (mới) | 0,9799 | 0,9717 | 0,437 ± 0,007 |
  | `afno2d` (mới) | 0,9773 | 0,9650 | 0,454 ± 0,059 |
  | `fno2d` (mới) | 0,9720 | 0,9360 | 0,496 ± 0,047 |
  | `convnext2d` (cũ) | 0,9692 | 0,9374 | 0,486 ± 0,056 |
  | `hybridsn` (cũ) | 0,8256 | 0,8575 | (không chạy) |

  Kết luận trung thực: **không họ tần số nào thắng ResNet quá nhiễu về OA**. Điểm đáng giữ: `spec_fft2d` (hệ số Fourier của phổ từng pixel) khi giữ đủ băng có OA ngang ResNet và AA trung bình cao hơn (0,984 so với 0,969) nhưng chỉ 2/3 seed (seed 2: 0,975 so với 0,978): gợi ý tốt hơn trên các lớp nhỏ, chưa chắc chắn; `ffc2d`, `fcanet2d` ngang ResNet ở PCA 30; `gfnet2d` ổn định nhất khi chia theo khối (std 0,016). FNO và AFNO kém hơn CNN trên patch 9×9 (quá ít tần số để mô hình hóa). Chia theo khối: mọi họ 0,42–0,52 với nhiễu ±0,02–0,09, không phân biệt được. Các họ vẫn ở trong không gian tìm kiếm để Hyperband/tiến hóa chọn theo dữ liệu, không ưu tiên sẵn.
- **Lưu ý**: FFT trên trục phổ chỉ có nghĩa khi giữ thứ tự băng (`--pca 0`); FLOPs của chính FFT không được `cost.py` đếm.

### 9.9 v1.0 (M27): làm đầy kho khối theo khảo sát 2023–2025
- **Nguồn**: khảo sát `qa/reports/v0.24.0/module-survey.md`. Mọi khối viết lại bằng torch thuần theo mã chính thức, ghim commit; bảng nguồn và các chỗ khác gốc nằm trong THIRD_PARTY.md, mục v1.0. Kho từ 100 lên 133 op (33 op mới) và thêm 17 họ; tổng 80 họ trên 8 miền dữ liệu.
  - **Drop-in** (`ops_next.py`): `gated_attn_block`, `diff_attn_block`, `dyt`, `rope_vit_block`, `wtconv`, `wtconvnext_block`, `uib`, `mobile_mqa`, `lsnet_block`, `star_block`, `efficientvit_block`, `msca`, `msca_block`. Ngữ pháp tự dùng qua thẻ.
  - **HSI** (`ops_hsi.py`): họ `mambahsi`, `mambahsi_plus`, `ssftt`, `fahm`.
  - **Bảng / chuỗi** (`ops_domain.py`, `ops_seq2.py`): họ `realmlp`, `timemixer`, `timexer`, `xlstm` (mLSTM), `gated_deltanet`.
  - **Đồ thị / CTR** (`ops_graph2.py`): họ `gps_net` (kèm RWSE), `polynormer_net`, `sgformer_net`, `finalmlp`, `dcnv3`, `wukong`.
  - **Ảnh**: họ `mambavision`, `kat` (GR-KAN).
- **Kiểm chứng**: 5 job CI so đầu ra với mã gốc sau khi chép trọng số (THIRD_PARTY.md). `tests/test_v1.py` kiểm bằng đáp án đóng:
  - wavelet Haar khôi phục hoàn hảo;
  - SKA bằng tổng trực tiếp;
  - delta rule một bước;
  - các bộ trộn nhân quả không nhìn trước;
  - RWSE bằng lũy thừa ma trận numpy;
  - GAT cho nút cô lập ra 0 như PyG;
  - mỗi họ mới huấn luyện được, chạy dưới autocast bf16 và `torch.export` được (trừ các họ có vòng lặp Python).
- **Đo** (`scripts/bench_v1.py`, `scripts/bench_freq_hsi.py`; cùng split, cùng epoch, 3 seed nếu không ghi khác):
  - **Bảng** (sklearn, 60 epoch): wine: tabm 0,9915 ± 0,0121, **realmlp 0,9829 ± 0,0242**, mlp 0,9658, resmlp 0,9402; breast_cancer: **realmlp 0,9855 ± 0,0109**, resmlp 0,9797, mlp 0,9768, tabm 0,9768; diabetes (−RMSE): tabm −55,21, mlp −56,02, **realmlp −56,07**, resmlp −60,67. RealMLP nhất 1/3 bộ, không bộ nào hơn rõ quá độ lệch seed.
  - **Ảnh** (digits 8×8, 30 epoch): cnn 0,9909, resnet 0,9900, **mambavision 0,9420**, **kat 0,8922**, vit 0,8850. Hai họ mới thua CNN trên ảnh nhỏ, ít dữ liệu.
  - **Chuỗi thời gian** (tổng hợp, 30 epoch): `ts_freq`: itransformer 1,0000, **xlstm 0,9973**, **gated_deltanet 0,9946**, **timemixer 0,9919**, tcn 0,9785, **timexer 0,7984**, patchtst 0,7608, ts_transformer 0,3898; `ts_crosschannel`: **xlstm 0,9945**, tcn 0,9918, **gated_deltanet 0,9727**, ts_transformer 0,9426, **timemixer 0,7923 ± 0,2027** (không ổn định giữa các seed), itransformer 0,3416, **timexer 0,3306**, patchtst 0,2978 (ba họ cuối ở mức đoán ngẫu nhiên: bài cần so trễ giữa các kênh). xLSTM và Gated DeltaNet ngang TCN; TimeXer kém trên cả hai bài phân loại (bản gốc thiết kế cho dự báo có biến ngoại sinh).
  - **HSI** (Indian Pines, chia pixel 10/10/80, PCA 30, patch 9, **20 epoch, 2 seed**; chia pixel thổi phồng OA): resnet2d OA 0,9841 ± 0,0005 (AA 0,9485; 166k tham số), **fahm 0,9733 ± 0,0007** (AA 0,8257; 213k), **mambahsi_plus 0,9732 ± 0,0025** (AA 0,9202; 278k), cnn2d 0,9720 ± 0,0027 (AA 0,9038; 64k), **ssftt 0,9620 ± 0,0056** (AA 0,8958; 153k), **mambahsi 0,9419 ± 0,0045** (AA 0,7796; 76k). Không họ mới nào hơn resnet2d. Thời gian train trên CPU (máy chạy song song việc khác nên chỉ so bậc độ lớn): resnet2d 15–71 s, ssftt 9–36 s, mambahsi 5–6 phút, fahm 8–25 phút, mambahsi_plus 23–60 phút.
  - **Kết luận trung thực**: v1.0 làm kho đa dạng hơn, chưa cho thấy họ mới nào thắng họ cũ ở ngân sách nhỏ trên CPU; các họ Mamba/xLSTM/GDN chạy vòng lặp Python nên chậm hơn CNN hàng chục tới hàng trăm lần (đã sửa backward O(L²), §9.10).
- **Chưa làm và lý do**:
  - OverLoCK cần kernel NATTEN, và kiến trúc ba mạng con truyền ngữ cảnh không lắp vào ngữ pháp tuần tự.
  - xLSTM-Mixer cần sLSTM.
  - TabR/ModernNCA cần truy hồi kNN trong vòng huấn luyện.
  - Pretrained bị proxy chặn.
- **Hạn chế đã biết**:
  - Các họ Mamba/mLSTM/Gated DeltaNet chạy bằng quét tuần tự Python, chậm trên CPU với chuỗi dài. Ví dụ `mambahsi_plus` trên Indian Pines chậm hơn ResNet nhiều lần.
  - FLOPs của các vòng quét không được `cost.py` đếm đầy đủ.

### 9.10 Lỗi tìm ra khi chạy QA `examples/` trước v1.0 và cách sửa
- **Backward O(L²) của các vòng quét** (`selective_scan`, `recurrent_gated_delta_rule`): chỉ số `dA[:, :, i]` làm autograd cấp một tensor 0 bằng cả khối cho mỗi bước. Đổi sang `unbind`: bước train batch 64, 1 luồng: FAHM 13,8 → 2,1 s; MambaHSI 5,3 → 1,3 s; MambaHSI+ 23,1 → 5,8 s; số học không đổi.
- **Ngân sách giây bị vượt**: `--budget 90s` trên MNIST chạy hơn 12 phút vì ứng viên đầu luôn được train và ứng viên chưa train được định giá theo FLOPs (không tính vòng lặp Python). Nay `Trainer.probe_epoch_seconds` đo một bước thật; cùng lệnh xong trong 85 s.
- **`model.py` không chứa mạng**: gói xuất ra chỉ có `from zoo import build_model` + bản sao `zoo/`. Nay `model.py` là nguồn PyTorch thuần của chính mạng (`dlnet/standalone.py`), được chạy thử và so với mạng đã tìm kiếm trước khi ghi; 238/238 cặp (op, kiểu) và 79 họ đều giống hệt.
- **Quy trình**: file dữ liệu > 50 MB và mã chưa format làm đỏ CI nhiều lượt; thêm `scripts/ci_local.py` + hook pre-push (chạy cổng CI trên máy trước khi push), và push chỉ đổi dữ liệu/kết quả ví dụ không kích hoạt CI.
- **`model.pt2` chứa cả tập dữ liệu** (sau v1.0.0): đầu vào mẫu của `torch.export` là lát cắt chung vùng nhớ với toàn bộ dữ liệu; CIFAR-10: 74 MB cho mô hình 73 nghìn tham số. Sao riêng đầu vào mẫu: 0,4 MB.
- **Dựng sẵn mọi ứng viên** (sau v1.0.1): ISBI 256×256: một khối attention đòi 60 GB làm hỏng cả lượt `design`; Oxford Pets trimap: 100 mô hình cùng nằm trong bộ nhớ làm máy khởi động lại. Nay dựng lần lượt, loại ứng viên không dựng được (`search._open`). Còn lại: một ứng viên đòi vài GB (dưới RAM máy) vẫn được thử; ước lượng bộ nhớ của cost model không tính ma trận attention bên trong khối.
- **37 lớp bị hiểu là hồi quy** (sau v1.0.1): nhãn nguyên hơn 20 giá trị → hồi quy, không cảnh báo. Nay file ảnh/chuỗi nhận mã lớp liên tiếp là phân loại; trường hợp mơ hồ có cảnh báo.
- **v1.1 (yêu cầu người dùng sau QA)**: mỗi thư mục đầu ra có `baselines.ipynb` (Jupyter / Colab: setup, dữ liệu, sơ đồ, huấn luyện từng baseline, so sánh, nạp lại; `dlnet/notebook.py`, test e2e chạy notebook như script) và mỗi gói có `architecture.svg` (sơ đồ khối kiểu paper từ spec, `dlnet/diagram.py`; vẽ được cho 238/238 cặp op/kiểu và mọi họ). Đồ thị từng lớp trong notebook dùng torchview (cần chương trình graphviz, Colab có sẵn).
- **v1.2: mạng từ paper làm baseline** (`dlnet/paper.py`, `papers/`, doc/user-guide.md): thẻ YAML (`paper`, `official`, `architecture` dạng đồ thị `{id, from, module, args}`, `blocks` có tham số, `training`), lệnh `dlnet paper check | build | verify`, mức trung thành A/B/C đo được. Kiểm chứng thật (job `verify-papers`, run 43257787470): **StarNet-S1** (CVPR 2024, thẻ viết tay từ sơ đồ bằng lớp nguyên thủy `conv`/`norm`/`act`/`mul`/`add`) và **SSFTT** (TGRS 2022) trùng từng bit với mô hình của tác giả, số tham số bằng nhau (2.869.168 và 153.224) → mức B. ResNet-20 viết từ paper (mức C, 272.378 tham số so với 0,27M). FAHM ở mức C trong hệ thẻ vì mã gốc cần kernel CUDA (bằng chứng riêng ở `verify-hsi`). Gói mẫu: `examples/papers/`. Chưa làm ở v1.2: sinh nháp thẻ từ mã gốc, huấn luyện hai giao thức và bảng ba cột, `design --papers` (v1.3).
- **v1.3: sinh nháp thẻ và so sánh cùng giao thức**: `dlnet paper import` (torch.fx trên mã gốc → thẻ; test với một mạng có residual, concat, pooling: nháp đạt mức B khi `verify`); `dlnet design --papers` huấn luyện thẻ theo giao thức của top-1 và, với `--paper-recipe`, theo công thức tác giả; bảng "Paper baselines". Đo thật (Indian Pines, chia pixel 10/10/80, patch 13, PCA 30, ngân sách 6 phút, 3 lần chạy × 30 epoch): SSFTT (mức B, 153.224 tham số) val 0,9730 ± 0,0051, test OA 0,9772; công thức tác giả (Adam, lr 1e-3, 100 epoch) test 0,9810; top-1 của DLNet `tucker_cnn2d` (40.080 tham số) val 0,9847 ± 0,0028, test 0,9828. Một lần chia, chia pixel: chênh lệch nhỏ, chưa đủ để kết luận hơn kém. Chưa làm: thẻ làm hạt giống cho tìm kiếm (đột biến trên `graph`), gấp các khối lặp khi import.
- **v1.4: làm đầy kho** (`dlnet/zoo/ops_more.py`, `scripts/verify_more.py`, job `verify-more`): 21 module và 13 họ mới sau khi rà kho với timm / PyG / Time-Series-Library / FuxiCTR (bảng nguồn ở `THIRD_PARTY.md`). Đo (doc/algorithms.md): không họ mới nào thắng họ cũ trên các bài nhỏ; `dlinear` ở mức ngẫu nhiên trên phân loại chuỗi (đúng bản chất tuyến tính), giữ làm mốc. TimesNet không port (chọn chu kỳ theo cả batch, không xuất được). Bài học: module mới dùng tên import giống các tệp op khác, nếu không `model.py` độc lập báo xung đột tên.
- **v1.5: MCP server qua HTTP** (`dlnet serve`, doc/mcp-server.md): thay bản stdio cũ. Bài học: (1) đặc tả MCP đổi nhanh, đọc bản gốc trên GitHub (`modelcontextprotocol/modelcontextprotocol`, `docs/specification/<phiên bản>`) chứ không dựa trí nhớ: bản 2026-07-28 bỏ phiên và bước bắt tay, SDK chính thức đã dùng nó; (2) kiểm bằng SDK chính thức làm client (`scripts/verify_mcp.py`) mới lộ lỗi lược đồ, test tự viết không đủ; (3) việc dài phải là job nền + `wait`, không chặn lời gọi tool; (4) forward hook trả giá trị khác None sẽ thay đầu ra của module (lỗi trong test của v1.4).
- Kết quả QA từng bộ: `examples/README.md`.

## 10. Quy ước cho agent (người/AI) phát triển repo

- Mọi thay đổi kiến trúc phải cập nhật `agent.md` và `ArchSpec` schema.
- Mỗi module mới: có test shape + forward/backward + docstring nêu *trục nào của paper* nó hiện thực.
- Không thêm phụ thuộc nặng ngoài danh sách §6 nếu chưa thống nhất.
- Kết quả thực nghiệm lưu kèm seed/config để tái lập.
- Không commit dữ liệu thô hoặc khóa/bí mật.
- **Môi trường Python:** không commit `.venv` (nặng hàng GB, riêng từng máy, vượt giới hạn 100 MB/file của GitHub; `tests/test_env_setup.py` chặn). Môi trường tái lập bằng `scripts/setup_env.sh` / `scripts/setup_env.ps1` (hoặc `requirements*.txt`), kiểm bằng `dlnet selfcheck`; job CI `env-setup` chạy đúng quy trình này trên Ubuntu, macOS, Windows. Thêm phụ thuộc mới: sửa `pyproject.toml` (extras `design`/`dev`), không sửa riêng script.
- Trước khi làm, chọn skill phù hợp trong `skills/` (§15).
- **Mọi thay đổi hay nâng cấp đều phải cập nhật `README.md` và tài liệu trong `doc/`** (§17, quy tắc 0); CI kiểm bằng `python scripts/release.py docs`.
- **Thuật toán chuẩn:** mọi bộ ước lượng chất lượng mạng hoặc hyperparameter phải tham khảo `ref/optimazeDL/` và implement đúng công thức của paper/mã chính thức (§16); không tự chế.
- Mỗi thay đổi người dùng thấy được phải thêm một dòng vào `CHANGELOG.md` mục `[Unreleased]`; phát hành theo §11.2.
- **Chạy CI trên máy trước khi push** (`python scripts/ci_local.py`; hook pre-push cài bằng `--install`), để lỗi lint/test không lên GitHub rồi mới lộ (§11).
- Mỗi lỗi sửa phải có regression test đi kèm (`tests/test_regression.py`, đỏ trước xanh sau); thay đổi chỉ xanh khi `lint`, `test`, `regression`, `e2e` đều xanh.

## 11. Quy trình Git, CI/CD & Release

**Chỉ có đúng hai nhánh, không tạo thêm:**
- `dev`: nơi duy nhất để phát triển; commit và push trực tiếp vào `dev`.
- `main`: luôn ổn định, chỉ nhận code từ `dev` (tự động, sau khi CI xanh); không ai push trực tiếp.
- Mọi nhánh khác (kể cả nhánh do công cụ tự tạo) phải bị xóa ngay; không dùng nhánh hotfix hay feature. Sửa nóng cũng đi qua `dev`.

### 11.1 Luồng đầy đủ

```
commit ► push dev ► CI (lint · test 3.10-3.12 · regression · e2e)
                      │ tất cả xanh
                      ▼
              promote: tạo PR dev→main và merge tự động
                      │ merge xong
                      ▼
              release: version trong pyproject.toml đã có tag vX.Y.Z chưa?
                 ├─ rồi  → dừng (main cập nhật nhưng không phát hành)
                 └─ chưa → kiểm tra version·CHANGELOG
                           ► installers.yml: Windows · macOS · Linux, mỗi máy build,
                             chạy thử chương trình, tạo gói cài, CÀI THẬT, dùng, GỠ
                           ► đủ mọi gói cài? (release.py assets) ─ thiếu → dừng, không tag
                           ► build wheel, thử cài, tạo tag vX.Y.Z và GitHub Release
                             (wheel + sdist + gói cài 3 OS + SHA256SUMS)
```

Workflow (`.github/workflows/`):
- `ci.yml`: chạy khi push vào `dev`/`main`, khi được `promote.yml` gọi, và hàng tuần (bắt lỗi do nâng cấp thư viện). Các job: `lint` (ruff check + format), `test (py3.10–3.12)` (unit + regression + kiểm tra release, torch CPU, **fail nếu có test bị skip**), `regression` (riêng, để thấy ngay lỗi cũ quay lại), `e2e` (CLI thật → gói xuất ra → `pytest` và `train.py` trong gói, cho tabular và HSI). Không có trigger `pull_request`: PR do `GITHUB_TOKEN` tạo chưa từng chạy được job nào (8 lần đều đỏ hoặc `action_required`, 0 job); kiểm tra thật nằm ở lần push vào `dev` trên đúng SHA đó.
- `promote.yml`: khi push vào `dev`, gọi `ci.yml`; **mọi job xanh** và `dev` có commit mới hơn `main` → tạo PR `dev → main` và merge (nếu merge ngay không được thì bật auto-merge); xong thì gọi `release.yml`.
- `release.yml`: tạo tag và GitHub Release từ `main` (xem 11.2). Idempotent; cũng chạy được tay (`workflow_dispatch`, dùng khi auto-merge bị trễ) và hàng tuần để bắt kịp.
Thiết lập thủ công trên GitHub (không làm được từ code): xem `.github/README.md`.

### 11.2 Release theo version tag

- **Phiên bản** theo SemVer `X.Y.Z`. Trước 1.0: tăng **minor** khi thêm tính năng hoặc đổi hành vi (kể cả định dạng gói xuất ra, CLI), tăng **patch** khi sửa lỗi hoặc tài liệu. Từ 1.0: theo SemVer đầy đủ.
- **Nguồn sự thật:** `pyproject.toml` (`version`). `dlnet/__init__.py` (`__version__`) và `CHANGELOG.md` phải khớp; `tests/test_release.py` và `python scripts/release.py check` bắt lệch.
- **Tag:** `vX.Y.Z`, chỉ tạo trên `main` bởi `release.yml`; không tạo hay di chuyển tag bằng tay. Không đẩy lại một tag đã công bố: sai thì phát hành bản vá mới.
- **Gói cài bắt buộc theo tag (từ v1.5):** mỗi tag phải kèm đủ 6 tệp `DLNet-<v>-windows-x64-setup.exe`/`.zip`, `-macos-arm64.pkg`/`.tar.gz`, `-linux-x64.deb`/`.tar.gz` và `SHA256SUMS-<os>-<arch>.txt`. Gói chỉ được tính khi `scripts/build_installer.py` (chạy thử chương trình đã đóng gói) **và** `scripts/test_install.py` (cài thật, gọi `dlnet` từ PATH, gỡ) đều qua. Thiếu một tệp thì `release.yml` dừng trước khi tạo tag. Không phát hành bằng tay để lách quy tắc này. Sửa gì trong `packaging/` thì push `dev` để `installers.yml` chạy lại trước khi bump. Hướng dẫn: `doc/install.md`.
- **Changelog:** `CHANGELOG.md` (Keep a Changelog). Mỗi thay đổi người dùng thấy được thêm một dòng vào `[Unreleased]` ngay trong commit gây ra nó (mục Thêm, Đổi, Sửa, Loại bỏ, Bảo mật, Hạn chế đã biết).

**Cách phát hành một phiên bản (toàn bộ trên `dev`):**
1. Đảm bảo CI trên `dev` xanh và không còn việc dở cho bản này.
2. Quyết định số phiên bản theo quy tắc trên.
3. Trong **một commit** `release: vX.Y.Z`: đổi `## [Unreleased]` thành `## [X.Y.Z] - YYYY-MM-DD` và thêm `## [Unreleased]` rỗng lên trên; sửa `version` trong `pyproject.toml` và `__version__` trong `dlnet/__init__.py`; cập nhật các dòng link cuối `CHANGELOG.md`.
4. Chạy `python scripts/release.py check` và `python3 -m pytest -q -m "not e2e"`; push `dev`.
5. Đợi `promote` rồi `release` xanh; kiểm tra có tag và Release mới (`gh`/trang Releases). Nếu `promote` chỉ bật auto-merge mà chưa merge, chạy tay workflow `Release` sau khi PR đã merge.
6. Báo cho người dùng số phiên bản và link Release.

Push lên `dev` mà **không** tăng version thì vẫn lên `main` nhưng **không** tạo bản phát hành.
**Gỡ bản lỗi:** không xóa tag/Release đã công bố; sửa trên `dev` rồi phát hành patch kế tiếp, ghi mục "Sửa" và nêu rõ bản lỗi trong CHANGELOG.

## 12. Tiết kiệm thời gian & Giao thức đánh giá

### 12.1 Mục tiêu tiết kiệm thời gian
Đo bằng **time-to-good-baseline**: từ lúc đưa dữ liệu vào đến lúc có baseline đạt chất lượng đã kiểm chứng. Cơ chế:
1. **Một lệnh**: `dlnet design --data cube.mat --gt gt.mat --budget 2h` (mọi bước dưới tự động).
2. **Cache dùng chung**: tiền xử lý (PCA/Tucker, patch, split) làm một lần cho mọi ứng viên.
3. **Cost model chặn sớm**: ước lượng params/FLOPs/VRAM của từng ứng viên, loại cái không vừa trước khi huấn luyện.
4. **Loại sớm**: zero-cost proxy xếp hạng sơ bộ, rồi Successive Halving (huấn luyện ngắn → giữ nửa tốt nhất → huấn luyện dài hơn).
5. **Warm-start** bằng kho kết quả benchmark trước đó (dữ liệu tương tự → bắt đầu từ ứng viên từng thắng).
6. **Resume & song song**: checkpoint từng ứng viên, chạy nhiều ứng viên cùng lúc theo VRAM.
7. **Một báo cáo leaderboard duy nhất** thay cho việc tự gom kết quả.

### 12.2 Giao thức đánh giá (bắt buộc; cấu hình theo plugin, ví dụ dưới đây là HSI)
- **Cách chia dữ liệu**
  - Chuẩn few-shot theo pixel (vd. 10% / 5% / N mẫu mỗi lớp), nhưng **cảnh báo rò rỉ không gian**: patch của pixel train và test chồng lấn nhau làm OA bị thổi phồng. Mặc định có thêm chế độ **chia theo vùng không chồng lấn (spatial block split)** để cho con số trung thực.
  - Mọi chế độ ghi rõ trong `report.md` (seed, kích thước patch, số mẫu mỗi lớp).
- **Bộ baseline tham chiếu** chạy trên đúng cùng split: SVM-RBF, Random Forest, 1D-CNN, 2D-CNN, 3D-CNN, HybridSN (có thể thêm SSRN, SpectralFormer khi library đủ).
- **Metric**: OA, AA, Kappa, F1 từng lớp; tác vụ khác dùng PSNR/SSIM/SAM (khôi phục), RMSE/SAD (unmixing), AUC (anomaly/target).
- **Lặp lại**: ≥ 5 seed, báo cáo trung bình ± std.
- **Quy tắc chấp nhận**: mô hình được đề xuất chỉ khi tốt hơn baseline tham chiếu mạnh nhất với chênh lệch lớn hơn std, và dùng cùng split. Nếu không, `report.md` nói thẳng "chưa vượt baseline" và trả về baseline tốt nhất thay vì mô hình tự ghép.
- **Tái lập**: seed, phiên bản thư viện, commit, config đều ghi trong báo cáo.

### 12.3 Xếp hạng và xuất Top-K
**Mục tiêu tối ưu**: metric chính của plugin (accuracy, F1, RMSE, PSNR, AUC…) trên tập validation đúng giao thức; có thể thêm ràng buộc params/FLOPs/latency.

**Quy trình chấm điểm theo tầng** (ngân sách tăng dần, số ứng viên giảm dần):
1. Lọc bằng prior + cost model → ~50–200 ứng viên hợp lệ.
2. Zero-cost proxy → giữ ~30. (Hiện chỉ đo và ghi tương quan, chưa loại ứng viên: xem §16.3.)
3. Short-train (Successive Halving) → giữ ~8–10.
4. Huấn luyện vừa, nhiều seed và nhiều lần chia (`--cv-splits`) → xếp hạng cuối theo trung bình ± std.

**Chọn Top-K**:
- Xếp theo metric trung bình; ứng viên có chênh lệch nhỏ hơn std được gom thành **nhóm "chưa phân biệt được"** và ghi rõ trong leaderboard.
- **Đa dạng**: không để K slot bị chiếm bởi các biến thể gần giống nhau; cố gắng phủ nhiều họ kiến trúc.
- Gắn nhãn vai trò: `top1` (chính xác nhất), `best-small` (hiệu quả/độ chính xác), `most-stable` (std thấp nhất), `most-different` (họ khác, đáng thử).

**Mỗi baseline trong Top-K kèm**: điểm ± std, #params/FLOPs/VRAM, thời gian train đã dùng, **hyperparameter đầy đủ** (`hyperparameters` trong `config.yaml` và `leaderboard.json`, xem §16.2: optimizer, lr, weight decay, batch size, epochs, lr schedule, dropout, width/depth/rank, cùng nguồn gốc: tìm kiếm hay ước lượng) và lưu ý rủi ro.

**Chính trực**: nếu không ứng viên nào vượt bộ baseline tham chiếu (§12.2), `leaderboard` ghi rõ và đưa baseline tham chiếu tốt nhất lên đầu.

**CLI**:
```bash
dlnet design --data data.csv --task auto --budget 2h --topk 5
```

## 13. Module Library & tự động ghép baseline

### 13.1 Registry
**Hiện trạng (v0.3.0):** registry nằm trong `dlnet/hsi_lib.py` (một file vì nó được copy thành `model.py` của gói xuất ra): `BLOCKS`, `@register(kind, name)`, lớp `Block` với `features(in_ch, width, depth, patch)` là suy luận shape tĩnh; `plan_composed` kiểm tra grammar; `sample_composed` lấy mẫu; `PRESETS` là các họ preset. Cost vẫn là đếm tham số (FLOPs/VRAM: M5). Đoạn dưới là thiết kế đích.

Mọi khối nằm trong `dlnet/library/` và đăng ký qua một giao diện chung:
```python
@register("spectral_encoder", name="spectral_mamba")
class SpectralMamba(Block):
    in_spec = TensorSpec(("B", "L", "C"))  # hợp đồng shape vào
    out_spec = TensorSpec(("B", "L", "D"))  # hợp đồng shape ra

    def cost(self, in_shape) -> Cost: ...  # params, FLOPs, VRAM
```
Hợp đồng shape cho phép **suy luận shape tĩnh và loại ứng viên ghép sai** trước khi tạo model.

### 13.2 Danh mục khối (v1 → mở rộng)
| Loại | Khối ví dụ |
|---|---|
| Stem / tiền xử lý | PCA, chọn băng (band selection), **Tucker/HOSVD giảm chiều**, trích patch không gian–phổ |
| Mã hóa phổ | 1D-CNN, spectral attention, Transformer theo băng, Mamba/SSM |
| Mã hóa không gian | Conv 2D, ResNet block, depthwise-separable, ConvNeXt, ViT/Swin nhẹ |
| Mã hóa chung | Conv 3D, khối kiểu HybridSN (3D → 2D) |
| Hợp nhất | concat, gated, cross-attention |
| Lớp tensor | Linear TT/TR, Conv Tucker, Tensor-LoRA (§4) |
| Khối unfolded | ISTA/ADMM unrolled cho unmixing, fusion, khôi phục (§5) |
| Đầu ra | phân loại, phân vùng, tái tạo |

### 13.3 Preset baseline nổi tiếng
Mỗi mô hình kinh điển được viết lại như **một cấu hình của registry** (không phải code riêng), ví dụ: 1D-CNN, 2D-CNN, 3D-CNN, HybridSN, SSRN, SpectralFormer, SSFTT, các biến thể Mamba cho HSI. Lợi ích: vừa là bộ baseline tham chiếu (§12.2), vừa là điểm xuất phát của tìm kiếm. Đã có dạng preset: `resnet2d`, `dwsep2d`, `convnext2d`, `spec_transformer`, `ssrn` (viết từ trí nhớ, chưa đối chiếu mã gốc). Danh sách này là tham chiếu; **phải kiểm tra lại nguồn gốc và tái hiện số liệu khi cài đặt**, chưa có trong `ref/`.

### 13.4 Tự ghép (grammar có kiểu)
```
Model := Stem → ( SpectralEnc ∥ SpatialEnc | JointEnc ) → Fusion → Head
```
1. Chọn nhánh theo `ProblemSpec` (vd. `has_forward_model` → thêm khối unfolded; rank thấp rõ → lớp tensor).
2. Lấy mẫu từ prior (preset trước, tổ hợp ghép sau), kiểm tra shape, tính cost.
3. Loại ứng viên vi phạm ngân sách, giữ Top-K đưa vào Search (§3.4).
4. Siêu tham số (độ sâu, độ rộng, rank, kích thước patch) do Search chọn.

### 13.5 Thư viện khối dùng chung `dlnet/blocks.py` (M10)
Thư viện khối torch-only, được copy vào gói xuất ra (`blocks.py`). Nguồn, commit và giấy phép: `THIRD_PARTY.md`. Liệt kê: `python -m dlnet blocks`.

| Loại (`kind`) | Khối | Nguồn | Trục của paper 2302.11529 |
|---|---|---|---|
| stage (backbone) | `lenet`, `allconv`, `vgg`, `resnet`, `densenet` | rasbt/deeplearning-models (MIT), viết lại | computation |
| stage (backbone) | `mobilenetv2` | paper, từ trí nhớ | computation |
| backbone | `Encoder` (stem + các stage, trả feature mỗi mức stride) | DLNet | function composition tuần tự |
| attention | `se`, `self_attention_2d` | paper / `torch.nn.MultiheadAttention` | computation |
| decoder | `fcn`, `unet` (paper, từ trí nhớ), `deconv` (rasbt) | | aggregation ở mức representation (skip) |
| head | `Classifier`, `Segmenter` | DLNet | |
| routing | `MixtureOfExperts` (soft, top-k, product) | phương pháp của yamsgithub/modular_deep_learning, **cài lại, không sao chép** (repo không có giấy phép) | routing học được + aggregation |
| interaction / loss | `LatentFactor`, `SecondOrderInteraction`, `pairwise_log_loss` | ylongqi/openrec (Apache-2.0), chuyển từ TensorFlow | computation |
| composition | `NeuralTree`, `SumTrunk`, `MeanPooling` | prescient-design/cortex (Apache-2.0), viết lại | aggregation (nhiều đầu vào, nhiều tác vụ) |

Dùng ở đâu:
- **Xây mạng:** họ ảnh `lenet`, `allconv`, `vgg`, `densenet`, `mobilenetv2` (tùy chọn `attention`), họ segmentation `fcn`, `unet`, `deconvnet` (tùy chọn `encoder`, `attention`), họ bảng `moe_mlp`, `fm_mlp`.
- **Ước lượng:** mọi ứng viên dựng từ thư viện đều đi qua cùng pipeline (đếm tham số làm cost model, zero-cost proxy, successive halving, LR range test, noise scale).
- Chưa được plugin nào dùng: `NeuralTree`, `SumTrunk`, `MeanPooling`, `LatentFactor`, `pairwise_log_loss`, `self_attention_2d` chỉ khi chọn `attention: self`. Có test nhưng chưa có bài toán nhiều đầu vào / recommendation.
- Không có trọng số pretrained; không khối nào là bản tái hiện đã kiểm chứng của mô hình gốc.

### 13.6 Kho module `dlnet/zoo/` (M11): nơi kiến trúc mới được ghép và tìm ra
Mục đích của platform là **tìm kiến trúc mới cho từng ứng dụng**, nên nguồn ứng viên không còn là một danh sách họ cố định mà là một **không gian kiến trúc**: module có kiểu + toán tử ghép + tìm kiếm trên cây ghép. Toàn bộ torch-only và được copy vào gói xuất ra (`zoo/`).

**Kiểu tensor** (bỏ trục batch): `vec (D)`, `seq (T,D)`, `map (C,H,W)`, `ser (C,T)`, `vol (C,D,H,W)`. HSI là `map` với C = số băng sau PCA; ảnh là `map`; chuỗi thời gian là `ser`; bảng là `vec`.

**Module (`@op`)**: khai báo kiểu vào, kiểu ra, không gian tham số rời rạc (`space`), thẻ chức năng (`block`, `down`, `attn`, `stem`, `adapt`, `norm`, `tensor`, ...) và nguồn gốc. Hiện có 54 op (`python -m dlnet zoo`): khối conv tổng quát cho 1-D/2-D/3-D (basic, bottleneck/ResNeXt/Wide, pre-act, dense, NiN, LeNet, VGG, MBConv, fused MBConv, MobileNetV3, ConvNeXt, RegNet, ShuffleNet, Fire, Inception, depthwise-separable), attention (SE, ECA, CBAM, self-attention không gian, window attention kiểu Swin, transformer block, attention chuỗi), chuỗi (RNN/GRU/LSTM, MLP-Mixer, Mamba, conv1d residual kiểu cortex, TextCNN), vector (MLP, FFN, GLU, TT/TR/low-rank, tương tác FM, cross network), cầu nối giữa các kiểu (patchify, tokens, pooling, feature tokens, cls token, vị trí).

**Kiến trúc = cây spec JSON** gồm lá `{"op": ...}` và toán tử ghép: `seq`, `res` (residual, tự chiếu 1×1 khi đổi shape, có stochastic depth), `par` (concat/sum/gate), `rep` (lặp, có chia sẻ trọng số), `moe` (soft / top-k / product-of-experts, expert là spec bất kỳ), `encdec` (encoder–decoder cho dự đoán dày: unet, fcn, fpn, aspp, deconv). Đầu ra (head) không nằm trong spec: được thêm theo tác vụ (`cls` hoặc `dense`), nên đột biến không thể phá head.

**Kiểm tra tĩnh**: `zoo.check(spec, kind, dims)` dựng mô hình trên thiết bị `meta` và chạy một batch giả, trả số tham số và shape đầu ra, hoặc `ValueError` (sai kiểu, bản đồ co về 0, số head không chia hết độ rộng...) trong vài mili-giây, **trước khi huấn luyện**. Cost model hiện chỉ là đếm tham số.

**Tìm kiếm có phương pháp** (`zoo/space.py`):
1. **Ứng viên ban đầu** (`--zoo N`, mặc định 12): tối đa N/2 preset kiến trúc nổi tiếng viết dưới dạng spec (LeNet, VGG, ResNet/ResNeXt/Wide/pre-act/SE, DenseNet, MobileNet, EfficientNet, ConvNeXt, RegNet, ShuffleNet, SqueezeNet, NiN, Inception, ViT, Swin, hybrid CNN-transformer, TCN, TextCNN, GRU/LSTM, Mamba, MLP-Mixer, FT-Transformer, DCN, MoE, TT/TR-ResMLP, U-Net/FCN/FPN/DeepLab) và phần còn lại là mẫu ngẫu nhiên từ ngữ pháp theo thẻ (stem → các tầng độ phân giải → khối bổ sung → attention).
2. **Tiến hóa** (`--evolve G`, mặc định 1 thế hệ, `--evolve-pop` con mỗi thế hệ): cha mẹ là các ứng viên zoo tốt nhất theo validation ở rung 1; con sinh bằng 8 phép đột biến (`param`, `swap` cùng kiểu và thẻ, `insert`, `delete`, `wrap_res`, `repeat`, `widen`, `to_moe`) và lai ghép theo tầng; mọi con đều qua `zoo.check` và ngân sách tham số, loại bản trùng (chữ ký SHA-1 của spec).
3. Mọi ứng viên đi qua **cùng pipeline cũ**: successive halving, nhiều seed và lần chia, LR range test, noise scale, chọn Top-K đa dạng, verdict so với baseline cổ điển. Nhãn `origin` (preset / random / mutation / crossover) ghi trong leaderboard và `report.md` để biết kiến trúc đến từ đâu.

**Thêm module mới**: một hàm có decorator `@op(name, kinds, space=..., tags=..., source=...)`; không cần sửa ngữ pháp, đột biến hay test: test tự quét mọi op trên mọi kiểu và mọi giá trị tham số (`tests/test_zoo.py`). Skill: `module-zoo`.

**Nguồn và đối chiếu** (`THIRD_PARTY.md`): rasbt/deeplearning-models, prescient-design/cortex, ylongqi/openrec, yamsgithub/modular_deep_learning (bốn repo bạn chỉ định), thêm pytorch/vision và state-spaces/mamba. Giấy phép không được xét (công cụ nội bộ). `scripts/verify_zoo.py` so 14 khối với torchvision gốc: 14/14 khớp số tham số và shape (ghim ở `tests/test_zoo.py::GOLDEN`). Danh sách lớp chưa port và lý do nằm ở `THIRD_PARTY.md`.

**Kết quả đo (`python scripts/bench_zoo.py`, ngân sách 100 s mỗi lần, 2 seed, 6 bộ dữ liệu; "họ cũ" = `--zoo 0 --evolve 0`, "zoo" = `--zoo 14 --evolve 1 --evolve-pop 8`):**

| Bộ dữ liệu | seed | val top-1 họ cũ | test họ cũ | val top-1 zoo | test zoo | Người thắng của zoo (nguồn) |
|---|---|---|---|---|---|---|
| breast_cancer | 0 | 0,9617 | 0,9768 | 0,9617 | 0,9623 | lowrank_linear (đột biến insert) |
| breast_cancer | 1 | 0,9735 | 0,9826 | 0,9852 | 0,9768 | lowrank+tr+tt linear (ngẫu nhiên) |
| wine | 0 | 0,9804 | 0,9915 | 0,9804 | 0,9829 | họ cũ `resmlp` |
| wine | 1 | 0,9608 | 0,9487 | 0,9804 | 0,9744 | lowrank+mlp+tr (đột biến repeat) |
| make_classification 1500×30 | 0 | 0,6544 | 0,6370 | 0,6790 | 0,6634 | `mlp` (đột biến param) |
| make_classification 1500×30 | 1 | 0,6980 | 0,6920 | 0,7013 | 0,6678 | `tt_linear` (lai ghép) |
| ảnh digits 8×8 | 0 | 0,9944 | 0,9973 | 0,9916 | 0,9946 | bottleneck + MoE (đột biến to_moe) |
| ảnh digits 8×8 | 1 | 0,9944 | 0,9973 | 0,9916 | 1,0000 | họ cũ `cnn` |
| chuỗi thời gian tổng hợp | 0 | 0,9887 | 0,9785 | 1,0000 | 1,0000 | mixer + mlp (ngẫu nhiên) |
| chuỗi thời gian tổng hợp | 1 | 0,9859 | 0,9839 | 0,9915 | 0,9839 | mixer + mlp (đột biến repeat) |
| phân đoạn tổng hợp (mIoU) | 0 | 0,9148 | 0,9339 | 0,9208 | 0,9333 | basic_block + CBAM + convnext (đột biến widen) |
| phân đoạn tổng hợp (mIoU) | 1 | 0,9232 | 0,9412 | 0,9243 | 0,9381 | basic + convnext + pre-act (ngẫu nhiên) |

- **Validation**: zoo cao hơn ở 8/12 cặp, bằng ở 2, thấp hơn ở 2 (digits, chênh một mẫu). **Test**: zoo cao hơn ở 4/12, bằng 1, thấp hơn 7. Phần lớn lợi thế validation là **thiên lệch lựa chọn** (top-1 của nhiều ứng viên hơn nên validation lạc quan hơn), test không xác nhận. Verdict so với baseline cổ điển đổi từ không thắng sang thắng ở một cặp (chuỗi thời gian, seed 0).
- Người thắng của zoo thường đến từ đột biến hoặc ngẫu nhiên (10/12), không phải từ preset: tức là tìm kiếm có tạo ra cấu trúc mới được chọn, nhưng **chưa có bằng chứng chúng tổng quát hóa tốt hơn họ cũ** ở ngân sách này.
- Giới hạn của phép đo: bộ dữ liệu nhỏ hoặc tổng hợp, ngân sách 100 s (ứng viên chỉ được huấn luyện 4–30 epoch), 2 seed, chỉ 1 thế hệ tiến hóa. Chưa có kết luận cho ngân sách giờ trở lên hoặc dữ liệu thật lớn. Việc nên làm tiếp để kiểm tra: chạy lại với `--budget 30m --evolve 3` trên dữ liệu thật của bạn.

**Họ mô hình = công thức trên zoo (M12)**: `library.py`, `array_lib.py`, `hsi_lib.py`, `blocks.py` đã bị gỡ. 33 họ (bảng 8, ảnh 8, phân đoạn 3, chuỗi thời gian 3, HSI 11) là các hàm `cấu hình → spec` trong `dlnet/zoo/families.py`; plugin vẫn lấy mẫu `SPACE` nhỏ như trước nhưng mọi mạng đều là kiến trúc zoo, nên cùng một registry, bộ kiểm tra tĩnh, cost model, đột biến và lai ghép. Gói xuất ra có `model.py` là shim 1 dòng (`from zoo import build_model`) cùng thư mục `zoo/`. 22/33 họ có số tham số trùng tuyệt đối với bản cũ, phần còn lại lệch ≤ 10% (xem `THIRD_PARTY.md`). HSI cần thêm các module `center_spectrum`, `to_vol`, `vol_to_map`, `tucker_conv`, `pool_flatten`, nhân conv theo từng trục (`k=[7,3,3]`) và hợp nhất có cổng (`par` với `merge: "gated"`).

**Cost model (M12)** (`dlnet/zoo/cost.py`, áp cho mọi ứng viên, cũ lẫn mới): đo trên thiết bị `meta` (không tốn bộ nhớ): FLOPs ma trận/tích chập của một mẫu (`FlopCounterMode`; không tính phép elementwise, chuẩn hóa, kích hoạt và vòng lặp quét của Mamba), bộ nhớ kích hoạt từ hook trên các module lá, và VRAM huấn luyện ước lượng `16·params + 2·batch·act` byte (trọng số, gradient, hai moment Adam, kích hoạt lưu và gradient của chúng). `--max-flops 50M` và `--max-vram 4000` (MB) loại ứng viên **trước khi huấn luyện**; leaderboard có cột MFLOPs và MB. **Chưa đối chiếu với VRAM đo trên GPU** (môi trường không có GPU): coi là ước lượng ±30%.

### 13.7 Các kiểu dữ liệu và tác vụ mới trong zoo (M13)
- **Kiểu `graph`** `(N, F+N)` (đặc trưng nút rồi hàng kề của nó; đồ thị dày, N nhỏ) với các lớp GCN, GAT, GraphSAGE, GIN, readout; test bất biến theo thứ tự nút (hoán vị nút và ma trận kề không đổi đầu ra). Hai bộ lọc lưới của rasbt: `grid_graph_avg`, `grid_spectral_conv` (Laplacian của đồ thị pixel, `eigh` thay `eigsh`, bộ lọc được huấn luyện).
- **Kiểu `ids`** (dims là số giá trị của từng trường, tensor `(B,F)` nguyên): `embed_fields` (LatentFactor của openrec), `gmf`, `pair_dots`; tác vụ điểm (pointwise) dạng hồi quy hoặc phân loại. Cold-start: id chưa thấy ở train giữ embedding khởi tạo.
- **Tác vụ `reconstruction`**: mục tiêu là chính ảnh vào, metric MSE và PSNR (điểm = −MSE); `vae_bottleneck` thêm KL do `ZooNet.kl_loss()` gom và trainer cộng `beta·KL`. Baseline PCA 0/8/32 thành phần: **công bằng phụ thuộc kích thước mã** của ứng viên (xem `code` trong `config.yaml`), verdict không tự khớp kích thước mã.
- **Lệnh `dlnet gan`**: DCGAN dựng từ zoo, huấn luyện Adam(2e-4, β=0,5/0,999) với loss generator không bão hòa, chọn checkpoint theo MMD² (nhân Gaussian, băng thông trung vị × {0,5; 1; 2}) trên ảnh validation; baseline là Gaussian khớp ảnh train và chính ảnh train. **Không có FID** (không có mạng Inception): MMD² trong không gian pixel là thước đo thô, ưu ái ảnh mờ.
- **Đo trên dữ liệu tổng hợp (một lần chạy 120 s, `design`, 8 cấu hình zoo + 1 thế hệ tiến hóa):**

| Bài toán | Dữ liệu | Top-1 | Baseline tốt nhất | Kết luận |
|---|---|---|---|---|
| đồ thị | 400 đồ thị 16 nút: ER so với thưa hơn cộng tam giác cài sẵn | GIN+GCN: val 0,72, test 0,75 | HistGradientBoosting trên thống kê đồ thị: val 0,90, test 0,87 | **GNN kém baseline** (baseline có đặc trưng tam giác và hệ số cụm đúng thứ phân biệt hai lớp) |
| recommender | 150 người dùng × 80 mục, 8000 tương tác sinh từ phân rã hạng 4 | MF: RMSE test 0,129 | random forest trên one-hot: RMSE 1,77 | MF thắng xa, **nhưng dữ liệu sinh đúng theo mô hình MF** nên chỉ chứng minh khôi phục được cấu trúc đó |
| autoencoder | digits 8×8 (thật) | `conv_ae` MSE 0,039 (mã 8–32 phần tử) | PCA 32 thành phần: MSE 0,019 | **PCA thắng** trên ảnh nhỏ gần tuyến tính |

- **Cost model áp cho cả ứng viên con** của vòng tiến hóa.

### 13.8 Khối ảnh từ thư viện chính thống (M14)
- **Nguồn**: timm 1.0.31 (thư viện khối ảnh lớn nhất của PyTorch), segmentation_models_pytorch 0.6.0 (decoder phân đoạn), MONAI 1.6.1 (ảnh y tế 2-D/3-D), torchvision 0.29.2. Port bằng PyTorch thuần (`dlnet/zoo/ops_img.py`, `dlnet/zoo/dense.py`), không thành phụ thuộc lúc chạy.
- **Mới**: 9 decoder smp (`unet`, `unetpp`, `fpn`, `deeplabv3`, `deeplabv3plus`, `pspnet`, `linknet`, `upernet`, `segformer`) + `attention: scse`; 13 op (`scse`, `coord_attn`, `gc_block`, `grn`, `sk_conv`, `split_attn`, `ghost_bottleneck`, `res2net_block`, `convmixer_block`, `repvgg_block` có `fuse()`, `poolformer_block`, `segres_block`, `unet_res_block`); preset `ghostnet`, `repvgg`, `poolformer`, `convmixer`, `segresnet`, `nnunet_res`, `res2net`, `sknet`, `resnest`, `resnet_cbam/eca/coord/gc` và 8 preset phân đoạn; phép đột biến `decoder`; 8 họ phân đoạn mới. Zoo có 85 op.
- **Sửa theo mã gốc**: `cbam`, `eca`, `inception`, `window_attn` (relative bias + mask), `mixer_block`; decoder `unet`/`fpn`/`aspp` cũ thay bằng bản smp (`aspp` thành tên khác của `deeplabv3`).
- **Đối chiếu**: `scripts/verify_vision_libs.py` 39/39 khớp (THIRD_PARTY.md). Cùng với `verify_numeric.py` (24/24), tổng 63 khối/decoder đã đối chiếu số học với mã gốc.
- **Chưa đo** khối mới có giúp độ chính xác trên dữ liệu thật hay không: phép đo nằm trong M15 (chọn hạt giống theo dữ liệu).

## 14. Domain plugins (HSI là plugin đầu tiên)

### 14.1 Giao diện plugin
Mỗi loại dữ liệu cung cấp cùng một giao diện; lõi không biết chi tiết từng domain:
```python
class DomainPlugin:
    name: str
    def detect(self, data) -> float            # độ tin cậy "đây là dữ liệu của tôi"
    def load(self, path) -> Dataset            # loader
    def split(self, ds, mode, seed) -> Splits  # chia dữ liệu chống rò rỉ đặc thù domain
    def metrics(self, task) -> list[Metric]    # metric chuẩn của domain
    def stems(self) -> list[Block]             # tiền xử lý / giảm chiều
    def presets(self) -> list[ArchSpec]        # baseline nổi tiếng của domain
    def prior(self, profile) -> Rules          # luật chọn họ kiến trúc
```

| Plugin | Dữ liệu | Preset ví dụ | Ưu tiên |
|---|---|---|---|
| `tabular` | bảng đặc trưng | MLP, ResNet-MLP, FT-Transformer | M1 (kiểm chứng lõi nhanh) |
| `hsi` | khối `H×W×B` siêu phổ | 1D/2D/3D-CNN, HybridSN, SSRN, SpectralFormer | M2 |
| `image_classification` (tên cũ `image` vẫn dùng được) | ảnh RGB/xám, `.npz` có `X` (N,H,W[,C]) và `y` (N) | **đã có:** `cnn`, `resnet`, `vit`, `lenet`, `allconv`, `vgg`, `densenet`, `mobilenetv2` | M9, M10 |
| `graph` | `.npz` có `X` (G,N,F), `A` (G,N,N), `y` (G): phân loại cả đồ thị | **đã có:** `gcn_net`, `gat_net`, `sage_net`, `gin_net`, `graph_mlp`; baseline cổ điển trên thống kê đồ thị (mật độ, bậc, tam giác, hệ số cụm, phổ Laplacian) | M13 |
| `recsys` | `.npz` có `ids` (N,F) số nguyên và `y` (điểm hoặc nhấp) | **đã có:** `mf` (GMF), `ncf`, `fm`, `dlrm`; baseline one-hot | M13 |
| `image_reconstruction` | `.npz` có `X` ảnh, không nhãn | **đã có:** `conv_ae`, `vae` (mã 2–8 kênh); baseline PCA 0/8/32 thành phần | M13 |
| `image_segmentation` | ảnh + mask, `.npz` có `X` (N,H,W[,C]) và `y` (N,H,W) | **đã có:** `fcn`, `unet`, `deconvnet` × encoder `vgg`/`resnet`/`densenet`/`mobilenetv2`; metric mIoU | M10 (v0.9.0) |
| `timeseries` | chuỗi thời gian, `.npz` có `X` (N,T[,C]) và `y` | **đã có:** `tcn`, `gru`, `ts_transformer` | M9 (v0.8.0) |
| `signal`/khác | tín hiệu, tensor đa chiều | tuỳ domain | sau |

`--task auto`: Plugin Router gọi `detect` của từng plugin, chọn plugin tin cậy nhất, hỏi lại người dùng nếu nhập nhằng.

### 14.2 Plugin HSI (chi tiết)

**Dữ liệu**: khối `H × W × B` (B thường 100–250 băng), nhãn thưa theo pixel. Dataset benchmark chuẩn: Indian Pines, Pavia University, Salinas, Houston.

**Tác vụ hỗ trợ theo thứ tự ưu tiên**: phân loại → khôi phục/khử nhiễu → unmixing → anomaly/target detection.

**Vai trò của tensor decomposition (tách bạch)**
| Chỗ dùng | Cách làm |
|---|---|
| Tiền xử lý / giảm chiều | HOSVD/Tucker trên khối `H×W×B`, chọn multilinear rank bằng phổ singular từng mode (thay cho PCA chỉ theo phổ) |
| Khử nhiễu, bù dữ liệu | Low-rank tensor (Tucker/TT/TR) làm tiền xử lý hoặc làm lớp trong mạng |
| Nén mô hình | Lớp TT/TR cho FC, Tucker cho conv, Tensor-LoRA khi fine-tune (§4) |
| Unfolding | Mô hình tuyến tính `y = A x + n` (trộn phổ, đo nén, fusion HSI–MSI) → ISTA/ADMM unrolled (§5) |

**`ProblemSpec` bổ sung cho HSI**
```yaml
modality: hsi
cube: {height: 145, width: 145, bands: 200}
task: classification            # classification | denoise | unmixing | detection
labels: {classes: 16, labeled_ratio: 0.1}
split: {mode: spatial_block, seeds: 5}
budget: {time: 2h, vram_gb: 12}
```

**Cạm bẫy đã biết**: rò rỉ do patch chồng lấn; chọn siêu tham số bằng chính tập test; số mẫu thiểu số rất ít; đánh giá một seed. Giao thức §12.2 xử lý các điểm này.

### 14.3 Plugin ảnh, chuỗi thời gian và recommender meta-learning (M9)
- **Plugin `image` / `timeseries`** (`dlnet/plugins/arrays.py`, `dlnet/array_data.py`, `dlnet/array_lib.py`): đọc `.npz` có `X` và `y`; chia 60/20/20 phân tầng; chuẩn hóa theo kênh bằng thống kê train; baseline tham chiếu cổ điển chạy trên đầu vào duỗi phẳng (PCA 64 chiều fit trên train nếu > 512 chiều). Router phân biệt theo số chiều của `X`: 4 chiều hoặc (N,S,S) là ảnh, (N,T,C) với T≠C là chuỗi; nhập nhằng thì dùng `--plugin`.
- Các họ mô hình viết từ trí nhớ theo ResNet (arXiv:1512.03385), ViT (arXiv:2010.11929), TCN (arXiv:1803.01271, lệch: padding không nhân quả, BatchNorm), **chưa đối chiếu** mã tác giả và các paper này chưa có trong `ref/`. Chưa có Mamba/SSM và ConvNeXt cho ảnh.
- Chỉ mới chạy trên dữ liệu tổng hợp nhỏ (thanh ngang/dọc 8×8; sin hai tần số), chưa có số liệu trên dữ liệu thật.
- **Recommender** (`dlnet/meta.py`, bật bằng `--meta-store file.jsonl`): mỗi lần chạy ghi một bản ghi (meta-feature: log n, log d, intrinsic dim, số lớp; điểm validation rung 1 tốt nhất của từng họ). Lần chạy sau, k=5 dataset gần nhất cùng plugin bỏ phiếu thứ tự họ (trung bình hạng chuẩn hóa, trọng số 1/(khoảng cách+0,1)). **Chỉ sắp thứ tự, không loại họ nào.**
- **Kết quả đo (`python scripts/bench_meta.py`, 12 dataset bảng, ngân sách 30s mỗi bộ, leave-one-dataset-out):**

| Top-k | kNN theo meta-feature | Mọi dataset (không lọc láng giềng) | Ngẫu nhiên |
|---|---|---|---|
| 1 | 4/12 = 33% | 3/12 = 25% | 17% |
| 3 | 7/12 = 58% | 8/12 = 67% | 51% |

- **Kết luận:** tiêu chí M9 "Top-K chứa mô hình tốt nhất trong ≥80% dataset giữ lại" **chưa đạt** (58%). Với 12 dataset, chênh lệch so với ngẫu nhiên (1 dataset) không có ý nghĩa thống kê, và meta-feature chưa hơn thứ tự toàn cục. Nhãn "họ tốt nhất" lấy từ rung 1 (4 epoch) nên nhiễu: bốn họ khác nhau chia nhau vị trí đầu. Cần kho lớn hơn nhiều (hàng chục đến hàng trăm dataset) và nhãn từ huấn luyện đầy đủ trước khi dùng recommender cho việc gì hơn sắp thứ tự.

### 14.4 Plugin ảnh phân loại và phân đoạn (M10)
- **Segmentation** (`image_segmentation`): `.npz` có `X` (N,H,W[,C]) và `y` (N,H,W); pixel nhãn âm bị bỏ qua khi tính loss và metric. Chia 60/20/20 theo **ảnh** (không chia theo pixel, tránh rò rỉ giữa các pixel cùng ảnh). Metric chính: mIoU (trung bình IoU trên các lớp có trong nhãn thật); có thêm pixel accuracy và kappa. Baseline tham chiếu: logistic regression và random forest trên lân cận 5×5 của từng pixel (tối đa 20.000 pixel train).
- Router nhận dạng theo `y`: `y` 3 chiều khớp `X` là segmentation; `y` 1 chiều là phân loại ảnh hoặc chuỗi.
- **Kết quả đo (`dlnet design`, ngân sách 4 phút, 2 ứng viên mỗi họ):**

| Bài toán | Dữ liệu | Top-1 (val) | Test | Baseline tốt nhất | Kết luận của verdict |
|---|---|---|---|---|---|
| phân loại ảnh | sklearn digits 8×8, 1797 ảnh, 10 lớp (dữ liệu thật, nhỏ) | `cnn` 0,9925 ± 0,0027; top-2 `vgg` 0,9925; top-3 `mobilenetv2` 0,9916 (89k tham số) | accuracy 0,9991 | random forest: val 0,9775, test 0,9891 | **thắng** (val +0,015 > nhiễu 0,005; test +0,010) |
| phân đoạn | tổng hợp: 300 ảnh 24×24, hai hình vuông sáng/tối trên nền nhiễu | `fcn` (4.797 tham số) mIoU 0,985 ± 0,006; `deconvnet` 0,951; `unet` 0,871 ± 0,172 | mIoU 0,989 | random forest 5×5: val 0,971, test 0,989 | **không thắng có ý nghĩa** (margin 0,0145 < nhiễu 0,0156; test +0,0004) |

- Giới hạn: digits rất dễ (mọi mô hình ≥ 0,97), 16 ứng viên và một lần chia, nên chưa nói được họ nào tốt hơn họ nào; hai ứng viên top đứng cùng nhóm "chưa phân biệt được". Segmentation chỉ có dữ liệu tổng hợp rất dễ (một baseline 5×5 đã gần tối đa), chưa có ảnh y tế / ảnh vệ tinh / Pascal thật; `unet` có độ lệch chuẩn lớn (0,17) nghĩa là huấn luyện chưa ổn định trong ngân sách ngắn. Sinh được 5 ứng viên hợp lệ cho segmentation (một số cấu hình bị loại vì shape hoặc ngân sách tham số).
- Chưa có: augmentation (lật, cắt), loss cho lớp mất cân bằng (Dice, focal), mô hình pretrained, đầu vào lớn (ảnh > 128×128 chưa kiểm tra tốc độ và bộ nhớ).

## 15. Skills (bắt buộc đọc trước khi làm)

Quy trình đã đúc kết nằm trong `skills/` (chỉ mục: `skills/README.md`).

**Quy tắc cho agent (người/AI):**
1. Trước khi làm bất kỳ việc nào, tìm skill phù hợp: đọc bảng trong `skills/README.md` hoặc chạy `grep -ril "<từ khóa>" skills/*/SKILL.md`.
2. Chọn skill khớp nhất, đọc đủ `SKILL.md` rồi làm theo. Việc nhiều phần có thể cần nhiều skill (ví dụ thêm plugin: `add-domain-plugin` + `evaluate-honestly` + `git-dev-main`).
3. Không có skill nào khớp: làm việc đó, rồi thêm skill mới (thư mục `skills/<tên>/SKILL.md` có `name`, `description`, `triggers`) và một dòng vào `skills/README.md`.
4. Skill sai hoặc lỗi thời: sửa skill trong cùng commit với thay đổi gây ra sự lệch.
5. Báo cáo cho người dùng nêu skill nào đã dùng.

## 16. Thuật toán chuẩn và ước lượng hyperparameter

### 16.1 Quy tắc: tham khảo và implement theo thuật toán chuẩn
`ref/optimazeDL/` chứa paper (nhóm `zc_` zero-cost proxy, `wq_` chất lượng trọng số, `hp_` hyperparameter, `bm_` benchmark) và mã chính thức của tác giả (`official_code/`, ghim commit trong `OFFICIAL_CODE.md`). Khi thêm hoặc sửa một bộ ước lượng chất lượng mạng hay hyperparameter:
1. Đọc paper và mã chính thức trước (skill `implement-standard-algorithm`). Công thức trong code phải khớp paper; mỗi hàm có docstring nêu paper + arXiv id + phương trình.
2. Điều gì chưa đối chiếu với paper (lấy từ trí nhớ hoặc cách làm phổ biến) phải ghi là "chưa đối chiếu" trong docstring.
3. Có test với **đáp án giải tích** (ví dụ ước lượng noise scale phải khớp công thức đóng) chứ không chỉ test hình dạng.
4. Zero-cost proxy chỉ được dùng để loại ứng viên sau khi đo tương quan hạng (Spearman) với kết quả huấn luyện thật trên dữ liệu của DLNet và so với baseline "đếm tham số". Chưa có phép đo đó thì chỉ ghi thông tin, không loại.
5. Một ước lượng chỉ là **đề xuất**: phải được kiểm chứng bằng huấn luyện cùng ngân sách và chỉ được chấp nhận khi tốt hơn rõ rệt (xem 16.2).

### 16.2 Hyperparameter đi kèm mỗi baseline
Sau khi tìm kiếm, `dlnet/hparams.py` (chạy trong 20% cuối của `--budget`) làm cho từng finalist:
- **LR range test** (Smith, arXiv:1506.01186 §3.3): quét lr tăng theo cấp số nhân, lấy lr lớn nhất còn hội tụ `lr_max`; đề xuất `lr_max/2` (tối ưu thường trong hệ số 2 của lr lớn nhất hội tụ), khoảng `[lr_max/4, lr_max]`.
- **Gradient noise scale** (McCandlish, arXiv:1812.06162, phương trình 2.9 và phụ lục A.1): `B_simple = tr(Σ)/|G|²` từ hai cỡ batch (16 và 256); đề xuất batch là lũy thừa của 2 không vượt `B_simple`, trong [16, 512] và ≤ n_train/4.
- **Normalized weight decay** (Loshchilov & Hutter, arXiv:1711.05101 phụ lục B.1): khi đổi batch `b`, giữ `λ_norm` không đổi, tức `λ ∝ sqrt(b)`.
- **Kiểm chứng**: huấn luyện lại bằng đề xuất với cùng số epoch và seed; chỉ **chấp nhận** nếu validation trung bình hơn bản đã tìm kiếm quá độ lệch chuẩn seed lớn hơn. Ngược lại giữ bản đã tìm và ghi cả hai con số.

Output: `top<i>_*/config.yaml` và `leaderboard.json` có khối `hyperparameters` (giá trị cuối dùng cho `train.py`, kèm `source`: đã tìm/ước lượng/chấp nhận, giá trị `noise scale`, khoảng lr, điểm validation của hai bản); `leaderboard.md` có cột lr, wd, batch.

Chưa làm: epochs (hiện là công thức mặc định 100 epoch, cosine, chọn checkpoint tốt nhất trên validation, chưa ước lượng), Hyperband/BOHB/ASHA (search hiện là một bracket successive halving).

### 16.3 Zero-cost proxy, cache và phép đo tiết kiệm thời gian (M5)
- `dlnet/proxies.py`: `naswot` (Mellor, arXiv:2006.04647, theo `score_networks.py` chính thức: `K += x xᵀ + (1-x)(1-x)ᵀ` trên mỗi ReLU, điểm `log|det K|`), `synflow` (Tanaka, arXiv:2006.05467: tham số lấy trị tuyệt đối, đầu vào toàn 1, điểm `Σ|θ·∂R/∂θ|`), `grad_norm` (Abdelfattah, arXiv:2101.08134), `params` (baseline). Phần "chưa đối chiếu" ghi trong docstring.
- **Phép đo theo quy tắc 16.1 mục 4** chạy trong mọi lần `design`: Spearman của từng proxy với điểm validation thật sau rung 1, ghi vào leaderboard. `used_for_pruning` luôn là `false`.
- **Kết quả đo (`scripts/bench_search.py`, dữ liệu tổng hợp, 2 seed, 16 epoch, so với huấn luyện đầy đủ mọi ứng viên):**

| Dữ liệu | Ứng viên | Thời gian giảm | Cùng top-1 | Spearman: naswot / synflow / grad_norm / params |
|---|---|---|---|---|
| tabular seed 0 | 10 | 50,0% | có | không tính được / −0,59 / 0,51 / 0,43 |
| tabular seed 1 | 10 | 66,5% | có | không tính được / −0,60 / 0,27 / −0,26 |
| HSI seed 0 | 20 | 68,9% | không (0,969 so với 0,994) | 0,14 / 0,13 / −0,12 / 0,33 |
| HSI seed 1 | 20 | 69,7% | không (0,956 so với 0,963) | 0,02 / 0,15 / −0,21 / −0,02 |

- **Kết luận:** không proxy nào hơn đếm tham số một cách ổn định, dấu tương quan còn đổi giữa các seed. Vì vậy **không dùng proxy để loại ứng viên**; muốn bật phải có số đo trên dữ liệu thật trước. Tiết kiệm thời gian hiện đến từ successive halving.
- **Cache** (`search.RunCache`, file `<out>/.dlnet_cache.json`): khóa = phiên bản DLNet + cấu hình + dấu vân tay của lần chia + seed + số epoch. Chỉ áp dụng cho các lần huấn luyện từ đầu (giai đoạn nhiều seed, kiểm chứng hyperparameter).

### 16.4 Ước lượng intrinsic dimension và dung lượng (M6)
- `dlnet/profile.py`: `twonn_intrinsic_dim` (Facco 2017, dạng maximum-likelihood `N/Σ log(r2/r1)`), `mle_intrinsic_dim` (Levina & Bickel 2004 với hiệu chỉnh MacKay & Ghahramani, `k=20`). Cả hai viết từ trí nhớ, **chưa đối chiếu** paper (chưa có trong `ref/`); test dùng dữ liệu biết trước số chiều và một trường hợp tính tay.
- `intrinsic_dim_report`: chạy cả hai trên 5 mẫu con 80%, trả mean ± std, `rel_disagreement` và cờ `stable` (std ≤ 15% mean và hai bộ lệch nhau ≤ 30%; ngưỡng là heuristic). Ghi vào `profile.json` (`intrinsic_dim_estimators`) cho cả plugin tabular và HSI.
- `capacity_hint`: `min_width` = lũy thừa của 2 nhỏ nhất ≥ 2·d, `scaling_exponent` ≈ 4/d (Sharma & Kaplan, arXiv:2004.10802). **Chỉ ghi thông tin**, không loại ứng viên (quy tắc 16.1 mục 5).
- HSI: `multilinear_rank_levels` ghi rank Tucker ở ba mức năng lượng 0,90 / 0,95 / 0,99 để thấy độ nhạy theo ngưỡng.
- **Kết quả đo (`python scripts/bench_estimators.py`):**

| Dữ liệu | d thật | TwoNN | MLE | Lệch nhau | stable |
|---|---|---|---|---|---|
| tuyến tính d=5 trong R^50 | 5 | 4,8 ± 0,1 | 5,0 ± 0,0 | 4% | có |
| tuyến tính d=10 | 10 | 9,3 ± 0,1 | 8,4 ± 0,0 | 10% | có |
| tuyến tính d=20 | 20 | 15,7 ± 0,5 | 12,9 ± 0,1 | 20% | có |
| mặt cầu S^5 trong R^20 | 5 | 5,1 ± 0,1 | 4,8 ± 0,0 | 8% | có |
| d=10 + nhiễu 5% | 10 | 9,9 ± 0,1 | 8,6 ± 0,0 | 14% | có |
| sklearn iris (150×4) | ? | 3,3 ± 0,3 | 2,4 ± 0,1 | 34% | không |
| sklearn wine (178×13) | ? | 7,4 ± 0,5 | 5,6 ± 0,0 | 28% | có |
| sklearn breast_cancer (569×30) | ? | 9,5 ± 0,3 | 7,3 ± 0,1 | 27% | có |
| sklearn digits (1797×64) | ? | 8,8 ± 0,1 | 7,0 ± 0,0 | 22% | có |
| sklearn diabetes (442×10) | ? | 7,1 ± 0,4 | 5,7 ± 0,0 | 21% | có |

- **Kết luận:** ước lượng ổn định qua mẫu con (std ≤ 0,5) nhưng **thiên thấp khi d lớn** (d=20 ra 13–16 với 1000 điểm; đây là thiên lệch đã biết của ước lượng dựa trên láng giềng gần). Trên dữ liệu thật hai bộ lệch nhau 21–34%, MLE luôn thấp hơn. Vì vậy con số chỉ nên đọc là "thấp / vừa / cao", chưa đủ tin để cắt không gian tìm kiếm.

### 16.5 Data card, chọn hạt giống và thuật toán tìm kiếm chuẩn (M15)
- **Data card** (`dlnet/datacard.py`, chỉ trên split train): kích thước, số mẫu/lớp, mất cân bằng; độ dốc phổ công suất và năng lượng tần số cao; độ dài tương quan (tuyệt đối và theo cạnh ảnh); mật độ cạnh; độ phụ thuộc vị trí (phương sai giữa các ảnh trung bình theo lớp); tương quan kênh màu; 1-NN; linear probe trên pixel so với trên đặc trưng tích chập ngẫu nhiên (`conv_bias_gain`). Test đáp án đóng: nhiễu 1/f cho slope ≈ 2, nhiễu trắng slope ≈ 0, phụ thuộc vị trí 1 khi lớp là một pixel cố định.
- **Chọn hạt giống** (`dlnet/seeds.py`): đặc tính kiến trúc đọc từ module trong spec; quy tắc DLNet (data card + ngữ cảnh `--context`, mục tiêu `--goal`) cho điểm prior, in lý do; tổng hợp Borda với meta-learning (kNN của Feurer 2015 + **portfolio tham lam Auto-sklearn 2.0**, arXiv:2007.04074 Algorithm 1) và **vote đa số** của synflow/naswot/snip (Abdelfattah, arXiv:2101.08134 §4) khi bật `--seed-proxies`. `--seeds N` giữ N ứng viên, luôn có một ResNet làm mốc, tối đa 2 mỗi họ.
- **Thuật toán tìm kiếm**: `--scheduler hyperband` (Li et al., arXiv:1603.06560 Algorithm 1; Table 1 của paper lệch với Algorithm 1 ở s = 1..3, DLNet theo Algorithm 1); tiến hóa thành **aging evolution** (Real et al., arXiv:1802.01548 Algorithm 1: tournament S lấy có hoàn lại, con thêm bên phải, già nhất bị loại; chạy theo lô; DLNet thêm lai ghép xác suất 0,3).
- **Proxy mới**: SNIP (eq. 1 của 2101.08134), ZiCo (theo `compute_zico.py` chính thức), có test đáp án đóng.
- **Đo 1: chọn hạt giống, cùng ngân sách** (`scripts/bench_seeds.py`, 120 s, N = 10, pool 104 ứng viên, 2 lần chạy; MNIST và Fashion-MNIST: 3000 ảnh con của tập train gốc; textures, shapes: tổng hợp 1600 ảnh 32×32). Giá trị: trung bình 2 lần của top-1 (val / test):

| Dữ liệu | Tất cả, xếp theo prior | Ngẫu nhiên 10 | Theo dữ liệu 10 | Theo dữ liệu 10 + Hyperband |
|---|---|---|---|---|
| MNIST 3k | 0,930 / 0,916 | 0,944 / 0,933 | 0,928 / 0,908 | **0,955 / 0,945** |
| Fashion 3k | 0,767 / 0,763 | 0,762 / 0,768 | **0,791 / 0,788** | 0,770 / 0,757 |
| textures | 0,991 / 0,985 | 0,991 / 0,978 | 0,991 / 0,985 | 0,989 / 0,985 |
| shapes | 0,991 / 0,983 | 0,989 / 0,986 | 0,991 / 0,983 | 0,978 / 0,988 |

- **Đo 2: prior và proxy có dự đoán thứ hạng không** (`scripts/bench_prior.py`, 20 ứng viên ngẫu nhiên từ cùng pool, huấn luyện 4 epoch, 2 lần lấy mẫu). Spearman với validation (lần 1 / lần 2), trong ngoặc: số lần nằm trong top-3 thật trên 6:

| Dữ liệu | params | synflow | snip | zico | vote | **seed_prior** |
|---|---|---|---|---|---|---|
| MNIST 3k | +0,52 / +0,83 (2/6) | +0,35 / +0,49 | +0,39 / +0,62 | −0,09 / +0,58 | +0,26 / +0,51 | +0,07 / −0,16 (0/6) |
| Fashion 3k | +0,45 / +0,77 (3/6) | +0,34 / +0,36 | +0,37 / +0,41 | −0,04 / +0,47 | +0,23 / +0,34 | −0,15 / −0,18 (0/6) |
| textures | −0,25 / −0,44 (1/6) | −0,06 / +0,07 | −0,23 / +0,06 | −0,41 / −0,04 | −0,14 / +0,08 | −0,24 / +0,35 (0/6) |
| shapes | +0,22 / −0,20 (3/6) | +0,68 / +0,46 | +0,40 / +0,34 | −0,07 / +0,18 | +0,63 / +0,36 | +0,12 / +0,23 (0/6) |

- **Kết luận (trung thực):** (1) Prior theo data card **không dự đoán được** thứ hạng sau 4 epoch (Spearman −0,18…+0,35, không lần nào trúng top-3), nên `--seeds` mặc định 0 (giữ mọi ứng viên, chỉ đổi thứ tự) và chênh lệch ở Đo 1 nằm trong nhiễu của 2 lần chạy. (2) Đếm tham số vẫn là baseline mạnh trên MNIST/Fashion; synflow và vote hơn nó trên shapes nhưng không ổn định qua dữ liệu, nên proxy vẫn **không** dùng để loại (§16.1). (3) Hyperband tốt nhất trên MNIST, kém hơn trên Fashion: chưa đủ bằng chứng để đổi mặc định. (4) Data card mô tả đúng dữ liệu (textures/shapes: phụ thuộc vị trí ≈ 0, lợi ích tích chập +0,5…0,66; MNIST/Fashion: lợi ích ≈ 0), nhưng mapping "đặc trưng → kiến trúc" bằng tay thì chưa đúng: cần học mapping đó từ meta-store khi có nhiều lần chạy thật (`--meta-store`).
- **Phát hiện phụ khi đo**: ảnh 3 kênh bị lưu với stride channels-last; torch 2.15 crash (double free) khi backward đa luồng. Đã sửa (`array_data.to_channels_first` trả mảng liền mạch) và có regression test.

### 16.6 Surrogate zero-shot ZAP và AI advisor (M17, M18)
- **Research**: các cách học ánh xạ đặc trưng dữ liệu → cấu hình tốt: kNN trên meta-feature (Feurer 2015, đã có), portfolio tĩnh (Auto-sklearn 2.0, đã có), phân rã ma trận dataset × cấu hình (PMF, Fusi et al., arXiv:1705.05355: cần cùng tập cấu hình trên mọi dataset, không hợp với pool thay đổi theo dữ liệu của DLNet), **surrogate zero-shot học bằng loss xếp hạng cặp (ZAP, Öztürk et al., arXiv:2206.08476)**: dùng được với cấu hình mô tả bằng đặc trưng, nên được chọn. Task2Vec (arXiv:1902.03545) cho meta-feature tốt hơn nhưng cần mạng probe huấn luyện sẵn: chưa làm.
- **Áp dụng**: `dlnet/surrogate.py` (phương trình 4 của ZAP ở dạng điểm), meta-store ghi kết quả từng ứng viên, `scripts/build_meta.py`, meta-dataset có sẵn `dlnet/data/meta_image.jsonl` (14 bộ: CIFAR-10, CIFAR-10 xám, CIFAR mèo/chó, CIFAR-100 20 lớp, hoa, MNIST 3k/14×14/500/màu, Fashion 3k/14×14, digits, textures, shapes; 24 kiến trúc ngẫu nhiên, 4 epoch).
- **Đo leave-one-dataset-out** (`scripts/bench_surrogate.py`): Borda(ZAP, ZAP không φ, kNN) Spearman +0,67, regret@1 0,061; ZAP +0,54 / 0,061; ZAP không φ +0,69 / 0,075; kNN họ +0,55 / 0,096 (best trong top-3: 9/14); prior quy tắc tay +0,19 / 0,104; đếm tham số +0,33 / 0,108; ngẫu nhiên 0 / 0,175.
- **Kết luận**: học từ lần chạy trước là cách tốt nhất DLNet có để xếp hạng hạt giống, nên bật mặc định cho thứ tự (`--surrogate-weight 1`); `--seeds` vẫn mặc định 0 vì chưa đo việc **loại** ứng viên theo surrogate cùng ngân sách. Data card chưa tạo khác biệt về Spearman với 14 bộ: cần nhiều bộ hơn. Thứ hạng sau 4 epoch là mục tiêu học, chưa phải kết quả sau huấn luyện đầy đủ.
- **AI advisor / MCP**: cài đặt và kiểm thử bằng advisor giả lập và test JSON-RPC; **chưa** chạy với Claude Code, Codex, Antigravity thật, chưa đo hiệu quả.

## 17. Tài liệu người dùng `doc/` và quy tắc cập nhật theo phiên bản

`doc/` là tài liệu cho người dùng và người phát triển; `agent.md` là kế hoạch, quy tắc và số đo chi tiết. Cấu trúc:

| File | Nội dung | Cách cập nhật |
|---|---|---|
| `doc/README.md` | mục lục, dòng **"Phiên bản tài liệu: X.Y.Z"** | dòng phiên bản do `scripts/release.py bump` sửa |
| `doc/user-guide.md` | cài đặt, dữ liệu, chạy, hướng tối ưu, đọc kết quả, huấn luyện lại, sự cố | viết tay |
| `doc/architecture.md` | pipeline, thư mục mã, zoo, plugin, gói đầu ra | viết tay |
| `doc/algorithms.md` | thuật toán, nguồn, chỗ chưa đối chiếu, số đo tóm tắt | viết tay |
| `doc/cli.md`, `doc/zoo.md` | tham chiếu lệnh, danh mục module | **sinh tự động** (`python scripts/gen_docs.py`), không sửa tay |

**Quy tắc (bắt buộc cho mọi thay đổi và mọi bản phát hành):**
0. **Mỗi khi có thay đổi hay nâng cấp, phải cập nhật `README.md` và tài liệu trong `doc/`** (ít nhất một trang viết tay: `user-guide.md`, `architecture.md` hoặc `algorithms.md`; `cli.md`/`zoo.md` sinh tự động không tính). Số liệu trong README (số module, số họ, phiên bản, tính năng) phải khớp mã; `tests/test_docs.py` kiểm số module. Tự động: `python scripts/release.py docs` so với tag gần nhất; mã trong `dlnet/` đổi mà README hoặc `doc/` không đổi thì job `lint` báo cảnh báo trên commit thường và **báo lỗi trên commit phát hành** (phiên bản chưa có tag), nên không phát hành được khi tài liệu chưa cập nhật.
1. Commit nào đổi hành vi người dùng thấy (lệnh, tùy chọn, định dạng dữ liệu, đầu ra, thuật toán, kết quả đo) thì sửa luôn trang `doc/` liên quan trong **cùng commit**, giống quy tắc CHANGELOG. Thêm tùy chọn CLI hoặc op mới: chạy `python scripts/gen_docs.py`.
2. Tài liệu chỉ mô tả điều đã có trong mã và đã đo; điều chưa làm hoặc kết quả âm tính ghi rõ như trong `agent.md`.
3. Commit phát hành dùng `python scripts/release.py bump X.Y.Z`: đổi phiên bản trong `pyproject.toml`, `dlnet/__init__.py`, `doc/README.md`, mở mục CHANGELOG, sinh lại `doc/cli.md`, `doc/zoo.md`.
4. Kiểm tra tự động: `python scripts/release.py check` (phiên bản tài liệu = phiên bản gói) và `tests/test_docs.py` (tài liệu sinh tự động khớp mã, không có link hỏng, mọi hướng tối ưu và lệnh đều được nhắc tới). CI chạy cả hai, nên tài liệu lệch thì không phát hành được.
5. Mỗi mốc mới: thêm mục vào `doc/algorithms.md` (thuật toán + nguồn + số đo) và hướng dẫn dùng vào `doc/user-guide.md`.

## 18. Lộ trình nâng cấp đề xuất (sau audit v0.17)

Ưu tiên theo (lợi ích cho người dùng) / (công sức). P0 = ổn định, nên làm trước; P1 = hiệu quả; P2 = thông minh hơn. Mỗi mục có bằng chứng từ §9.1 hoặc các phép đo trước.

### P0. Ổn định và tái lập
1. ~~**Ngân sách giây phải được tôn trọng**~~ (xong v0.18.0, §9.2): hạn giờ cho từng giai đoạn trước tìm kiếm (data card, baseline, proxy), ghi bảng "ngân sách đã tiêu theo giai đoạn" vào leaderboard, cảnh báo khi vượt > 10%. Bằng chứng: 30 s chạy 35–138 s.
2. ~~**Kiểm soát số luồng**~~ (xong v0.18.0, §9.2) (`--threads`, `threadpoolctl` cho sklearn/OpenMP + `torch.set_num_threads`): tránh tranh luồng (HistGradientBoosting 0,07 s → 11,6 s khi máy bận) và crash đa luồng kiểu torch 2.15 (§16.5).
3. ~~**JSON hợp lệ**~~ (xong v0.18.0): ghi `NaN`/`inf` thành `null` (`allow_nan=False` + làm sạch), có test; hiện `profile.json` có thể chứa `NaN`.
4. **Manifest tái lập** trong `leaderboard.json`: phiên bản DLNet, torch, numpy, số luồng, thiết bị, seed, chế độ ngân sách, dấu vân tay dữ liệu; `--deterministic` bật `torch.use_deterministic_algorithms`.
5. **Tiếp tục sau khi bị ngắt**: nhật ký từng bước (journal) để `design` chạy lại tiếp từ rung đang dở, không chỉ từ cache giai đoạn 2.
6. **Kiểm tra dữ liệu đầu vào sớm**: lớp có < 3 mẫu, mất cân bằng cực đoan, ảnh quá lớn so với RAM, nhãn phân đoạn ngoài dải: báo lỗi rõ trước khi tiêu ngân sách.
7. **Test mạnh hơn**: property-based test (hypothesis) cho ngữ pháp zoo ("mọi spec lấy mẫu đều dựng và lan truyền ngược được"), mutation testing (mutmut) cho `objectives`, `search`, `export.verdict`; nâng sàn coverage dần.
8. **Một nguồn cho mã trong gói xuất ra**: `gan_data.py` sao chép từ `gan_run.py` thay vì viết lại.

### P1. Hiệu quả
1. ~~**GPU và mixed precision**~~ (mã xong v0.19.0, chưa đo trên GPU thật, §9.3) (`--device auto`, AMP bf16/fp16, `channels_last`): hiện DLNet chỉ chạy CPU; đây là tăng tốc lớn nhất có thể (thường 10–50 lần với CNN).
2. **Huấn luyện song song nhiều ứng viên** (nhiều tiến trình / nhiều GPU) với **ASHA** (arXiv:1810.05934, đã có trong `ref/`): successive halving bất đồng bộ không phải chờ cả rung.
3. **Đa độ chính xác theo dữ liệu**: rung đầu dùng tập con dữ liệu / ảnh thu nhỏ (Hyperband với tài nguyên là số mẫu), chỉ ứng viên tốt mới thấy toàn bộ dữ liệu.
4. **Dừng sớm theo đường học** (ngoại suy learning curve, freeze-thaw BO, arXiv:1406.3896 trong `ref/`): bỏ ứng viên giữa rung khi đường cong không thể vượt ngưỡng.
5. **Khởi tạo con từ cha** trong tiến hóa (network morphism / Net2Net): con kế thừa trọng số, không huấn luyện từ đầu.
6. **Baseline cổ điển chạy song song** với tìm kiếm và có giới hạn cỡ dữ liệu (HistGradientBoosting trên dữ liệu lớn).
7. **Gói xuất ra nhẹ hơn**: xuất ONNX/TorchScript, gộp RepVGG khi xuất, lượng tử hóa cho `--objective params|flops`.

### P2. Thông minh hơn
1. **Meta-dataset lớn hơn và đúng mục tiêu hơn**: thêm bộ dữ liệu, nhãn là kết quả sau huấn luyện đầy đủ thay vì 4 epoch; chỉ khi đó mới đo việc **loại** ứng viên theo surrogate cùng ngân sách (§16.6).
2. **Meta-feature tốt hơn**: Task2Vec (arXiv:1902.03545) hoặc embedding từ mạng probe nhỏ thay cho data card viết tay.
3. **Đề xuất cấu hình có mô hình** (BOHB, arXiv:1807.01774; TPE): sinh kiến trúc từ phân phối học được thay vì lấy mẫu ngẫu nhiên trong ngữ pháp.
4. **Đo AI advisor** trên dữ liệu thật, nhiều seed, cùng ngân sách epoch; chỉ tăng trọng số khi thắng rõ.

### 18.1 Thứ tự mới sau kiểm định chất lượng v0.19 (thay thế thứ tự ở trên)
Đội 4 agent kiểm định (`qa/`, skill `quality-audit`) chấm 34 tiêu chí: 6 ĐẠT, 17 MỘT PHẦN, 11 KHÔNG ĐẠT. Chi tiết, bằng chứng và nguồn: `qa/reports/v0.19.0/SUMMARY.md`.
- ~~**Mốc 1 (P0, sửa kết quả sai)**~~ (xong v0.20.0, §9.4): Hyperband sập khi chạm đúng hạn; cô lập ứng viên lỗi hoặc NaN; chỉ xếp hạng cùng số epoch, có cột epoch; nhiễu trung thực khi chỉ có 1 lần chạy; sửa `best-small` và `most-stable`; verdict có ngưỡng nhiễu trên test; cảnh báo one-hot bất thường và HSI chia pixel.
- ~~**Mốc 2 (P1, ngân sách)**~~ (xong v0.21.0, §9.5): pool ứng viên theo ngân sách (finalist ≥ rung 2–3); giai đoạn hyperparameter tắt ở ngân sách nhỏ và tính cả giờ thực; bỏ prior tay khỏi Borda; tiến hóa và Hyperband chỉ bật khi đủ ngân sách, về lâu dài dùng ASHA.
- ~~**Mốc 3 (P1, mạng đầu ra)**~~ (xong v0.22.0, §9.6; còn EMA, mixup): công thức huấn luyện hiện đại (augmentation, label smoothing, warmup, EMA, mixup; ablation cho thấy tới 17 điểm trên CIFAR); gói xuất ra có `model.pt`, `preprocess.json`, `predict.py`, ONNX; thông báo lỗi và kiểm tra đầu vào.
- **Mốc 4 (P2, cutting edge)**: pretrained (timm/DINOv2, TabPFN v2); TabM/PLR, PatchTST/iTransformer, Mamba-HSI, graph transformer; DEHB/PriorBand/LC-PFN; đo trên GPU.
Mỗi mốc kết thúc bằng một lần chạy lại đội kiểm định.

### 18.2 Lộ trình v1.5 – v1.7 (người dùng đã duyệt)
- **v1.5**: MCP server qua HTTP (`dlnet serve`, xong); nghiên cứu chọn hạt giống / xếp hạng vòng loại (TSE, đang đo đầu-cuối; chỉ đổi mặc định khi top-1 tốt hơn thật).
- **v1.6**: gói cài Windows / macOS / Linux; mỗi tag phát hành bắt buộc có đủ gói cài (máy ép: `release.yml` → `installers.yml` → `release.py assets`).
- **v1.7** (giao diện đã chốt: Artifact "DLNet UI tối giản", 6 màn: workspace, lần chạy, tệp, chạy mới, một lần chạy, AI/MCP):
  - CLI vẫn là chính; giao diện là tùy chọn, chỉ đọc thư mục workspace trên đĩa và chạy đúng lệnh CLI (hiện lệnh tương ứng trên mọi màn).
  - **Bắt buộc hai ngôn ngữ EN / VI, đồng bộ 100%**: một bộ khóa chung, test thất bại nếu một khóa thiếu ở một ngôn ngữ, chuỗi rỗng, hoặc chuỗi hiển thị nằm cứng trong HTML/JS ngoài bộ dịch.
  - **Chế độ sáng / tối**: theo hệ điều hành mặc định, có nút chuyển, nhớ lựa chọn; mọi màu là token, kiểm tương phản ≥ 4.5:1 ở cả hai chế độ.
  - So sánh 10–20 paper trong một lệnh: `dlnet plan` (ước lượng trước: tham số, FLOPs, VRAM, s/epoch đo thật, tổng thời gian, mức A/B/C, số paper báo, dự báo sớm thô), chạy chắc (lỗi một baseline không dừng cả lần chạy, tiếp tục được, nhiều GPU), kiểm định ý nghĩa thống kê, xuất bảng LaTeX.
  - Thứ tự: phần CLI trước, giao diện sau.

## Tài liệu tham khảo
Toàn bộ PDF nằm trong `ref/` (mục lục: `ref/README.md`); thuật toán ước lượng chất lượng và hyperparameter trong `ref/optimazeDL/` (§16).

- Pfeiffer, Ruder, Vulić, Ponti. *Modular Deep Learning.* arXiv:2302.11529.
- Hu et al. *LoRA*; Houlsby et al. *Adapters*; Li & Liang *Prefix-tuning*; Lester et al. *Prompt tuning*.
- Gregor & LeCun *LISTA*; Monga et al. *Algorithm Unrolling*.
- Kolda & Bader *Tensor Decompositions and Applications*; Oseledets *Tensor-Train*.
- Öztürk et al. *Zero-Shot AutoML with Pretrained Models* (ZAP, arXiv:2206.08476); Fusi et al. *Probabilistic Matrix Factorization for AutoML* (arXiv:1705.05355); Tan et al. *MnasNet* (arXiv:1807.11626); Elsken et al. *LEMONADE* (arXiv:1804.09081).
- Real et al. *Regularized Evolution* (arXiv:1802.01548); Feurer et al. *Auto-Sklearn 2.0* (arXiv:2007.04074); Öztürk et al. *Zero-Shot AutoML with Pretrained Models* (arXiv:2206.08476); Achille et al. *Task2Vec* (arXiv:1902.03545, chưa dùng).
