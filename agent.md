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

---

## 5. Deep Unfolding (khi cần)

Điều kiện kích hoạt: `has_forward_model = true` (bài toán nghịch/tối ưu lặp có cấu trúc).
- Chọn thuật toán lặp nền: ISTA/FISTA, ADMM, Proximal Gradient, AMP, Unrolled Gradient Descent, (với tensor) **iterative tensor completion/ALS**.
- Mỗi vòng lặp → một **tầng**; tham số (bước, ngưỡng, toán tử) trở thành **học được**, có thể chia sẻ hoặc riêng từng tầng.
- Số tầng `K` do Search chọn theo ngân sách tính toán.
- Trong khung paper: unfolding là **function composition tuần tự** (sequential aggregation); các tham số học được có thể sinh bằng hypernetwork.
- Lợi ích: ít tham số, diễn giải được, hội tụ nhanh; kèm kiểm tra so với thuật toán lặp gốc làm baseline.

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
| **M6** | Estimator: intrinsic dim → dung lượng, rank tensor | Ước lượng ổn định trên nhiều dataset |
| **M7** | Tensor: Tucker/HOSVD tiền xử lý, lớp TT/Tucker/TR, Tensor-LoRA | Nén ≥X% tham số, mất ≤Y% metric |
| **M8** | Deep unfolding (khi có mô hình thuận) | Vượt thuật toán lặp gốc ở cùng số vòng |
| **M9** | Plugin ảnh, chuỗi thời gian; recommender học từ kho kết quả (meta-learning) | Top-K chứa mô hình tốt nhất trong ≥80% dataset giữ lại |

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

## 10. Quy ước cho agent (người/AI) phát triển repo

- Mọi thay đổi kiến trúc phải cập nhật `agent.md` và `ArchSpec` schema.
- Mỗi module mới: có test shape + forward/backward + docstring nêu *trục nào của paper* nó hiện thực.
- Không thêm phụ thuộc nặng ngoài danh sách §6 nếu chưa thống nhất.
- Kết quả thực nghiệm lưu kèm seed/config để tái lập.
- Không commit dữ liệu thô hoặc khóa/bí mật.
- **Môi trường Python:** không commit `.venv` (nặng hàng GB, riêng từng máy, vượt giới hạn 100 MB/file của GitHub; `tests/test_env_setup.py` chặn). Môi trường tái lập bằng `scripts/setup_env.sh` / `scripts/setup_env.ps1` (hoặc `requirements*.txt`), kiểm bằng `dlnet selfcheck`; job CI `env-setup` chạy đúng quy trình này trên Ubuntu, macOS, Windows. Thêm phụ thuộc mới: sửa `pyproject.toml` (extras `design`/`dev`), không sửa riêng script.
- Trước khi làm, chọn skill phù hợp trong `skills/` (§15).
- **Thuật toán chuẩn:** mọi bộ ước lượng chất lượng mạng hoặc hyperparameter phải tham khảo `ref/optimazeDL/` và implement đúng công thức của paper/mã chính thức (§16); không tự chế.
- Mỗi thay đổi người dùng thấy được phải thêm một dòng vào `CHANGELOG.md` mục `[Unreleased]`; phát hành theo §11.2.
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
                 └─ chưa → kiểm tra version·CHANGELOG, build wheel, thử cài,
                           tạo tag vX.Y.Z và GitHub Release (ghi chú lấy từ CHANGELOG)
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
| `image` | ảnh RGB/xám | ResNet, ConvNeXt, ViT nhỏ | M9 |
| `timeseries` | chuỗi thời gian | TCN, Transformer, SSM/Mamba | M9 |
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

## Tài liệu tham khảo
Toàn bộ PDF nằm trong `ref/` (mục lục: `ref/README.md`); thuật toán ước lượng chất lượng và hyperparameter trong `ref/optimazeDL/` (§16).

- Pfeiffer, Ruder, Vulić, Ponti. *Modular Deep Learning.* arXiv:2302.11529.
- Hu et al. *LoRA*; Houlsby et al. *Adapters*; Li & Liang *Prefix-tuning*; Lester et al. *Prompt tuning*.
- Gregor & LeCun *LISTA*; Monga et al. *Algorithm Unrolling*.
- Kolda & Bader *Tensor Decompositions and Applications*; Oseledets *Tensor-Train*.
