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
| **M2** | **Plugin HSI**: loader, split chống rò rỉ, preset 1D/2D/3D-CNN + HybridSN, metric OA/AA/Kappa | Top-K trên Indian Pines, so được bộ tham chiếu |
| **M3** | Giao thức đánh giá đầy đủ (§12.2–12.3): nhiều seed, nhóm "chưa phân biệt được", quy tắc chấp nhận | Báo cáo tái lập, mỗi số có ± std |
| **M4** | Library mở rộng + tự ghép theo grammar (§13): SSRN, Transformer, Mamba/SSM, ResNet/ConvNeXt block | Sinh được ≥20 ứng viên hợp lệ từ registry |
| **M5** | Cost model + tiết kiệm thời gian: zero-cost proxy, successive halving, cache, resume | Giảm ≥X% thời gian, cùng chất lượng top-1 |
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

---

## 10. Quy ước cho agent (người/AI) phát triển repo

- Mọi thay đổi kiến trúc phải cập nhật `agent.md` và `ArchSpec` schema.
- Mỗi module mới: có test shape + forward/backward + docstring nêu *trục nào của paper* nó hiện thực.
- Không thêm phụ thuộc nặng ngoài danh sách §6 nếu chưa thống nhất.
- Kết quả thực nghiệm lưu kèm seed/config để tái lập.
- Không commit dữ liệu thô hoặc khóa/bí mật.

## 11. Quy trình Git & CI/CD

**Chỉ có đúng hai nhánh, không tạo thêm:**
- `dev`: nơi duy nhất để phát triển; commit và push trực tiếp vào `dev`.
- `main`: luôn ổn định, chỉ nhận code từ `dev` (tự động, sau khi CI xanh); không ai push trực tiếp.

```
commit ──► dev ──(CI xanh, tự động)──► main
```

Workflow (`.github/workflows/`):
- `ci.yml`: chạy khi push vào `dev`/`main` và khi có PR vào `main`: lint (ruff), test (pytest, Python 3.10–3.12); PR vào `main` mà nguồn không phải `dev` bị từ chối.
- `promote.yml`: khi push vào `dev` và CI xanh → tự tạo PR `dev → main` và merge.
Thiết lập thủ công trên GitHub (không làm được từ code): xem `.github/README.md`.

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
2. Zero-cost proxy → giữ ~30.
3. Short-train (Successive Halving) → giữ ~8–10.
4. Huấn luyện vừa, nhiều seed → xếp hạng cuối theo trung bình ± std.

**Chọn Top-K**:
- Xếp theo metric trung bình; ứng viên có chênh lệch nhỏ hơn std được gom thành **nhóm "chưa phân biệt được"** và ghi rõ trong leaderboard.
- **Đa dạng**: không để K slot bị chiếm bởi các biến thể gần giống nhau; cố gắng phủ nhiều họ kiến trúc.
- Gắn nhãn vai trò: `top1` (chính xác nhất), `best-small` (hiệu quả/độ chính xác), `most-stable` (std thấp nhất), `most-different` (họ khác, đáng thử).

**Mỗi baseline trong Top-K kèm**: điểm ± std, #params/FLOPs/VRAM, thời gian train đã dùng, **công thức huấn luyện đầy đủ gợi ý** (epochs, lr schedule, augmentation), và lưu ý rủi ro.

**Chính trực**: nếu không ứng viên nào vượt bộ baseline tham chiếu (§12.2), `leaderboard` ghi rõ và đưa baseline tham chiếu tốt nhất lên đầu.

**CLI**:
```bash
dlnet design --data data.csv --task auto --budget 2h --topk 5
```

## 13. Module Library & tự động ghép baseline

### 13.1 Registry
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
Mỗi mô hình kinh điển được viết lại như **một cấu hình của registry** (không phải code riêng), ví dụ: 1D-CNN, 2D-CNN, 3D-CNN, HybridSN, SSRN, SpectralFormer, SSFTT, các biến thể Mamba cho HSI. Lợi ích: vừa là bộ baseline tham chiếu (§12.2), vừa là điểm xuất phát của tìm kiếm. Danh sách này là tham chiếu; **phải kiểm tra lại nguồn gốc và tái hiện số liệu khi cài đặt**, chưa có trong `ref/`.

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

## Tài liệu tham khảo
Toàn bộ PDF nằm trong `ref/` (mục lục: `ref/README.md`).

- Pfeiffer, Ruder, Vulić, Ponti. *Modular Deep Learning.* arXiv:2302.11529.
- Hu et al. *LoRA*; Houlsby et al. *Adapters*; Li & Liang *Prefix-tuning*; Lester et al. *Prompt tuning*.
- Gregor & LeCun *LISTA*; Monga et al. *Algorithm Unrolling*.
- Kolda & Bader *Tensor Decompositions and Applications*; Oseledets *Tensor-Train*.
