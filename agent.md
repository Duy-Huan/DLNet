# agent.md — Kiến trúc dự kiến của DLNet

> Framework **tự động thiết kế mạng deep learning**: đầu vào là **dữ liệu**, đầu ra là **một baseline PyTorch đóng gói sẵn** (code + cấu hình + checkpoint khởi tạo + báo cáo giải thích lựa chọn).
> Cơ sở lý thuyết: *Modular Deep Learning* (Pfeiffer, Ruder, Vulić, Ponti — arXiv:2302.11529).

---

## 1. Mục tiêu & phạm vi

| | Nội dung |
|---|---|
| **Input** | Dataset (tabular / chuỗi thời gian / ảnh / văn bản / tín hiệu / tensor đa chiều), nhãn (nếu có), loại tác vụ, ràng buộc tài nguyên (tham số tối đa, FLOPs, VRAM, latency). |
| **Output** | Gói baseline PyTorch: `model.py`, `config.yaml`, `train.py`, `report.md`, tùy chọn pre-trained/khởi tạo bằng phân rã. |
| **Cách làm** | (1) Phân tích dữ liệu → (2) ước lượng "độ phức tạp" & cấu trúc → (3) chọn họ kiến trúc + module → (4) tìm kiếm cấu hình trong không gian có ràng buộc → (5) đóng gói. |
| **Ngoài phạm vi (v1)** | Tự thu thập dữ liệu; NAS toàn bộ kiểu từ-đầu quy mô lớn; huấn luyện LLM. |

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
 Data ──► [1 Ingest & Profile] ──► DataProfile
                                        │
                                        ▼
                         [2 Estimator]  (ước lượng độ phức tạp & cấu trúc)
              ┌─────────────┬──────────────┬──────────────────┐
              ▼             ▼              ▼                  ▼
        Capacity est.  Tensor-structure  Inverse-problem   Modularity est.
        (scaling/MDL)  (rank, unfolding) (có model vật lý?) (domain/task/cluster)
              └─────────────┴──────────────┴──────────────────┘
                                        │  ProblemSpec
                                        ▼
                  [3 Architecture Recommender]  (rule prior + learned meta-model)
                                        │  Top-K ArchSpec (4 trục)
                                        ▼
                  [4 Search & Refine]  (BO / evolutionary / zero-cost proxy → short-train)
                                        │  Best ArchSpec + Pareto
                                        ▼
                  [5 Compiler]  ArchSpec ─► nn.Module PyTorch
                                        │
                                        ▼
                  [6 Packager]  baseline/ + report + tests + (tuỳ chọn) khởi tạo từ phân rã
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
- Trả về Top-K `ArchSpec` kèm lý do.

### 3.4 Search & Refine (`dlnet/search`)
- Không gian **có cấu trúc, nhỏ** (4 trục × siêu tham số) thay vì NAS vô hạn.
- Thuật toán: Bayesian Optimization / Successive Halving / evolutionary; đánh giá bằng zero-cost proxy rồi short-train.
- Đa mục tiêu: accuracy, #params, FLOPs, latency → Pareto front; luôn tôn trọng ràng buộc cứng.
- Mọi lần chạy ghi seed, config, metric (tái lập).

### 3.5 Compiler (`dlnet/compile`)
`ArchSpec` (YAML/dataclass) ⟶ `torch.nn.Module` qua registry. Mỗi module đăng ký `Computation/Router/Aggregator`. Có kiểm tra shape tĩnh và `torch.fx`/`torch.compile` smoke test.

### 3.6 Packager (`dlnet/package`)
Xuất thư mục baseline độc lập, chỉ phụ thuộc PyTorch:
```
baseline_<name>/
 ├─ model.py        # nn.Module đã sinh, đọc được, có docstring
 ├─ config.yaml     # ArchSpec + hyperparams + seed
 ├─ train.py        # vòng huấn luyện tối giản (+ eval)
 ├─ init.pt         # (tuỳ chọn) khởi tạo từ phân rã tensor / pretrained
 ├─ report.md       # DataProfile, ước lượng, vì sao chọn, Pareto, rủi ro
 └─ tests/          # test shape, forward/backward, tái lập
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

| Mốc | Nội dung | Tiêu chí hoàn thành |
|---|---|---|
| **M0** | Khung repo, `ArchSpec/ProblemSpec`, compiler cho MLP/CNN | Sinh được model từ YAML, test pass |
| **M1** | Ingest/Profile + Estimator cơ bản (intrinsic-dim, learning-curve, SVD) | `ProblemSpec` ổn định trên ≥5 dataset |
| **M2** | ModuleZoo: LoRA, adapter, prefix, sparse + Router/Aggregator | Tái hiện số liệu PEFT cơ bản |
| **M3** | Tensor module (CP/Tucker/TT) + ước lượng rank | Nén ≥X% tham số, mất ≤Y% độ chính xác |
| **M4** | Deep unfolding (ISTA/ADMM) | Vượt thuật toán lặp gốc ở cùng số vòng |
| **M5** | Recommender (rule) + Search đa mục tiêu | Top-K ≥ baseline thủ công trên benchmark |
| **M6** | Packager + report + CLI end-to-end | `dlnet design` ra baseline chạy được từ data thô |
| **M7** | Meta-learned recommender, continual/hypernetwork | Cải thiện so với rule prior |

---

## 9. Rủi ro & quyết định cần chốt

1. **Phạm vi modality** cho v1 (đề xuất: tabular + chuỗi thời gian + ảnh nhỏ; mở rộng sau).
2. **Mức tự động**: chỉ *đề xuất* (human-in-the-loop) hay tự huấn luyện hoàn chỉnh? Đề xuất: đề xuất + short-train, người dùng duyệt.
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

## Tài liệu tham khảo
Toàn bộ PDF nằm trong `ref/` (mục lục: `ref/README.md`).

- Pfeiffer, Ruder, Vulić, Ponti. *Modular Deep Learning.* arXiv:2302.11529.
- Hu et al. *LoRA*; Houlsby et al. *Adapters*; Li & Liang *Prefix-tuning*; Lester et al. *Prompt tuning*.
- Gregor & LeCun *LISTA*; Monga et al. *Algorithm Unrolling*.
- Kolda & Bader *Tensor Decompositions and Applications*; Oseledets *Tensor-Train*.
