# Hướng dẫn sử dụng DLNet

Mục lục: [1. Cài đặt](#1-cài-đặt) · [2. Chạy lần đầu](#2-chạy-lần-đầu) · [3. Chuẩn bị dữ liệu](#3-chuẩn-bị-dữ-liệu) · [4. Chạy theo loại bài toán](#4-chạy-theo-loại-bài-toán) · [5. Chọn hướng tối ưu](#5-chọn-hướng-tối-ưu) · [6. Ngữ cảnh và chọn hạt giống](#6-ngữ-cảnh-và-chọn-hạt-giống) · [7. Ngân sách và giới hạn chi phí](#7-ngân-sách-và-giới-hạn-chi-phí) · [8. Đọc kết quả](#8-đọc-kết-quả) · [9. Huấn luyện lại một baseline](#9-huấn-luyện-lại-một-baseline) · [10. Học từ các lần chạy trước](#10-học-từ-các-lần-chạy-trước) · [11. Lệnh khác](#11-lệnh-khác) · [12. Xử lý sự cố](#12-xử-lý-sự-cố)

Mọi tùy chọn có trong [cli.md](cli.md).

## 1. Cài đặt

Cần Python 3.10–3.13 (CI kiểm tra 3.10, 3.11, 3.12 trên Linux, macOS, Windows).

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
# Máy KHÔNG có GPU NVIDIA: cài torch bản CPU TRƯỚC (nếu không, bước sau tải bản CUDA nhiều GB)
pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt      # numpy, pandas, scipy, scikit-learn, threadpoolctl, torch (ghim)
python -m dlnet selfcheck            # khoảng 20 giây; in "selfcheck OK"
```

Script `scripts/setup_env.sh` / `setup_env.ps1` làm cả các bước trên.

Máy có GPU NVIDIA: cài torch bản CUDA theo hướng dẫn trên pytorch.org (ví dụ `--index-url https://download.pytorch.org/whl/cu128`), kiểm tra bằng `python -c "import torch; print(torch.cuda.is_available())"`. DLNet tự dùng GPU khi torch thấy nó (mục 7).

## 2. Chạy lần đầu

```bash
python -m dlnet design --data my_data.csv --target label --budget 5m --topk 3
```

DLNet tự nhận loại dữ liệu, in từng bước (`[plugin]`, `[profile]`, `[datacard]`, `[reference]`, `[seeds]`, `[search]`, `[evolve]`, `[hparams]`, `[done]`, `[verdict]`) và ghi kết quả vào `dlnet_out/`. Mở `dlnet_out/leaderboard.md` trước tiên.

## 3. Chuẩn bị dữ liệu

| Bài toán | File | Nội dung |
|---|---|---|
| Bảng | `.csv` | một cột mục tiêu (`--target`); cột chữ được mã hóa tự động |
| Ảnh phân loại / hồi quy | `.npz` | `X` (N, H, W) hoặc (N, H, W, C); `y` (N,) |
| Phân đoạn ảnh | `.npz` | `X` như trên; `y` (N, H, W) số nguyên, `-100` hoặc giá trị âm = bỏ qua |
| Autoencoder / VAE | `.npz` | chỉ `X` (không có `y`) |
| Chuỗi thời gian | `.npz` | `X` (N, T) hoặc (N, T, C); `y` (N,) |
| Siêu phổ (HSI) | `.mat`, `.npz`, `.npy` | cube (H, W, bands) và ground truth (H, W), `0` = không nhãn |
| Đồ thị | `.npz` | `X` (G, N, F) đặc trưng nút, `A` (G, N, N) kề, `y` (G,) |
| Recommender | `.npz` | `ids` (N, F) số nguyên (ví dụ user, item), `y` (N,) điểm hoặc nhấp |

Tên mảng khác thì dùng `--x-key`, `--y-key` (ảnh, chuỗi), `--cube-key`, `--gt-key` (HSI). Muốn chỉ định plugin thay vì để DLNet đoán: `--plugin image_segmentation`.

Ví dụ tạo file ảnh:

```python
import numpy as np

np.savez("images.npz", X=images.astype("float32"), y=labels.astype("int64"))  # images: (N, H, W, C)
```

## 4. Chạy theo loại bài toán

```bash
# ảnh phân loại; mask phân đoạn; autoencoder
python -m dlnet design --data images.npz --budget 10m
python -m dlnet design --data masks.npz --budget 15m
python -m dlnet design --data images_only.npz --plugin image_reconstruction --budget 10m

# chuỗi thời gian, đồ thị, recommender
python -m dlnet design --data series.npz --budget 10m
python -m dlnet design --data graphs.npz --budget 10m
python -m dlnet design --data interactions.npz --budget 10m

# siêu phổ: chia theo khối để không rò rỉ không gian
python -m dlnet design --data cube.mat --gt gt.mat --split block --patch 9 --pca 30 --budget 10m
```

**Họ miền tần số cho ảnh siêu phổ** (FFT → phép học trên phổ tần số → iFFT, `dlnet/zoo/ops_freq.py`): `gfnet2d` (GFNet, lọc toàn cục 2-D), `afno2d` (AFNO), `fno2d` (Fourier Neural Operator), `ffc2d` (Fast Fourier Convolution), `fcanet2d` (chú ý kênh bằng DCT), `spec_fft2d` (hệ số Fourier của phổ từng pixel rồi ResNet), `fourier_ss` (lọc Fourier dọc trục phổ rồi GFNet không gian). Các họ này nằm sẵn trong không gian tìm kiếm HSI; ngữ pháp (`--zoo`) cũng tự chèn các op `gfnet_block`, `fno_block`, `afno_block`, `ffc_block`, `fca`, `spectral_gf` vào kiến trúc lấy mẫu. Hai họ xử lý **trục phổ** (`spec_fft2d`, `fourier_ss`) chỉ có ý nghĩa vật lý khi giữ nguyên thứ tự băng: dùng `--pca 0` (với PCA, trục kênh là các thành phần chính, không phải bước sóng).

```bash
python -m dlnet design --data cube.mat --gt gt.mat --split block --patch 9 --pca 0 --budget 20m
```

**Họ mới của v1.0** (viết theo mã chính thức, so số học với mã gốc; danh sách nguồn ở `THIRD_PARTY.md`):

| Dữ liệu | Họ mới | Ghi chú |
|---|---|---|
| Ảnh siêu phổ | `mambahsi`, `mambahsi_plus`, `ssftt`, `fahm` | Mamba / Transformer cho HSI; các họ Mamba chạy chậm trên CPU (quét tuần tự) |
| Bảng | `realmlp` | RealMLP-TD (NeurIPS 2024) |
| Chuỗi thời gian | `timemixer`, `timexer`, `xlstm`, `gated_deltanet` | `xlstm` / `gated_deltanet` trộn được thông tin giữa các kênh |
| Ảnh | `mambavision`, `kat` | lai Mamba–attention; ViT với GR-KAN |
| Đồ thị | `gps_net`, `polynormer_net`, `sgformer_net` | graph transformer (đồ thị nhỏ, ma trận kề dày) |
| Recommender | `finalmlp`, `dcnv3`, `wukong` | mô hình CTR |

Ngữ pháp `--zoo` còn tự dùng các khối mới như `gated_attn_block`, `diff_attn_block`, `dyt`, `rope_vit_block`, `wtconv`, `uib`, `mobile_mqa`, `lsnet_block`, `star_block`, `efficientvit_block`, `msca_block`, `gr_kan`. Xem danh sách bằng `python -m dlnet zoo`.

Gợi ý ngân sách trên CPU: 5–15 phút cho vài nghìn mẫu nhỏ; tăng `--budget` khi dữ liệu lớn hơn. Kết quả trong ngân sách ngắn là **xếp hạng sơ bộ**: các hạng liền nhau có thể đổi chỗ khi huấn luyện đầy đủ (cột `group` trong leaderboard cho biết những mô hình không phân biệt được trong nhiễu).

## 5. Chọn hướng tối ưu

`--objective` quyết định "tốt nhất" nghĩa là gì, và được áp dụng ở mọi bước: loại ứng viên giữa các rung, chọn cha mẹ trong tiến hóa, xếp Top-K.

| Lệnh | Khi nào dùng | Cách xếp hạng |
|---|---|---|
| `--objective accuracy` (mặc định) | cần độ chính xác cao nhất | điểm validation |
| `--objective params --tolerance 0.01` | cần mô hình nhỏ nhất mà gần như không mất độ chính xác | mô hình ít tham số nhất trong số các mô hình kém mô hình tốt nhất không quá `tolerance` (0,01 = 1 điểm phần trăm accuracy) |
| `--objective flops` | cần suy luận nhanh | như `params` nhưng theo FLOPs mỗi mẫu |
| `--objective balanced [--target-cost 200k] [--cost-weight -0.07] [--cost params\|flops]` | cân bằng hai mục tiêu bằng một con số | phần thưởng MnasNet: `acc × (cost / T)^w`; T mặc định là trung vị chi phí của các ứng viên; w âm hơn = ưu tiên mô hình nhỏ mạnh hơn |
| `--objective pareto [--cost params\|flops]` | muốn xem toàn bộ đánh đổi | mô hình không bị trội (không mô hình nào vừa tốt hơn vừa nhỏ hơn) xếp trước; trong cùng mặt Pareto, mô hình ở vùng thưa xếp trước để Top-K trải đều |

Ví dụ:

```bash
# nhỏ nhất trong vòng 0,5 điểm phần trăm của mô hình tốt nhất, không quá 300 nghìn tham số
python -m dlnet design --data images.npz --objective params --tolerance 0.005 --max-params 300k

# cân bằng với mục tiêu 100 nghìn tham số
python -m dlnet design --data images.npz --objective balanced --target-cost 100k

# cả mặt Pareto (accuracy, FLOPs)
python -m dlnet design --data images.npz --objective pareto --cost flops --topk 5
```

Dù chọn hướng nào, `leaderboard.md` luôn có bảng **Pareto front** (validation, tham số) để bạn tự chọn điểm đánh đổi.

## 6. Ngữ cảnh và chọn hạt giống

Với ảnh, DLNet tính **data card** (in ở `[datacard]`): phổ tần số, độ lớn cấu trúc, mức phụ thuộc vị trí, tương quan kênh màu, độ khó, và phép thử xem đặc trưng tích chập có giúp hơn pixel thô. Kết hợp với mô tả bài toán của bạn, DLNet xếp thứ tự các kiến trúc hạt giống:

```bash
python -m dlnet design --data scans.npz --context "medical CT, small lesions"
python -m dlnet design --data frames.npz --context "real-time on mobile camera" --objective flops
python -m dlnet design --data fabric.npz --context "texture defect inspection"
```

Từ khóa hiểu được (tiếng Anh và tiếng Việt): y tế / medical / CT / MRI, vệ tinh / satellite / aerial, mobile / edge / real-time / nhanh / nhẹ, texture / defect / bề mặt, document / chữ / digit, small object / vật nhỏ, natural / photo. Dòng `[seeds]` in từng quy tắc đã áp dụng và lý do.

- `--seeds N`: chỉ giữ N hạt giống tốt nhất theo xếp hạng (mặc định 0 = giữ tất cả, chỉ xếp thứ tự). **Lưu ý trung thực:** trên MNIST, Fashion-MNIST và hai bộ tổng hợp, xếp hạng theo data card chưa dự đoán được thứ hạng sau huấn luyện (`agent.md` §16.5), nên đừng đặt `--seeds` quá nhỏ.
- `--seed-proxies`: thêm phiếu bầu của zero-cost proxy vào xếp hạng hạt giống.
- `--scheduler hyperband`: thay một bracket successive halving bằng Hyperband (nhiều mức cắt sớm).

### Học ánh xạ dữ liệu → hạt giống từ các lần chạy (surrogate ZAP)

Với ảnh phân loại, DLNet có sẵn một meta-dataset (14 bộ ảnh × 24 kiến trúc) và các lần chạy của bạn với `--meta-store` được cộng thêm vào (mỗi lần `design` tự ghi kết quả từng ứng viên). Từ đó DLNet huấn luyện một **surrogate zero-shot** (thuật toán ZAP, xem [algorithms.md](algorithms.md#3-chọn-hạt-giống-và-học-từ-lần-chạy-trước)) dự đoán kiến trúc nào hợp với data card của bộ dữ liệu mới, và đưa nó vào xếp hạng hạt giống (`--surrogate-weight`, mặc định 1; `0` để tắt; `--no-builtin-meta` để chỉ dùng dữ liệu của bạn). Đo leave-one-dataset-out: Spearman +0,67 so với +0,19 của quy tắc tay. Dựng sẵn một meta-dataset từ nhiều bộ ảnh:

```bash
python scripts/build_meta.py --data a.npz b.npz c.npz --n 24 --epochs 4 --out meta_image.jsonl
python -m dlnet design --data new.npz --meta-store meta_image.jsonl
```

### AI tham gia chọn hạt giống (tùy chọn)

`--advisor claude | codex | gemini` hoặc `--advisor-cmd "..."`: một AI CLI xếp hạng hạt giống và đề xuất kiến trúc mới; DLNet kiểm tra mọi đề xuất trước khi dùng. Hướng dẫn đầy đủ, kèm MCP server `dlnet mcp` cho Claude Code / Codex / Antigravity: [ai-advisor.md](ai-advisor.md).

`--extra-specs my_specs.json`: thêm kiến trúc do bạn (hoặc AI) viết dạng spec zoo vào pool.

## 7. Ngân sách và giới hạn chi phí

| Tùy chọn | Ý nghĩa |
|---|---|
| `--budget 10m` | tổng thời gian (giây/phút/giờ); luôn có ít nhất một kết quả kể cả khi hết giờ sớm. Kết quả phụ thuộc tốc độ máy |
| `--budget 400ep` | ngân sách **theo epoch**: mỗi epoch của bất kỳ ứng viên nào tính 1; các bước không huấn luyện (hồ sơ, baseline, proxy) không tính. Cùng lệnh cho cùng leaderboard trên máy nhanh hay chậm: dùng khi cần tái lập kết quả, so sánh hai cấu hình, hoặc chạy trong CI |
| `--device auto` | thiết bị huấn luyện: `auto` (GPU CUDA nếu có, rồi Apple MPS, rồi CPU), `cpu`, `cuda`, `cuda:1`, `mps`. Chọn `cuda` mà không có GPU thì báo lỗi ngay, trước khi tiêu ngân sách |
| `--amp auto` | mixed precision: `auto` = bfloat16 trên GPU hỗ trợ (fp16 kèm loss scaling trên GPU cũ), float32 trên CPU/MPS; `off` tắt; `bf16`/`fp16` ép (trên CPU chậm hơn float32, chỉ để thử) |
| `--recipe modern` | công thức huấn luyện: `modern` (mặc định) = augmentation cho ảnh (dịch tối đa 1/8 cạnh; lật ngang cho ảnh màu và phân đoạn) và label smoothing 0,1; `plain` = không (như DLNet ≤ 0.21) |
| `--threads 4` | số luồng CPU cho torch, BLAS, OpenMP. Mặc định `0` = số CPU còn rảnh lúc bắt đầu. **Chạy nhiều lệnh cùng lúc** trên một máy: đặt tổng `--threads` ≤ số CPU (ví dụ 4 lệnh trên máy 8 CPU: `--threads 2`); để mỗi lệnh dùng hết CPU có thể làm baseline treo rất lâu. Muốn hai lần chạy giống hệt từng bit: cố định `--threads` |
| `--max-params 500k`, `--max-flops 50M`, `--max-vram 4000` | loại ứng viên vượt giới hạn trước khi huấn luyện (VRAM là ước lượng, chưa đối chiếu GPU) |
| `--zoo 12`, `--evolve 1`, `--evolve-pop 8` | số kiến trúc từ zoo, số thế hệ tiến hóa, số con mỗi thế hệ |
| `--per-family 8` | số cấu hình ngẫu nhiên mỗi họ mô hình |
| `--cv-splits 3` | xếp hạng finalist trên 3 lần chia dữ liệu (ổn định hơn, chậm hơn) |
| `--no-cache` | không dùng lại các lần huấn luyện đã xong trong `--out` |

**GPU.** Trên GPU, mỗi mảng dữ liệu được chép lên GPU một lần (nếu chiếm dưới một nửa bộ nhớ GPU còn trống, ngược lại chép từng batch), ảnh dùng bố cục `channels_last`, và huấn luyện dùng mixed precision. Các bước phụ (data card, zero-cost proxy, surrogate, baseline cổ điển) vẫn chạy trên CPU. Ngân sách giây trên GPU cho nhiều ứng viên hơn; ngân sách epoch cho cùng số epoch nhưng kết quả số có thể khác CPU một chút (phép tính GPU không tất định). Thiết bị và độ chính xác được ghi trong khối `budget` của leaderboard.

**Ngân sách được chia thế nào (từ v0.21).** Mỗi rung của successive halving (4, 12, 30 epoch) được một phần bằng nhau, nên với ngân sách vừa phải ít ứng viên được thử hơn nhưng finalist được huấn luyện sâu hơn. Tiến hóa tự bỏ qua khi không đủ ngân sách (`[evolve] skipped`). Phần còn lại dùng để chạy lại các finalist tốt nhất (đo được nhiễu thật), sau đó mới ước lượng hyperparameter (`--hparams auto`: chỉ cho finalist có ≥ 2 lần chạy; `on`: luôn; `off`: tắt).

DLNet chỉ bắt đầu một lần huấn luyện khi dự báo nó kết thúc trước hạn, nên ngân sách giây được giữ sát; ngân sách epoch không bao giờ bị vượt. Dự báo dùng thời gian mỗi epoch đã đo; với ứng viên chưa train, từ v1.0 DLNet đo thật một bước forward+backward rồi nhân với số bước mỗi epoch. Trước đây dự báo theo FLOPs, mà FLOPs không tính vòng lặp Python của MambaVision, KAT, xLSTM (chậm 10–100 lần so với FLOPs), nên `--budget 90s` từng chạy hơn 12 phút. Mục **Budget** cuối `leaderboard.md` cho biết từng giai đoạn tốn bao nhiêu giây và epoch.

## 8. Đọc kết quả

`leaderboard.md` gồm:

0. **Warnings** (nếu có): top-1 mới được huấn luyện ít epoch, mô hình chỉ chạy một lần, Top-K lẫn số epoch khác nhau, ứng viên bị lỗi/phân kỳ đã bị loại (`failed_candidates` trong JSON), HSI chia theo pixel (rò rỉ không gian), cột CSV bất thường. Đọc mục này trước.
1. **Bảng Top-K**: họ, vai trò, `group` (cùng số = không phân biệt được trong nhiễu), `epochs` (số epoch đã huấn luyện), `runs` (số lần chạy: seed × lần chia), tham số, MFLOPs, VRAM ước lượng, hyperparameter, validation (trung bình ± nhiễu), test.
   - **Xếp hạng chỉ so các mô hình cùng số epoch**: mô hình nhiều epoch hơn đứng trước; trước khi xếp, DLNet cố huấn luyện tiếp các mô hình ít epoch tới cùng mức nếu còn ngân sách.
   - **± nhiễu** là độ lệch qua các lần chạy, nhưng không nhỏ hơn sai số lấy mẫu của tập validation (ví dụ accuracy: √(p(1−p)/n)). Với `runs` = 1 đây là ước lượng nhiễu duy nhất, nên không còn "± 0".
   - Vai trò: `best-small` chỉ gán khi nhỏ hơn top-1; `most-stable` chỉ gán khi mọi mô hình được chạy ≥ 2 lần; `most-different` là họ khác top-1.
2. **Baseline cổ điển** và **kết luận** (`beats_reference`): chỉ "có" khi top-1 hơn baseline tốt nhất trên validation quá mức nhiễu **và** hơn mọi baseline trên test quá sai số lấy mẫu của tập test. "Không" nghĩa là nên cân nhắc baseline cổ điển trước.
3. **Objective và Pareto front**.
4. **Zero-cost proxy**: tương quan với kết quả huấn luyện thật (chỉ để tham khảo).
5. **Budget**: giây và epoch theo giai đoạn (load, reference, generate, proxies, seed_selection, search, evolution, hparams, export), số luồng, phần trăm vượt ngân sách.

`leaderboard.json` chứa cùng thông tin dạng máy đọc (JSON chuẩn: giá trị không xác định ghi là `null`), cộng `seed_selection` (data card, quy tắc, xếp hạng), `module_zoo` (tiến hóa), `objective`.

## 9. Huấn luyện lại một baseline

Mỗi `top<i>_*` là một gói độc lập: cần torch và numpy; gói bảng cần thêm pandas, gói HSI cần scipy (khi đọc `.mat`). Các lệnh chạy được từ bất kỳ thư mục nào.

```bash
python dlnet_out/top1_resnet/train.py --data images.npz --epochs 100 --seed 0   # GPU tự dùng nếu có
python dlnet_out/top1_resnet/train.py --data images.npz --device cpu --amp off  # ép CPU, float32
python -m pytest dlnet_out/top1_resnet/tests -q
```

### Dùng mạng trong notebook hoặc dự án riêng

`model.py` của mỗi gói là **chính mạng được sinh ra, viết bằng PyTorch thuần**: không import DLNet, chép riêng file này đi là dùng được.

```python
import sys, torch

sys.path.insert(0, "dlnet_out/top1_resnet")
from model import build_model

model = build_model()  # torch.nn.Module
model.load_state_dict(torch.load("dlnet_out/top1_resnet/model.pt"))  # tùy chọn: trọng số đã train
```

Docstring đầu file ghi kích thước đầu vào, đầu ra, số tham số và nguồn (paper, repo) của từng khối. `config.yaml` là định dạng của DLNet (nội dung JSON), không phải chuẩn của PyTorch: nó giữ spec, hyperparameter (`hyperparameters`) và cách dựng lại dữ liệu; `model.py` không cần nó. Gói xuất từ bản cũ (có thư mục `zoo/`): `python -m dlnet standalone dlnet_out/top1_resnet`.

`train.py` dùng công thức của lần tìm kiếm (`config.yaml` → `recipe`: warm-up rồi cosine, augmentation và label smoothing nếu `modern`), giữ checkpoint tốt nhất theo validation, rồi **lưu `model.pt` (trọng số) và `preprocess.json`** (thống kê chuẩn hóa của TRAIN, tên cột, danh sách lớp, ...).

### Dự đoán trên dữ liệu mới

```bash
python dlnet_out/top1_resnet/predict.py --data new_images.npz --out predictions.npy
python dlnet_out/top1_tabular_mlp/predict.py --data new_rows.csv --out predictions.csv   # bảng: CSV cùng cột, không cần cột đích
python dlnet_out/top1_resnet/predict.py --data new_images.npz --export model.pt2          # thêm: đồ thị triển khai (torch.export)
```

File mới cùng định dạng với dữ liệu huấn luyện, không cần nhãn. Kết quả theo đơn vị gốc: nhãn lớp gốc (ví dụ `"cat"`, `21`), giá trị hồi quy gốc, mask với giá trị lớp gốc (phân đoạn), ảnh tái tạo theo thang gốc, bản đồ lớp (H, W) cho ảnh siêu phổ. `--export model.onnx` cần `pip install onnx onnxscript`. Trong mã Python: `import inference; model, meta, cfg = inference.load("dlnet_out/top1_resnet"); y = inference.predict_file(model, meta, cfg, "new.npz")`.

## 10. Học từ các lần chạy trước

```bash
python -m dlnet design --data a.npz --meta-store runs.jsonl
python -m dlnet design --data b.npz --meta-store runs.jsonl   # dùng kinh nghiệm của a
```

Mỗi lần chạy thêm một dòng vào `runs.jsonl`. Từ 3 lần chạy cùng plugin trở đi, các bộ dữ liệu giống nhất bỏ phiếu thứ tự họ mô hình và DLNet dựng một **portfolio** (tập họ bổ sung cho nhau, thuật toán của Auto-sklearn 2.0). Đây là cách hiệu quả nhất để việc chọn hạt giống tốt dần trên dữ liệu của chính bạn.

## 11. Lệnh khác

```bash
python -m dlnet zoo                                               # liệt kê module
python -m dlnet unfold --data problem.npz --layers 4,8,16         # LISTA cho y = A x + nhiễu
python -m dlnet gan --data images.npz --budget-s 300              # DCGAN, xếp hạng theo MMD²
python -m dlnet mcp                                               # MCP server cho AI client (doc/ai-advisor.md)
```

## 12. Xử lý sự cố

| Triệu chứng | Cách xử lý |
|---|---|
| `cannot tell the data type` | thông báo liệt kê các mảng trong file: đổi tên thành `X`, `y` hoặc dùng `--x-key`, `--y-key` kèm `--plugin` |
| `--out` đã có kết quả cũ | DLNet không xóa thư mục lạ; chọn `--out` khác |
| `no valid candidate: ... none fits --max-params ...` | nới giới hạn (thông báo cho biết ứng viên nhỏ nhất có bao nhiêu tham số) hoặc tăng `--per-family`, `--zoo` |
| Chỉ vài ứng viên được huấn luyện | tăng `--budget`, hoặc giảm `--zoo`, `--per-family`, hoặc dùng `--seeds N` |
| Kết luận "không thắng baseline" | đó là kết quả thật: baseline cổ điển đủ tốt cho dữ liệu này, hoặc ngân sách quá ngắn |
| `every candidate failed to train` | xem `failed_candidates`/log: thường do lr quá lớn, dữ liệu có `inf`, hoặc spec lỗi trong `--extra-specs` |
| Lần chạy lâu hơn ngân sách nhiều, nhất là khi chạy nhiều lệnh cùng lúc | xem mục Budget trong `leaderboard.md`; đặt `--threads` để tổng số luồng ≤ số CPU |
| Hết bộ nhớ GPU | đặt `--max-vram`, giảm `--zoo-width` |
