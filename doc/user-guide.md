# Hướng dẫn sử dụng DLNet

Mục lục: [1. Cài đặt](#1-cài-đặt) · [2. Chạy lần đầu](#2-chạy-lần-đầu) · [3. Chuẩn bị dữ liệu](#3-chuẩn-bị-dữ-liệu) · [4. Chạy theo loại bài toán](#4-chạy-theo-loại-bài-toán) · [5. Chọn hướng tối ưu](#5-chọn-hướng-tối-ưu) · [6. Ngữ cảnh và chọn hạt giống](#6-ngữ-cảnh-và-chọn-hạt-giống) · [7. Ngân sách và giới hạn chi phí](#7-ngân-sách-và-giới-hạn-chi-phí) · [8. Đọc kết quả](#8-đọc-kết-quả) · [9. Huấn luyện lại một baseline](#9-huấn-luyện-lại-một-baseline) · [10. Học từ các lần chạy trước](#10-học-từ-các-lần-chạy-trước) · [11. Lệnh khác](#11-lệnh-khác) · [12. Xử lý sự cố](#12-xử-lý-sự-cố)

Mọi tùy chọn có trong [cli.md](cli.md).

## 1. Cài đặt

Cần Python 3.12 (đã kiểm tra trên Linux, macOS, Windows).

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt      # numpy, pandas, scipy, scikit-learn, torch (phiên bản ghim)
python -m dlnet selfcheck            # khoảng 20 giây; in "selfcheck OK"
```

Máy không có GPU: cài torch bản CPU trước (`pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu`). Script `scripts/setup_env.sh` / `setup_env.ps1` làm cả các bước trên.

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
| `--budget 10m` | tổng thời gian; luôn có ít nhất một kết quả kể cả khi hết giờ sớm |
| `--max-params 500k`, `--max-flops 50M`, `--max-vram 4000` | loại ứng viên vượt giới hạn trước khi huấn luyện (VRAM là ước lượng, chưa đối chiếu GPU) |
| `--zoo 12`, `--evolve 1`, `--evolve-pop 8` | số kiến trúc từ zoo, số thế hệ tiến hóa, số con mỗi thế hệ |
| `--per-family 8` | số cấu hình ngẫu nhiên mỗi họ mô hình |
| `--cv-splits 3` | xếp hạng finalist trên 3 lần chia dữ liệu (ổn định hơn, chậm hơn) |
| `--no-cache` | không dùng lại các lần huấn luyện đã xong trong `--out` |

## 8. Đọc kết quả

`leaderboard.md` gồm:

1. **Bảng Top-K**: họ, vai trò (`top1`, `best-small`, `most-stable`, `most-different`), `group` (cùng số = không phân biệt được trong nhiễu), tham số, MFLOPs, VRAM ước lượng, hyperparameter, validation (trung bình ± độ lệch qua seed), test.
2. **Baseline cổ điển** và **kết luận** (`beats_reference`): chỉ "có" khi top-1 hơn baseline tốt nhất trên validation quá mức nhiễu và hơn mọi baseline trên test. "Không" nghĩa là nên cân nhắc baseline cổ điển trước.
3. **Objective và Pareto front**.
4. **Zero-cost proxy**: tương quan với kết quả huấn luyện thật (chỉ để tham khảo).

`leaderboard.json` chứa cùng thông tin dạng máy đọc, cộng `seed_selection` (data card, quy tắc, xếp hạng), `module_zoo` (tiến hóa), `objective`.

## 9. Huấn luyện lại một baseline

Mỗi `top<i>_*` là một gói độc lập (chỉ cần torch và numpy):

```bash
cd dlnet_out/top1_resnet
python train.py --data ../../images.npz --epochs 100 --seed 0
python -m pytest tests -q
```

`config.yaml` (JSON hợp lệ) ghi spec kiến trúc, hyperparameter (`hyperparameters.source` cho biết giá trị đã tìm hay đã ước lượng), cách dựng lại dữ liệu và seed chia. Muốn dùng mô hình trong mã của bạn: `from model import build_model; m = build_model(cfg["arch"], cfg["in_dim"], cfg["out_dim"])`.

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
| `cannot tell the data type` | thêm `--plugin ...` hoặc đổi tên mảng / dùng `--x-key`, `--y-key` |
| `--out` đã có kết quả cũ | DLNet không xóa thư mục lạ; chọn `--out` khác |
| `no valid candidate was generated` | nới `--max-params` / `--max-flops`, tăng `--per-family` |
| Chỉ vài ứng viên được huấn luyện | tăng `--budget`, hoặc giảm `--zoo`, `--per-family`, hoặc dùng `--seeds N` |
| Kết luận "không thắng baseline" | đó là kết quả thật: baseline cổ điển đủ tốt cho dữ liệu này, hoặc ngân sách quá ngắn |
| Hết bộ nhớ GPU | đặt `--max-vram`, giảm `--zoo-width` |
