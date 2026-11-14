# DLNet

Tu dong de xuat Top-K baseline deep learning (PyTorch) cho mot tap du lieu. Kien truc: xem `agent.md`.

**Tài liệu đầy đủ: [`doc/`](doc/README.md)** (hướng dẫn sử dụng, tham chiếu lệnh, kiến trúc, thuật toán).

## Cài như một phần mềm (không cần Python)

Mỗi bản phát hành (trang **Releases**) có gói cài cho Windows (`.exe`), macOS Apple silicon (`.pkg`) và Linux x64 / ARM64 (`.deb`), kèm bản chạy ngay `.zip`/`.tar.gz`. Cài xong dùng lệnh `dlnet` trong terminal (`dlnet --version`, `dlnet selfcheck`, `dlnet design ...`, `dlnet serve`). Mỗi gói được CI cài thật, chạy thử và gỡ trước khi tag được tạo. Chi tiết, GPU và cách gỡ: [doc/install.md](doc/install.md).

## Cài đặt nhanh (môi trường Python)

Môi trường ảo (`.venv`) **không** nằm trong git (nặng hàng GB, riêng từng máy, GitHub từ chối file trên 100 MB). Repo ghi lại **phiên bản Python** (`.python-version` = 3.12) và **danh sách package** (`requirements.txt`). Sau khi `git pull`, dùng Python 3.12:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt    # thêm pytest/ruff: requirements-dev.txt
python -m dlnet selfcheck          # ~20 giây, in "selfcheck OK" là dùng được
```

Chạy lệnh từ thư mục gốc của repo, không cần cài thêm gói. Muốn gõ `dlnet ...` thay cho `python -m dlnet ...`: `pip install -e .`. Linux/Windows chỉ cần torch CPU nhẹ hơn: xem dòng chú thích trong `requirements.txt`.

Tùy chọn (làm hết các bước trên bằng một lệnh, tự tìm Python 3.10–3.13 và tự lùi về torch PyPI nếu mạng chặn): `./scripts/setup_env.sh` hoặc `powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1`. CI kiểm chứng cả hai đường (script và `requirements.txt`) ở mỗi lần push.

## Trang thai
- Plugin `tabular` (CSV) va plugin `hsi` (anh sieu pho: cube + ground truth, phan loai pixel).
- Lenh `dlnet unfold`: deep unfolding (LISTA) cho bai toan nghich thua `y = A x + nhieu` khi biet ma tran `A`.
- Module thu vien: Tensor-LoRA (`dlnet/tensor_lora.py`), HOSVD (`dlnet/tensor.py`); chua duoc `design` dung.
- Plugin `graph` (X, A, y), `recsys` (ids, y), `image_reconstruction` (chi X); lenh `dlnet gan --data images.npz` (DCGAN, xep hang theo MMD).
- Plugin `image_classification`, `image_segmentation` (mask `y` (N,H,W), metric mIoU) va `timeseries` (file `.npz` co `X` va `y`).
- **Kho module `dlnet/zoo/` (210 module + 11 decoder phan doan, tu torchvision, timm, segmentation_models_pytorch, MONAI va ma chinh thuc cua cac paper 2022–2025; doi chieu so hoc voi ma goc tren GitHub Actions, xem `THIRD_PARTY.md`)**: moi `design` them cac kien truc tu zoo (`--zoo`, `--evolve`: aging evolution); `python -m dlnet zoo` liet ke module; nguon o `THIRD_PARTY.md`. v1.4 them 21 module va 13 ho (DLinear, TSMixer, TiDE; GATv2, GCNII, APPNP, SGC; xDeepFM, AutoInt, FiBiNET; Set Transformer, attention pooling; InceptionNeXt, FocalNet, non-local, PVTv2, BlurPool, SimAM, GeM); 53/53 phep doi chieu voi ma goc khop.
- **Phan tich du lieu anh va chon hat giong**: `[datacard]` (pho, do dai tuong quan, phu thuoc vi tri, probe tich chap ngau nhien...), `--context "medical CT"`, `--goal small`, `--seeds N`, `--scheduler hyperband`. Do tren MNIST/Fashion: prior chua du doan duoc thu hang (`agent.md` §16.5), nen mac dinh khong loai ung vien.
- **DLNet la mot dich vu MCP qua HTTP** (`python -m dlnet serve`): Claude Code, Codex, Antigravity ket noi toi `http://<host>:8765/mcp` va dung DLNet de phan tich du lieu, sinh mang, huan luyen, danh gia, du doan (20 tool, job nen, token, workspace co lap). Huong dan: [doc/mcp-server.md](doc/mcp-server.md).
- **v1.8**: thu vien lop co ban (gan du `torch.nn` + cac phep tensor, `dlnet/zoo/ops_prim.py`) va `dlnet paper import` viet lai: **75/75 mang phan loai cua torchvision dung lai trung tung bit** tu ma goc (ResNet, EfficientNet, ConvNeXt, ViT, Swin v1/v2, MaxViT, ...), kiem bang `scripts/bench_import.py`; v1.8.1: timm (mau co dinh 200 mang) **161/163 dung lai trung tung bit**, ke ca mang co `if` tren kich thuoc trong forward; 5 khoi hieu qua moi cho tim kiem (FasterNet, MambaOut, MobileViT v1/v2, EdgeNeXt), trung tung bit voi timm.
- **v1.7**: `dlnet plan` (uoc luong truoc khi chay), so voi 10-20 paper trong mot lenh co kiem dinh t ghep cap + hieu chinh Holm, bang LaTeX `results.tex` (`dlnet latex`), va **giao dien web tuy chon `dlnet ui`** (6 man hinh, tieng Anh/Viet dong bo, sang/toi). Huong dan: [doc/ui.md](doc/ui.md).
- `--rung-score val|tse|val+tse`, `--rungs`: cach xep hang moi vong successive halving (TSE, Ru et al. 2021). Do o v1.5: khong cai thien top-1, mac dinh giu `val` (`qa/reports/v1.5.0/`).
- `--meta-store file.jsonl`: hoc thu tu ho mo hinh tu cac lan chay truoc (chi sap thu tu; do duoc 58% top-3, chua dat muc tieu 80%).
- Cost model FLOPs/VRAM (`--max-flops`, `--max-vram`), Mamba/Mamba-2. Chua co: so lieu tren du lieu anh mau that (CIFAR...) va chuoi thoi gian that.

## Dung thu
```bash
pip install -e ".[design]"

# tabular
python -m dlnet design --data data.csv --target label --budget 5m --topk 3

# hyperspectral (.mat / .npz / .npy)
python -m dlnet design --data cube.mat --gt gt.mat --split block --patch 9 --pca 30 --budget 10m --topk 3

# anh (.npz: X (N,H,W[,C]), y) va chuoi thoi gian (.npz: X (N,T[,C]), y)
python -m dlnet design --data images.npz --budget 10m --topk 3          # y (N,): phan loai
python -m dlnet design --data masks.npz --budget 10m --topk 3           # y (N,H,W): phan doan
python -m dlnet design --data series.npz --budget 10m --topk 3

# bai toan nghich thua y = A x + nhieu (.npz chua A, Y, X): LISTA so voi ISTA / FISTA cung so vong
python -m dlnet unfold --data problem.npz --layers 4,8,16 --epochs 100
```
Moi baseline co kem hyperparameter (lr, weight decay, batch, epochs, schedule) o `hyperparameters` trong `config.yaml`/`leaderboard.json`, uoc luong bang LR range test + gradient noise scale va chi nhan khi tot hon ban da tim kiem.
Nhan so nguyen la ma lop lien tiep (0..K-1) trong file anh/chuoi duoc hieu la phan loai du K > 20; khi khong chac DLNet in canh bao, ep bang `--task`. Ung vien duoc dung lan luot, ung vien khong dung duoc (het bo nho) bi loai chu khong lam hong luot chay.
`--cv-splits K`: xep hang finalist tren K lan chia du lieu khac nhau (HSI chia khoi mac dinh 3), de +/- std gom ca nhieu do cach chia.
`leaderboard.md` co them bang tuong quan cua zero-cost proxy (chi de tham khao, khong loai ung vien). Chay lai dung lenh cu se dung lai cac lan huan luyen da xong (`--no-cache` de tat).

Moi thu muc dau ra co `baselines.ipynb` (Jupyter / Google Colab: setup moi truong, du lieu, so do khoi, train tung baseline, so sanh, nap lai) va moi goi co `architecture.svg` (so do khoi kieu paper); thu muc cu: `python -m dlnet notebook <out> --data <file>`. `dlnet paper import` sinh nhap the tu ma goc (torch.fx); `dlnet design --papers papers/` train cac mang tu paper tren cung du lieu, cung giao thuc voi top-1 va them bang so sanh (so paper bao / giao thuc chung / cong thuc tac gia). **Mang tu paper**: viet mot mang da cong bo thanh the YAML trong `papers/` (moi khoi mot dong `{id, from, module, args}`), `python -m dlnet paper check|build|verify` dung lai no thanh baseline va do muc trung thanh so voi ma goc cua tac gia (A/B/C; xem doc/user-guide.md). `predict.py --export model.pt2` luu do thi `torch.export` (chi do thi va trong so). `model.py` cua moi goi la chinh mang duoc sinh ra, viet bang PyTorch thuan (chi `import torch`, khong phu thuoc DLNet): chep rieng file do vao notebook la dung duoc (`from model import build_model`). Goi cu: `python -m dlnet standalone <goi>`.

Ket qua: `dlnet_out/leaderboard.md` va `dlnet_out/top1_*/ ...` (moi goi doc lap: `model.py`, `core.py`, `train.py`,
`config.yaml`, `report.md`, `tests/`). Train day du mot goi: `python train.py -h`.

## Luu y ve danh gia HSI
- `--split pixel`: chia few-shot theo pixel, nhung patch cua train/test chong lan nen OA thuong bi thoi phong.
- `--split block`: chia theo khoi khong chong lan, bo pixel ma patch cham vung split khac. So lieu trung thuc hon.
- Thu hang chi la uoc luong tu ngan sach nho; cac hang cung `group` co the doi cho sau khi train day du.

## Phát triển
Trước khi push: `python scripts/ci_local.py` chạy các cổng của CI trên máy (`--quick` vài giây, `--full` toàn bộ test). Cài hook một lần: `python scripts/ci_local.py --install`, từ đó mọi `git push` tự kiểm trước.

## Phiên bản
Xem [CHANGELOG.md](CHANGELOG.md) và trang Releases (tag `vX.Y.Z`). Quy trình phát hành: `agent.md` §11.2.
