# DLNet

Tu dong de xuat Top-K baseline deep learning (PyTorch) cho mot tap du lieu. Kien truc: xem `agent.md`.

**Tài liệu đầy đủ: [`doc/`](doc/README.md)** (hướng dẫn sử dụng, tham chiếu lệnh, kiến trúc, thuật toán).

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
- **Kho module `dlnet/zoo/` (133 module + 11 decoder phan doan, tu torchvision, timm, segmentation_models_pytorch, MONAI va ma chinh thuc cua cac paper 2022–2025; doi chieu so hoc voi ma goc tren GitHub Actions, xem `THIRD_PARTY.md`)**: moi `design` them cac kien truc tu zoo (`--zoo`, `--evolve`: aging evolution); `python -m dlnet zoo` liet ke module; nguon o `THIRD_PARTY.md`.
- **Phan tich du lieu anh va chon hat giong**: `[datacard]` (pho, do dai tuong quan, phu thuoc vi tri, probe tich chap ngau nhien...), `--context "medical CT"`, `--goal small`, `--seeds N`, `--scheduler hyperband`. Do tren MNIST/Fashion: prior chua du doan duoc thu hang (`agent.md` §16.5), nen mac dinh khong loai ung vien.
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
`--cv-splits K`: xep hang finalist tren K lan chia du lieu khac nhau (HSI chia khoi mac dinh 3), de +/- std gom ca nhieu do cach chia.
`leaderboard.md` co them bang tuong quan cua zero-cost proxy (chi de tham khao, khong loai ung vien). Chay lai dung lenh cu se dung lai cac lan huan luyen da xong (`--no-cache` de tat).

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
