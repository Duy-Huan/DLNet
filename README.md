# DLNet

Tu dong de xuat Top-K baseline deep learning (PyTorch) cho mot tap du lieu. Kien truc: xem `agent.md`.

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
- Plugin `image_classification`, `image_segmentation` (mask `y` (N,H,W), metric mIoU) va `timeseries` (file `.npz` co `X` va `y`).
- Thu vien khoi `dlnet/blocks.py` (backbone, decoder, attention, mixture of experts; nguon va giay phep o `THIRD_PARTY.md`): `python -m dlnet blocks`.
- `--meta-store file.jsonl`: hoc thu tu ho mo hinh tu cac lan chay truoc (chi sap thu tu; do duoc 58% top-3, chua dat muc tieu 80%).
- Chua co: cost model FLOPs/VRAM, Mamba/SSM, so lieu tren du lieu anh / chuoi thoi gian that.

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

## Phiên bản
Xem [CHANGELOG.md](CHANGELOG.md) và trang Releases (tag `vX.Y.Z`). Quy trình phát hành: `agent.md` §11.2.
