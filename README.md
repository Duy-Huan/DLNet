# DLNet

Tu dong de xuat Top-K baseline deep learning (PyTorch) cho mot tap du lieu. Kien truc: xem `agent.md`.

## Cài đặt nhanh (môi trường Python)

Môi trường ảo (`.venv`) **không** nằm trong git (nặng hàng GB, chứa đường dẫn riêng của từng máy, GitHub từ chối file trên 100 MB). Thay vào đó repo có script tạo đúng môi trường trong vài phút. Sau khi `git pull`:

```bash
# Linux / macOS / Git Bash
./scripts/setup_env.sh            # thêm --gpu nếu muốn torch bản CUDA
source .venv/bin/activate         # Git Bash: source .venv/Scripts/activate
dlnet selfcheck                   # ~20 giây, in "selfcheck OK" là dùng được
```

```powershell
# Windows PowerShell
powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1
.venv\Scripts\Activate.ps1
dlnet selfcheck
```

Cần Python 3.10–3.13. Script tự chọn Python phù hợp, cài torch CPU (nhẹ; nếu chỉ mục pytorch.org bị chặn thì tự dùng bản PyPI), cài DLNet cùng pytest/ruff, rồi tự kiểm tra. Ai quen `pip` có thể dùng `pip install -r requirements.txt` (hoặc `requirements-dev.txt`). CI chạy đúng quy trình này trên Ubuntu, macOS và Windows ở mỗi lần push.

## Trang thai
- Plugin `tabular` (CSV) va plugin `hsi` (anh sieu pho: cube + ground truth, phan loai pixel).
- Chua co: zero-cost proxy, cost model day du, Tensor-LoRA, deep unfolding, plugin anh / chuoi thoi gian.

## Dung thu
```bash
pip install -e ".[design]"

# tabular
python -m dlnet design --data data.csv --target label --budget 5m --topk 3

# hyperspectral (.mat / .npz / .npy)
python -m dlnet design --data cube.mat --gt gt.mat --split block --patch 9 --pca 30 --budget 10m --topk 3
```
Ket qua: `dlnet_out/leaderboard.md` va `dlnet_out/top1_*/ ...` (moi goi doc lap: `model.py`, `core.py`, `train.py`,
`config.yaml`, `report.md`, `tests/`). Train day du mot goi: `python train.py -h`.

## Luu y ve danh gia HSI
- `--split pixel`: chia few-shot theo pixel, nhung patch cua train/test chong lan nen OA thuong bi thoi phong.
- `--split block`: chia theo khoi khong chong lan, bo pixel ma patch cham vung split khac. So lieu trung thuc hon.
- Thu hang chi la uoc luong tu ngan sach nho; cac hang cung `group` co the doi cho sau khi train day du.

## Phiên bản
Xem [CHANGELOG.md](CHANGELOG.md) và trang Releases (tag `vX.Y.Z`). Quy trình phát hành: `agent.md` §11.2.
