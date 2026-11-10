# DLNet

Tu dong de xuat Top-K baseline deep learning (PyTorch) cho mot tap du lieu. Kien truc: xem `agent.md`.

## Trang thai
Prototype tabular (lat cat M0+M1). Chua co plugin HSI, cost model day du, Tensor-LoRA, unfolding.

## Dung thu
```bash
pip install -e ".[design]"
python -m dlnet design --data data.csv --target label --budget 5m --topk 3 --out dlnet_out
```
Ket qua: `dlnet_out/leaderboard.md`, `dlnet_out/top1_*/ ...` (moi goi co `model.py`, `core.py`, `train.py`,
`config.yaml`, `report.md`, `tests/`). Huan luyen day du mot goi: `python train.py --data data.csv --target label`.

Thu hang chi la uoc luong tu ngan sach nho; cac hang sat nhau (cung `group`) co the doi cho sau khi train day du.
