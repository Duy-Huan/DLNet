# Mã chính thức của các thuật toán (ghim commit)

Commit lấy bằng `git ls-remote` ngày 2026-11-10. Tải lại một file: `curl -sSL https://raw.githubusercontent.com/<repo>/<commit>/<đường dẫn>`.

| Repo | Commit | Giấy phép | File chính | Trong repo này |
|---|---|---|---|---|
| ganguli-lab/Synaptic-Flow | `9dfa57aa4012c34028ee646d367f18e32f0e3bc7` | không thấy file LICENSE | `Pruners/pruners.py` (SynFlow, SNIP, GraSP) | chỉ index (chưa rõ giấy phép) |
| BayesWatch/nas-without-training | `b3a82a6642564df115f989ff940ec6b8ef9ca9d3` | MIT (README ghi "licensed under the MIT licence", không có file LICENSE) | `score_networks.py`, `scores.py` | `official_code/nas-without-training/` |
| SLDGroup/ZiCo | `b0fec65923a90e84501593f675b1e2f422d79e3d` | Apache-2.0 | `ZeroShotProxy/compute_zico.py` | `official_code/ZiCo/` + LICENSE |
| idstcv/ZenNAS | `d1d617e0352733d39890fb64ea758f9c85b28c1a` | Apache-2.0 (badge trong README, không có file LICENSE) | `ZeroShotProxy/compute_zen_score.py` | `official_code/ZenNAS/` |
| VITA-Group/TENAS | `9df78ffd98573035375b12e19b9007578cc4155d` | MIT | `lib/procedures/ntk.py`, `lib/procedures/linear_region_counter.py` | `official_code/TENAS/` + LICENSE |
| CalculatedContent/WeightWatcher | `a5940f79e9d5bbc37c82360c6f2f8a3aefe6543a` | Apache-2.0 | `weightwatcher/weightwatcher.py` | chỉ index (file lớn) |
| automl/HpBandSter | `0ec0ce23448ae7f27c0868b35a7086bc6d0fba2a` | BSD-3-Clause | `hpbandster/` (BOHB, Hyperband) | chỉ index |
| SamsungLabs/zero-cost-nas | không truy cập được (git đòi đăng nhập) | chưa biết | `foresight/pruners/measures/` | không |

Các file trong `official_code/` giữ nguyên nội dung gốc (không sửa, không chạy trong CI; `pyproject.toml` loại `ref/` khỏi ruff). Chúng là tài liệu để đối chiếu công thức, không phải phụ thuộc của DLNet.

Không có mã chính thức công khai cho: LR range test (Smith), gradient noise scale (McCandlish), normalized weight decay (AdamW paper). Phần implement trong `dlnet/hparams.py` theo công thức trong paper; những chi tiết lấy từ cách làm phổ biến (fastai) được ghi trong docstring là "chưa đối chiếu".

## Phát hiện khi đọc mã chính thức (để implement đúng sau này)

- ZiCo (`compute_zico.py`): gradient của từng lớp Conv/Linear trên nhiều batch; với mỗi lớp lấy `sum(mean|g| / std(g))` trên các phần tử có std khác 0, rồi cộng `log` qua các lớp. Mã gốc dùng `.cuda()` cứng.
- Zen-Score (`compute_zen_score.py`): khởi tạo Gaussian, đầu ra trước GAP của `x` và `x + gamma * noise`, `log(mean|Δ|) + sum(log sqrt(mean(running_var) của BN))`. Cần mạng CNN có BN; không áp dụng trực tiếp cho MLP.
- SynFlow (Synaptic-Flow): lấy `abs` mọi tham số, đưa đầu vào toàn số 1, `sum(output).backward()`, điểm = `|grad * param|`, rồi khôi phục dấu.
- NASWOT (`score_networks.py`): hook trên mỗi ReLU lấy `x = (input > 0)`, cộng dồn `K += x xᵀ + (1-x)(1-x)ᵀ` qua các lớp, điểm = `slogdet(K)`.
