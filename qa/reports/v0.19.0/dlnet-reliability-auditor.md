# Kiểm định độ tin cậy (R1–R11) — DLNet 0.19.0

## Môi trường và dữ liệu

- Máy 4 CPU, **không GPU**, torch 2.15.0, Python 3.13. Máy dùng chung với 3 agent kiểm định khác (load average 4–8 trong lúc đo), nên mọi số đo **giây** đều bị nhiễu do tranh CPU; số đo **epoch** thì không.
- Mọi lệnh `dlnet design` dùng `--threads 1 --no-cache` và ngân sách epoch (`--budget Nep`), trừ hai lần đo R7 ở chế độ giây.
- Dữ liệu thật (`$QA_DATA`): `fashion3k.npz`, `fashion14.npz`, `digits.npz`, `breast_cancer.csv`, `wine.csv`, `diabetes_reg.csv` (hồi quy), `Indian_pines_corrected.mat` + `Indian_pines_gt.mat`. Dữ liệu xấu tự tạo cho R8 (26 file).
- Thư mục làm việc: `$QA_WORK/reliability/` (log, leaderboard, script `analyze.py`, `analyze2.py`, `retrain.py`, `r1.py`, `r9.py`).
- Tổng thời gian khoảng 85 phút máy (cộng thời gian chờ do tranh CPU).

Cấu hình chung của các lần chạy nhiều seed (để vừa ngân sách máy):
- bảng: `--budget 2000ep --topk 6 --per-family 2 --zoo 4 --evolve 0` (ngân sách này mới đủ để có giai đoạn nhiều seed, xem R11);
- ảnh nhỏ (`digits`, `fashion14`): `--budget 300ep --topk 6 --per-family 1 --zoo 4 --evolve 0`;
- `fashion3k`: một lần chạy cấu hình mặc định với `--budget 150ep --topk 3` (mất 1744 s, nên không chạy thêm seed).

## Bảng chấm

| ID | Kết quả | Bằng chứng (lệnh, số đo, file:dòng) |
|---|---|---|
| R1 | MỘT PHẦN | Các bước fit chỉ dùng train: impute median và chuẩn hóa (`dlnet/core.py:76-83`), chuẩn hóa ảnh/chuỗi (`array_data.py:87-89`), PCA và chuẩn hóa HSI trên pixel train (`hsi_data.py:182-195`), baseline cổ điển chỉ fit trên train (`reference.py:37`). Phép thử `r1.py`: nhân đặc trưng test ×100 (bảng) hoặc ghi đè ảnh test (fashion3k) thì tensor train/val **không đổi** (True/True). Nhưng **HSI mặc định `--split pixel`** (`plugins/hsi.py:110`): trên Indian Pines, 99,6% pixel val và 99,6% pixel test (n = 1025 / 8198) nằm trong cửa sổ 9×9 của một pixel train; đổi phổ pixel test làm thay đổi patch train (True). Với `--split block` là 0,0%. Tài liệu khuyên dùng `block` (`doc/user-guide.md:66`) nhưng mặc định vẫn là `pixel` và leaderboard không cảnh báo. |
| R2 | MỘT PHẦN | `export.verdict` (`export.py:58-85`): thắng khi val hơn baseline tốt nhất quá nhiễu **và** test hơn **mọi** baseline. Ca biên (gọi trực tiếp `verdict`): thắng val nhưng thua test → False (đúng); chênh val trong nhiễu → False (đúng); thua 1 baseline trên test → False (đúng). Lỗ hổng: (a) test chỉ cần hơn > 0, không có ngưỡng nhiễu: `test_margin=0.0001` → True; (b) hồi quy với `val_std=0` (chỉ 1 lần chạy, chuyện thường gặp, xem R11) thì nhiễu = 0: val hơn 0,001 RMSE và test hơn 0,001 → `beats_reference=True`; (c) không hiệu chỉnh việc top-1 là max của nhiều ứng viên. |
| R3 | ĐẠT | Chạy 2 lần giống hệt: `breast_cancer` 400ep, `diabetes_reg` 400ep, `digits` 300ep: `leaderboard.json` trùng khít (bỏ khóa `budget` chứa giờ thực): True/True/True. |
| R4 | KHÔNG ĐẠT | Seed 0/1/2 (seed đổi cả cách chia lẫn pool ứng viên, nên so ở mức **họ** mô hình). Top-1 nằm trong top-3 của **cả hai** seed kia: breast_cancer 1/3, wine 3/3, diabetes_reg 0/3, digits 0/3, fashion14 0/3. Kendall tau trên các họ chung (2–5 họ): từ −1,0 đến 1,0, đa số ≤ 0,33. Ngưỡng ≥ 2/3 chỉ đạt trên 1/5 bộ. |
| R5 | MỘT PHẦN | Spearman(val_mean, test) trên các finalist (top-k 5–6) và hạng test của top-1: fashion14 ρ = 0,67/0,72/0,60, hạng 1/1/2 (đạt); digits ρ = 0,36/0,15/0,63, hạng 2/2/2; breast_cancer ρ = −0,15/0,66/−0,07, hạng 6/1/1; wine ρ = −0,30/−0,62/−0,02, hạng 3/6/5; diabetes_reg ρ = 0,14/0,94/−0,09, hạng 4/2/5; fashion3k (1 seed, top-3) ρ = 0,50, hạng 2. Đạt trên bộ ảnh lớn hơn, không đạt trên bộ bảng nhỏ (val 34–113 mẫu). |
| R6 | KHÔNG ĐẠT | `retrain.py`: huấn luyện lại mỗi gói top-k bằng `train.py` của chính nó. fashion3k 30 epoch: Spearman(val tìm kiếm, val huấn luyện lại) = −0,50; top-1 đứng hạng 3/3 theo val, 2/3 theo test. digits 60 ep: ρ = 0,00, top-1 hạng 3 (val) / 2 (test). breast_cancer 60 ep: ρ = 0,02, hạng 3/2. diabetes_reg 60 ep: ρ = 0,20, hạng 5/5. wine 60 ep: ρ = −0,53, hạng 2/4. Không bộ nào đạt ρ ≥ 0,5; top-1 nằm trong top-2 theo test ở 2/5 bộ. |
| R7 | MỘT PHẦN | Epoch: luôn ≤ ngân sách (150ep → 144; 400ep → 372–398; 2000ep → ≤ 2000). Giây: 20s → 20 s nội bộ, 24,8 s cả tiến trình (×1,24 do import); 60s → 59 s nội bộ, 63,2 s cả tiến trình (×1,05). Nhưng ngân sách epoch **không giới hạn giờ thực**: `fashion3k --budget 150ep` chạy **1744 s**; giai đoạn hparams tốn 751 s cho 24 epoch được tính (search: 351 s / 64 ep), vì LR range test và gradient-noise-scale được coi là "miễn phí" ở chế độ work (`hparams.py:190-197`); proxies tốn 63 s, cũng không tính. |
| R8 | MỘT PHẦN | 26 ca. Xử lý đúng hoặc báo lỗi rõ (15): NaN trong đặc trưng (impute), cột hằng, nhãn chuỗi, nhãn NaN (bỏ hàng), lớp 1–2 mẫu, 1:50 (tự chuyển sang `balanced_accuracy`), 8 dòng, ảnh có NaN ("X contains NaN or infinite values; clean the data first"), X/y lệch số dòng, nhãn ảnh chuỗi, ảnh 1:50, ảnh có lớp 1 mẫu... Có traceback hoặc thông báo sai hướng (10): `inf` trong CSV → traceback sklearn "Input X contains NaN" từ `profile.py:20`; CSV rác → `UnicodeDecodeError` (`core.py:31`); CSV rỗng → `EmptyDataError`; 4 dòng → traceback `ValueError: too few samples` (`core.py:77`); 6 ảnh → traceback tương tự; ảnh 1 lớp → traceback sklearn "needs samples of at least 2 classes" (`reference.py:39`); `.npz` hỏng → `BadZipFile` trong `arrays.py:217` (`detect`); `.npz` có X 2 chiều, X 5 chiều, khóa khác X/y, hoặc `.npy` → bị định tuyến sang plugin **hsi** và báo "cannot pick the cube/ground truth ... pass --cube-key", sai hướng hoàn toàn. **Sai im lặng** (1): cột số lẫn chuỗi (`str_in_numeric.csv`) → `get_dummies` biến cột tín hiệu thành 133 cột one-hot (`core.py:34`), độ chính xác test 0,54 thay vì ~0,95, không có cảnh báo. Không có ca nào treo. |
| R9 | KHÔNG ĐẠT | `r9.py` bọc một họ ứng viên (`mlp`). (1) Ứng viên ném ngoại lệ khi huấn luyện → **cả lần chạy sập** (rc=1, `search.py:184` → `core.py:263`, không có leaderboard). (2) Ứng viên ra NaN ở bài hồi quy (diabetes_reg): val = NaN làm hỏng phép sắp xếp (`search.py:62`, `search.py:271`): ứng viên NaN đứng **hạng 2** ("nan +/- nan"), và top-1 (val −54,47) **kém hơn** hạng 3 (−52,12); khi không có ứng viên lỗi, top-1 là −52,12. Log còn báo sai "best val=-54.4674". (3) Phân loại: logits NaN → argmax = lớp 0, chỉ bị xếp thấp, không bị đánh dấu. |
| R10 | MỘT PHẦN | `train.py --epochs <epochs_used> --seed 0` so với `val_mean ± 2·val_std` trong `config.yaml`: bảng 15/18 nằm trong khoảng (breast_cancer 5/6, diabetes_reg 5/6, wine 5/6); 3 ca lệch đều **cao hơn**, phù hợp với việc `train.py` dùng lịch cosine và chọn checkpoint tốt nhất theo val (`core.py:303-313`), khác công thức tìm kiếm (LR cố định, epoch cuối). digits: std = 0 (1 lần chạy) nên không áp được tiêu chí; val huấn luyện lại 0,977–0,994 so với 0,975–0,989. |
| R11 | KHÔNG ĐẠT | Mỗi con số có `n_runs` và std, nhưng với ngân sách thường dùng giai đoạn nhiều seed **không chạy**: 300ep (digits, fashion14), 400ep (cả 3 bộ bảng), 150ep (fashion3k) → mọi finalist `n_runs=1`, leaderboard in "+/- 0.0" và không cảnh báo. Khi std = 0, `assign_groups` (`search.py:331-341`) dùng ngưỡng 1e-9 nên mỗi khác biệt 1 mẫu val thành một **nhóm riêng** (digits: 0,98873 và 0,98592, chênh 1/355 ảnh, ở nhóm 0 và 1), tức là báo "phân biệt được" đúng lúc không có thông tin nhiễu. Ngoài ra top-k trộn ứng viên huấn luyện **số epoch khác nhau** (fashion3k: top-1, top-2 có 12 epoch, top-3 có 4 epoch), trái với nguyên tắc ghi ở `search.py:189`; `leaderboard.md` không có cột epoch. Phải tới 1000–2000ep (bảng) mới có `n_runs=3`. |

## Phát hiện

### Nghiêm trọng (kết quả sai hoặc gây hiểu nhầm, sập, mất dữ liệu)

1. **Một ứng viên lỗi làm sập cả lần chạy, hoặc làm sai top-1 (R9).** Ngoại lệ trong `Trainer.fit` không được bắt ở `run_search`/`run_hyperband`/`stage2`/`hparams.refine`, nên mất toàn bộ ngân sách đã chạy (`search.py:184`). Val NaN (hồi quy) không bị loại: `sorted(key=-val)` với NaN cho thứ tự tùy ý, nên ứng viên NaN lọt top-2 và top-1 không phải mô hình có val tốt nhất (−54,47 so với −52,12). Bằng chứng: `$QA_WORK/reliability/r9_nan_diabetes_reg/leaderboard.md`, `r9_raise_breast_cancer.log`.
2. **Leaderboard báo độ chính xác giả khi ngân sách nhỏ (R11).** Với 150–400ep, mọi số là 1 lần chạy nhưng in "+/- 0.0", và cột `group` tách từng khác biệt 1 mẫu thành nhóm riêng. Người dùng đọc như vậy sẽ tin thứ hạng chắc chắn, trong khi R4/R6 cho thấy thứ hạng gần như là nhiễu.
3. **Thứ hạng trong ngân sách ngắn không giữ khi huấn luyện đầy đủ (R6) và không ổn định qua seed (R4).** Spearman sau khi huấn luyện lại từ −0,53 đến 0,20 trên 5 bộ; top-1 qua seed chỉ ổn định trên wine. Hệ quả: chỉ nên đọc leaderboard như một **tập ứng viên tốt**, không phải thứ tự. Tài liệu đã nói "các hạng liền nhau có thể đổi chỗ" (`doc/user-guide.md:69`), nhưng mức độ thực tế lớn hơn thế nhiều.
4. **Top-k trộn số epoch khác nhau (R11, R6).** Kết quả thế hệ 0 (dừng ở rung 1, 4 epoch) được nối với kết quả tiến hóa (12 epoch) rồi sắp xếp chung (`cli.py:355`, `hparams.refine`, `assign_groups`). Trên fashion3k, mobilenetv2 (4 ep) bị so trực tiếp với zoo_mbconv (12 ep).

### Quan trọng (làm giảm chất lượng hoặc khả năng dùng rõ rệt)

5. **HSI mặc định `--split pixel` gây rò rỉ không gian (R1).** 99,6% pixel test nằm trong patch của pixel train trên Indian Pines. Cần đổi mặc định sang `block` hoặc in cảnh báo trong leaderboard.
6. **Ngân sách epoch không giới hạn giờ thực (R7).** LR range test, gradient-noise-scale và proxies không tính vào ngân sách: `150ep` tốn 29 phút trên fashion3k, giai đoạn hparams chiếm 43%.
7. **Đầu vào xấu (R8).** 10/26 ca ra traceback hoặc thông báo sai hướng (định tuyến nhầm sang HSI khi `.npz` sai shape hoặc sai khóa). Cột số lẫn chuỗi bị one-hot im lặng làm hỏng mô hình.
8. **Quy tắc thắng baseline (R2)** không có ngưỡng nhiễu trên test và coi nhiễu = 0 khi chỉ có 1 lần chạy (hồi quy).

### Nên có

9. `train.py` dùng công thức khác tìm kiếm (cosine, checkpoint tốt nhất theo val), nên val trong gói luôn hơi lạc quan (R10). Nên in thêm val ở epoch cuối, hoặc ghi rõ trong `report.md`.
10. `train_full` sẽ gọi `load_state_dict(None)` nếu val NaN ở mọi epoch (`core.py:309-313`) (suy luận từ mã, chưa chạy).
11. Việc chấp nhận hparams ước lượng (chọn max của 2 cấu hình trên cùng val) làm `val_mean` bị lạc quan (`hparams.py:248`).

## Đề xuất nâng cấp

| Đề xuất | Vì sao (bằng chứng) | Nguồn (paper / mã chuẩn) | Công sức S/M/L | Lợi ích kỳ vọng |
|---|---|---|---|---|
| Bọc mỗi lần `fit`/`evaluate` bằng try/except; điểm không hữu hạn → loại (−inf) và ghi vào leaderboard mục "failed" | R9: sập rc=1; NaN lọt top-2 | Ray Tune, Optuna (`TrialState.FAIL`, prune khi NaN) | S | Lần chạy không bao giờ mất; top-1 đúng |
| Dành sẵn phần ngân sách cho stage 2 (≥ 3 seed cho top-k), hoặc cảnh báo rõ khi `n_runs=1`; không gán nhóm khi std không có (hiện "n/a" thay "+/- 0.0") | R11: n_runs=1 ở 150–400ep; nhóm giả | Bouthillier et al. 2021 "Accounting for variance in ML benchmarks"; Dodge et al. 2019 | S–M | Leaderboard trung thực |
| Không xếp chung ứng viên khác số epoch; thêm cột epoch vào `leaderboard.md`; huấn luyện bù các finalist về cùng số epoch | R11/finding 4 (fashion3k 12 so với 4 ep) | Li et al. 2018 (Hyperband: chỉ so cùng tài nguyên) | M | Thứ tự so sánh được |
| Bước "xác nhận" cuối: huấn luyện top-k với công thức đầy đủ (vài seed) trước khi xếp hạng chính thức, hoặc dùng learning-curve extrapolation | R6: ρ ≤ 0,2 | Domhan et al. 2015; Klein et al. 2017 (LC-Bayesian NN); ASHA (Li et al. 2020) | M | Top-1 giữ được khi huấn luyện dài |
| Kiểm định thống kê cho verdict: paired bootstrap / McNemar trên test, nhiễu tối thiểu từ cỡ mẫu cả cho hồi quy | R2 ca biên (b), (a) | Dietterich 1998; Bouthillier et al. 2021 | S | Tránh "thắng" do nhiễu |
| Mặc định HSI `--split block` hoặc cảnh báo khi dùng pixel | R1: 99,6% chồng lấn | Nalepa et al. 2019 (IEEE GRSL, "Validating hyperspectral image segmentation") | S | Số HSI không bị thổi phồng |
| Tính chi phí LR range test, GNS và proxies vào ngân sách epoch (quy về epoch tương đương) | R7: 150ep → 1744 s | — | S | Ngân sách epoch dự báo được giờ thực |
| Kiểm tra đầu vào trước khi định tuyến: `.npz` có X/y thì không gửi sang HSI; `inf` → báo như NaN; bắt lỗi đọc CSV/ZIP; cảnh báo cột object có nhiều giá trị số | R8: 10/26 ca | — | S | Lỗi dễ hiểu, không sai im lặng |

## Giới hạn của lần kiểm định này

- Chỉ chạy CPU, không GPU. Máy bị tranh CPU nặng (load 4–8), nên số giây (R7) có nhiễu; số epoch thì tái lập được.
- Để vừa ngân sách, các lần chạy nhiều seed dùng pool ứng viên thu nhỏ (`--per-family 1–2 --zoo 4 --evolve 0`) và bộ ảnh nhỏ (digits 8×8, fashion14 14×14). `fashion3k` chỉ có 1 seed (cấu hình mặc định, 150ep). **cifar10_3k và HSI không được chạy `design`**; HSI chỉ được kiểm bằng phép thử chia dữ liệu (R1).
- R4 so ở mức họ mô hình, vì `--seed` đổi cả cách chia dữ liệu lẫn cấu hình ngẫu nhiên của từng họ.
- R5/R6 tính trên top-k xuất ra (3–6 mô hình), không phải toàn bộ finalist (leaderboard không lưu chúng), nên Spearman có phương sai lớn. R6 huấn luyện lại 1 seed mỗi gói.
- R9 dùng monkeypatch (`r9.py`), vì `--extra-specs` không cho đặt lr. Ca `train_full` với NaN (mục 10) là suy luận từ mã, chưa chạy.
- Một số bước chạy song song 2 tiến trình của tôi (R8 và retrain cùng lúc với loạt ảnh) để kịp thời gian. Điều này không ảnh hưởng số epoch hay kết quả, chỉ ảnh hưởng số giây.
