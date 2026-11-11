# Kiểm định thuật toán (nhóm A1–A9) — DLNet 0.19.0

## Môi trường và dữ liệu

- Máy: Linux, 4 CPU, **không có GPU**, dùng chung với 3 agent kiểm định khác (load trung bình 4–7 trong lúc đo, nên số giây chỉ để tham khảo; mọi so sánh dùng ngân sách **epoch**).
- Mọi lệnh: `--threads 1 --no-cache`, ngân sách `--budget 150ep`, chạy tuần tự.
- Dữ liệu thật (chỉ đọc): `mnist14` (2000 ảnh 14×14), `fashion14` (2000 ảnh 14×14), `digits` (1797 ảnh 8×8); meta-dataset có sẵn `dlnet/data/meta_image.jsonl` (14 bộ) cho A7.
- Thư mục làm việc: `$QA_WORK/algorithm/` (`runs/`, `scripts/`, `pilot/`, `bench_surrogate.json`, `hb_crash_mnist14_s0.log`).
- Tổng thời gian khoảng 80 phút; 27 lần chạy `design` + 1 lần `bench_surrogate.py` + 1 bản LODO bổ sung.

**Thiết kế thí nghiệm A/B (A2, A3, A4).** Cấu hình gốc `base` = `--seeds 16 --seed-order data --scheduler sh --evolve 0 --no-proxies`; mỗi nhánh đổi đúng một yếu tố: `random` (`--seed-order random`), `hb` (`--scheduler hyperband`), `evo2` (`--evolve 2`). Ghép cặp theo (bộ dữ liệu, `--seed`), vì `--seed` đổi cả cách chia dữ liệu. Để đủ số lần lặp, giai đoạn ước lượng hyperparameter được thay bằng hàm giữ nguyên (cùng cho mọi nhánh) qua `scripts/driver.py` trong thư mục làm việc — không sửa repo. Giai đoạn này được đo riêng ở A5 bằng lệnh mặc định. Nhánh `hb` cần thêm một **lớp chặn lỗi** trong driver (xem phát hiện N1); không có lớp chặn thì nhánh này sập.

Độ nhiễu tham chiếu: tập val của mnist14/fashion14 có 198 ảnh (1 ảnh = 0,005), digits khoảng 355 ảnh (1 ảnh = 0,0028). Chênh lệch top-1 giữa seed 0 và seed 1 của `base`: mnist14 0,010 (val) / 0,012 (test); digits 0 / 0,008. Mọi top-1 trong các lần chạy này chỉ có **1 lần chạy** (`n_runs = 1`, std = 0), vì ngân sách hết trước giai đoạn nhiều seed, nên độ nhiễu phải lấy từ các cặp seed.

## Bảng chấm

| ID | Kết quả | Bằng chứng (lệnh, số đo, file:dòng) |
|---|---|---|
| A1 | MỘT PHẦN | Đã đối chiếu 7 thuật toán với PDF trong `ref/` hoặc mã chính thức: Hyperband (Algorithm 1 và Table 1 trong `ref/optimazeDL/hp_hyperband_1603.06560.pdf`), aging evolution (Algorithm 1 trong `ref/02_regularized-evolution_1802.01548.pdf`), MnasNet reward (pt. 2, α = β = −0,07), crowding distance của NSGA-II, gradient noise scale (McCandlish phụ lục A.1), ZiCo (`official_code/ZiCo/compute_zico.py`), NASWOT (`official_code/nas-without-training`). Công thức đúng ở cả 7; các sai lệch đều đã ghi trong docstring. Còn 2 sai lệch **chưa ghi**: (1) LR range test lấy lr_max = **cực tiểu** của loss rồi đề xuất lr_max/2 (`hparams.py:157-161`); Smith định nghĩa max_lr là mức LR mà độ chính xác bắt đầu giảm (cho SGD + CLR), còn quy tắc fastai thường dùng LR thấp hơn cực tiểu khoảng 10 lần. Áp nguyên cho AdamW sinh ra lr 0,013–0,056, gấp 5–56 lần lr đã tìm (xem A5). (2) `run_search` không phải SH của Jamieson & Talwalkar: rung 30 thay vì 36 = 4·3², và luôn giữ ít nhất `2·topk` = 6 ứng viên (`search.py:195`), nên với 16 ứng viên rung 2 không loại ai |
| A2 | KHÔNG ĐẠT | `--seeds 16`, data so với random, 150ep, mnist14 và digits × seed 0/1: val top-1 random − data = +0,0101, 0, −0,0028, −0,0028 (TB +0,0011, sd 0,0053); test: 0, 0, 0, +0,0081. Random thắng hoặc hòa ở 3/4 cặp trên val và 4/4 trên test. fashion14 seed 0: xem bảng A/B. Không có chênh lệch nào vượt nhiễu |
| A3 | MỘT PHẦN (số đo có; Hyperband **sập**) | `--scheduler hyperband` sập với `KeyError: 'val0'` ở `search.py:305` (`hb_crash_mnist14_s0.log`). Lớp chặn lỗi cũng bật ở 4/4 lần chạy còn lại, tức là cả 4 sẽ sập. Khi đã chặn lỗi: hb − sh: val +0,0177, −0,0253, −0,0028, −0,0141 (TB −0,0061); test +0,0074, −0,0123, 0, 0. Hyperband chỉ xuất **1** mô hình thay vì 3 ở mọi lần chạy. Khuyến nghị: giữ `sh` làm mặc định |
| A4 | KHÔNG ĐẠT | `--evolve 2` so với 0, cùng 150ep: val +0,0050, −0,0354, +0,0028, +0,0056; **test −0,0098, −0,0368, 0, −0,0027** (TB −0,0123; không thắng test lần nào trong 4). Top-1 của nhánh evo2 luôn chỉ được huấn luyện **4 epoch** (nhánh base: 12), vì chia 55/45 ngân sách khiến không rung nào vượt quá rung 1 (`cli.py:292`) |
| A5 | KHÔNG ĐẠT | 5 lần chạy mặc định (mnist14, fashion14, digits ×3): **0/15** đề xuất được nhận; val của đề xuất luôn thấp hơn, có lần rất xa (0,957→0,922; 0,809→0,655; 0,778→0,458; 0,986→0,870). lr đề xuất 0,013–0,056 so với lr đã tìm 0,001–0,003. Giai đoạn này tốn 39–50 % thời gian thực (100,9/261 s; 191,3/381 s; 53,4/130 s) mà chỉ chiếm 24/144 epoch. Top-1 không lần nào được ước lượng (hết ngân sách). Ngưỡng nhận `v − val > max(sd, val_std, 1e-9)` (`hparams.py:248`) thành 1e-9 khi chỉ có 1 seed, tức là không tính nhiễu |
| A6 | ĐẠT (đo được, kết quả xấu cho proxy) | Spearman với val ở rung 1 (n = 16), 3 bộ: grad_norm 0,67 / 0,73 / 0,56; snip 0,70 / 0,73 / 0,59; **params 0,70 / 0,60 / 0,78**; naswot −0,17 / −0,11 / 0,05; synflow −0,05 / 0,06 / 0,12; zico −0,15 / −0,13 / −0,31; vote 0,14 / 0,27 / 0,20; **seed_prior −0,35 / −0,37 / −0,25** (thứ tự mnist14 / fashion14 / digits). Chỉ grad_norm và snip nhỉnh hơn đếm tham số trên fashion14, ngang trên mnist14, thua trên digits. Prior tay **âm** trên cả 3 bộ |
| A7 | ĐẠT | `scripts/bench_surrogate.py --store dlnet/data/meta_image.jsonl` (28 s): Borda +0,668 / 0,0607 / 4/14; ZAP +0,536 / 0,0608 / 3/14; ZAP không φ +0,687 / 0,0745 / 5/14; kNN +0,553 / 0,0963 / 9/14; prior +0,192 / 0,1037; params +0,329 / 0,1084; random 0 / 0,175. Khớp `doc/algorithms.md` §3 tới chữ số thứ 2. Lưu ý: tổ hợp mà `design` thật sự dùng (Borda của prior và ZAP, `cli.py:262`, `surrogate.py:188-190`) **không** được đo trong bench; đo bổ sung (`scripts/lodo_design_combo.py`): Spearman +0,540, regret@1 0,048, hit@3 4/14, so với ZAP-Borda đơn thuần +0,659 / 0,051 / 2/14 |
| A8 | ĐẠT (có lưu ý) | digits 150ep: `--objective params --tolerance 0.01` → top-1 37 628 tham số, val 0,99155 = val tốt nhất (vgg cùng val nhưng 103 466 tham số đứng thứ 3), đúng định nghĩa ε-ràng buộc. `--objective pareto` → top-1 vgg (0,99155; 103 466) và top-2 res2net (0,8845; 2 043): cả hai là điểm biên có crowding vô hạn, đúng NSGA-II. Lưu ý: mặt Pareto và thứ tự trộn mô hình 4 và 12 epoch (xem N2) |
| A9 | KHÔNG ĐẠT (đã ghi số đo) | Trong 27 lần chạy 150ep: **0** ứng viên chạm rung 30 epoch; epoch tối đa của mọi top-1 là 12 (SH) hoặc 36 (một ứng viên duy nhất của HB). Mặc định (evolve 1): giai đoạn search chỉ hết rung 1 (64 ep), rồi "time budget reached"; thời gian không huấn luyện (load, reference, generate, proxies, seed) = 13–14 % thời gian thực; hparams 39–50 % thời gian thực và 17 % epoch, 0 lợi ích. HB: ngoặc 2 tiêu 36 ep mà không ứng viên nào tới R = 36 ⇒ bị loại khỏi xếp hạng cuối (30 % ngân sách search bị bỏ). Nhánh base không dùng 38/150 ep (25 %) vì driver bỏ hparams; ở chế độ mặc định phần này đi vào hparams |

## Kết quả các thí nghiệm A/B

Top-1 (theo val) của từng lần chạy, `--budget 150ep --threads 1 --seeds 16 --no-proxies`, hparams bỏ qua như nhau ở mọi nhánh. "epoch" = số epoch mà top-1 thực sự được huấn luyện.

FASHION_TABLE

## Phát hiện

### Nghiêm trọng

**N1. `--scheduler hyperband` sập với ngân sách epoch (KeyError 'val0').** `run_hyperband` (`dlnet/search.py:275`) mở ngoặc mới khi `clock.now() == deadline`, vì phép thử là `>` chứ không phải `>=`. Khi đó không ứng viên nào vừa ngân sách (`clock.fits` sai), `finished` rỗng, vòng rung `break`, rồi dòng log `alive[0]['val0']` truy cập ứng viên chưa huấn luyện. Nếu không sập, `done += full or alive[:1]` cũng sẽ đưa một ứng viên **chưa huấn luyện** vào xếp hạng. Tái hiện: `python -m dlnet design --data mnist14.npz --budget 150ep --threads 1 --no-cache --no-proxies --seeds 16 --seed 0 --scheduler hyperband --evolve 0 --out X` (log: `hb_crash_mnist14_s0.log`). Gặp ở 4/4 lần chạy (mnist14 s0/s1, digits s0/s1). Ngân sách epoch là chế độ được khuyến nghị cho tái lập, nên tùy chọn `--scheduler hyperband` thực tế không dùng được. Sửa: thử `>=`, và bỏ qua ngoặc không có ứng viên nào hoàn thành rung 0.

**N2. Xếp hạng cuối trộn mô hình huấn luyện số epoch khác nhau.** Kết quả của thế hệ 0 (dừng ở rung 1 = 4 epoch) và của tiến hóa (có lúc tới 12 epoch) được gộp (`cli.py:355`) rồi sắp theo `val_mean` (`search.assign_groups`, `hparams.refine`). Ví dụ lần chạy mặc định mnist14 (`pilot/`): top-1 `zoo_vgg_stage` có `epochs_used` 12, top-2 `vgg` có 4. Mặt Pareto ở A8 cũng trộn như vậy. Điều này trái với nguyên tắc ghi trong mã ("never rank different epoch counts", `search.py:193`). Hệ quả: tiến hóa được lợi vì được huấn luyện lâu hơn, và hạng top-1 bị lệch.

**N3. Với 150ep, SH không bao giờ đi quá rung 2.** Mọi top-1 chỉ được huấn luyện 4–12 epoch, và top-1 luôn chỉ có 1 lần chạy (std = 0), nên nhãn "group" và "most-stable" không có ý nghĩa. Ngân sách search = 0,8 × 150 = 120 ep (`cli.py:139`); khi bật tiến hóa, thế hệ 0 chỉ được 55 % (`cli.py:292`), tức 66 ep, đủ đúng một rung 4 epoch cho 16 ứng viên.

### Quan trọng

**Q1. Bộ chọn hạt giống không hơn ngẫu nhiên khi đo đầu-cuối (A2).** Trên LODO, chính prior tay có Spearman chỉ +0,19, và trong `design` thật nó **âm** (−0,25 tới −0,37 trên 3 bộ). Dù vậy, `seeds.select` vẫn cho prior trọng số 1,0, ngang với surrogate ZAP (`seeds.py:231-242`). Tổ hợp prior + ZAP có Spearman LODO thấp hơn ZAP đơn thuần (+0,54 so với +0,66). Ngoài ra, nhánh "random" cũng giữ neo resnet và trần 2 ứng viên mỗi họ (`seeds.py:243-266`), và log/leaderboard vẫn ghi `rankers ['prior', 'zap_surrogate']` dù thứ tự là ngẫu nhiên (báo cáo sai lệch).

**Q2. Ước lượng hyperparameter (A5) chỉ tốn chi phí, không đem lại lợi ích.** lr = lr_max/2, với lr_max là cực tiểu của loss khi quét 80 batch (`hparams.py:157-161`), quá lớn với AdamW. Lần nào đề xuất cũng thua; có đề xuất làm val tụt tới 0,11 (resnet, pilot). Giai đoạn này chiếm 39–50 % thời gian thực. Ngưỡng nhận không tính nhiễu khi `n_seeds = 1`.

**Q3. Tiến hóa (A4) không thêm giá trị ở ngân sách nhỏ.** 2 thế hệ × 8 con, mỗi con 4 epoch, với quần thể P ≈ 16, nên số chu kỳ C ≈ P. Hiệu ứng "aging" của Real et al. cần C ≫ P. Tiến hóa còn lấy 45 % ngân sách của thế hệ 0, khiến thế hệ 0 mất rung 12 epoch.

**Q4. SH giữ tối thiểu `2·topk`** (`search.py:195`), nên với nhóm nhỏ η thực tế là 1 ở rung 2: tốn 12 epoch cho cả 6 ứng viên mà không loại ai.

**Q5. Hyperband bỏ phí ngoặc chưa xong.** Chỉ cấu hình đạt R = 36 epoch mới vào xếp hạng cuối (`search.py:306-308`), nên ngoặc 2 (36 ep) bị bỏ hoàn toàn, và chỉ còn 1 finalist nên chỉ xuất được 1 mô hình (topk = 3). Ví dụ: digits s1 có ứng viên 12 epoch đạt val 0,9915, nhưng top-1 HB (36 epoch) chỉ 0,9803.

### Nên có

- Proxy (A6): naswot, synflow và zico không dự đoán được (|ρ| ≤ 0,31, thường âm); vote (0,14–0,27) kém đếm tham số. Giữ nguyên quy tắc "chỉ đo, không loại".
- NASWOT chỉ móc vào `nn.ReLU`, nên mạng dùng GELU/SiLU hoặc ReLU dạng hàm trả về None (n = 14/16 trên mnist14). Đây là giới hạn của mã gốc, nên ghi vào `doc/algorithms.md`.
- MnasNet: hệ số w = −0,07 được paper hiệu chỉnh cho **độ trễ** (gấp đôi độ trễ ≈ +5 % độ chính xác). DLNet dùng cho số tham số và FLOPs mà chưa hiệu chỉnh lại; nên ghi rõ, hoặc đo lại trên meta-dataset.
- Gradient noise scale đo ở chế độ train với BatchNorm: gradient theo batch 16 và batch 256 chịu thống kê BN khác nhau, nên giả thiết của McCandlish (cùng G, phương sai ∝ 1/B) bị lệch. Nên ghi là sai lệch.

## Đề xuất nâng cấp

| Đề xuất | Vì sao (bằng chứng) | Nguồn (paper / mã chuẩn) | Công sức S/M/L | Lợi ích kỳ vọng |
|---|---|---|---|---|
| Sửa Hyperband: `>=` deadline, bỏ ngoặc rỗng, cho ngoặc dở dang vào xếp hạng theo rung chung lớn nhất; thêm test với ngân sách epoch | N1, Q5: sập 4/4; chỉ 1 mô hình xuất | Li et al. arXiv:1603.06560 Alg. 1 | S | Dùng được `--scheduler hyperband` |
| Thay SH/HB đồng bộ bằng **ASHA** (thăng hạng bất đồng bộ, không cần ngoặc đầy đủ) | N3, Q4, Q5: ngân sách nhỏ không bao giờ chạm rung trên | Li et al., ASHA, arXiv:1810.05934 (PDF có ở `ref/optimazeDL/hp_asha_1810.05934.pdf`) | M | Không lãng phí ngoặc dở; luôn có ứng viên ở rung cao |
| Xếp hạng cuối chỉ trên cùng số epoch: huấn luyện tiếp (resume) top-K tới cùng rung trước khi so, hoặc so theo rung chung | N2: top-1 12 ep so với top-2 4 ep | nguyên tắc của SH (Jamieson & Talwalkar arXiv:1502.07943) | S | Thứ hạng công bằng; đánh giá tiến hóa không bị thiên lệch |
| Bỏ prior tay khỏi Borda (trọng số 0) hoặc học trọng số trên LODO; dùng ZAP-Borda + kNN | Q1: prior ρ âm trên 3/3 bộ; LODO tổ hợp 0,54 < 0,66 | ZAP arXiv:2206.08476; Feurer et al. AAAI 2015 | S | Bộ chọn hạt giống ít nhất không thua ngẫu nhiên |
| Thay LR range test bằng quy tắc chuẩn cho Adam (lr ≈ cực tiểu/10, hoặc "steepest slope"), hoặc bỏ giai đoạn hparams khi ngân sách < ~500 ep; nhận đề xuất chỉ khi `n_seeds ≥ 2` | Q2: 0/15 nhận, 39–50 % thời gian thực | Smith arXiv:1506.01186 §3.3; fastai `lr_find` (suggestion = min/10) | S | Lấy lại 40 % thời gian thực cho search |
| Thay tìm hyperparameter bằng **BOHB** hoặc **DEHB** kết hợp kiến trúc + lr/wd | A5 không đem lợi ích; HB/SH thiếu mô hình | Falkner et al. arXiv:1807.01774 (PDF có ở `ref/`); Awad et al. DEHB arXiv:2105.09821 | L | Học hyperparameter thay vì heuristic một bước |
| Tiến hóa chỉ bật khi ngân sách đủ để C ≫ P (ví dụ ≥ 600 ep), hoặc dùng mutation có định hướng (REA + predictor) | Q3: test −0,012 TB, 0/4 thắng | Real et al. arXiv:1802.01548 §Discussion; White et al. BANANAS arXiv:1910.11858 | M | Tránh mất rung 12 epoch ở ngân sách nhỏ |

## Giới hạn của lần kiểm định này

- Chỉ CPU, không GPU; máy bị chia sẻ, nên giây là số đo nhiễu (epoch thì không).
- Mỗi so sánh A/B: mnist14 và digits × seed 0/1 (đúng ngưỡng tối thiểu), cộng fashion14 seed 0. Bộ digits gần bão hòa (val 0,98–1,0), ít khả năng phân biệt. Với 4–5 cặp, chỉ phát hiện được chênh lệch lớn hơn khoảng 0,01–0,015; các kết luận "không hơn" nghĩa là "không phát hiện được lợi ích ở ngân sách 150ep", **không** khái quát cho ngân sách lớn.
- Ngân sách 150ep rất nhỏ so với cấu hình rung (4/12/30) và R = 36 của Hyperband; ở ngân sách lớn (≥ 600ep) kết quả A3/A4 có thể khác. Chưa đo do giới hạn thời gian.
- A2–A4 chạy với giai đoạn hparams bị bỏ qua (giống nhau ở mọi nhánh) và lớp chặn lỗi Hyperband trong driver của QA; mã repo không bị sửa.
- A1 đối chiếu 7 thuật toán bằng đọc mã so với PDF/mã gốc, chưa chạy lại test số học riêng.
