"""End-to-end QA/QC run of one example case: what a user does with DLNet, step by step, checked.

    python examples/run_example.py hsi/indian_pines [--budget 10m] [--epochs 30] [--threads 2]
    python examples/run_example.py all

Layout: <case>/input (data), <case>/output (what DLNet writes: leaderboard + Top-K PyTorch packages,
the top-1 one trained: model.pt, model.pt2), <case>/qc (logs of every step, QC.md, qc.json, pictures).
Steps:
  1. design   python -m dlnet design --data input/... --out output/ (search + Top-K packages)
  2. tests    pytest of every exported package (its own tests/)
  3. train    python train.py of the top-1 package: full training, writes model.pt
  4. predict  python predict.py on the input file, --export model.pt2 (torch.export graph)
  5. QC       predictions of step 4 re-scored on the TEST split the package used (rebuilt from
              its config), compared with what train.py printed; reference baselines; disjoint
              splits; pictures (HSI class map, segmentation overlays, confusion matrix).
Needs the cases' input/ (examples/prepare_data.py) and Pillow.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

HSI = {"plugin": "hsi", "patch": 9}
CASES = {
    "hsi/indian_pines": {**HSI, "data": "Indian_pines_corrected.mat", "gt": "Indian_pines_gt.mat", "args": ["--split", "pixel", "--pca", "30"]},
    "hsi/indian_pines_block": {**HSI, "data": "Indian_pines_corrected.mat", "gt": "Indian_pines_gt.mat", "args": ["--split", "block", "--pca", "30"]},
    "hsi/pavia_university": {**HSI, "data": "PaviaU.mat", "gt": "PaviaU_gt.mat", "args": ["--split", "pixel", "--pca", "30", "--frac", "0.02,0.02,0.96"]},
    "hsi/salinas": {**HSI, "data": "Salinas_corrected.mat", "gt": "Salinas_gt.mat", "args": ["--split", "pixel", "--pca", "30", "--frac", "0.02,0.02,0.96"]},
    "hsi/ksc": {**HSI, "data": "KSC.mat", "gt": "KSC_gt.mat", "args": ["--split", "pixel", "--pca", "30"]},
    "hsi/botswana": {**HSI, "data": "Botswana.mat", "gt": "Botswana_gt.mat", "args": ["--split", "pixel", "--pca", "30"]},
    **{f"image_classification/{c}": {"plugin": "image_classification", "data": "data.npz", "args": []}
       for c in ("mnist", "fashion_mnist", "cifar10", "imagenette", "oxford_pets_breeds")},
    **{f"image_segmentation/{c}": {"plugin": "image_segmentation", "data": "data.npz", "args": []}
       for c in ("isbi2012_em", "oxford_pets_trimap", "camvid")},
}  # fmt: skip

PALETTE = np.array(
    [[0, 0, 0], [230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48],
     [145, 30, 180], [70, 240, 240], [240, 50, 230], [210, 245, 60], [250, 190, 212], [0, 128, 128],
     [220, 190, 255], [170, 110, 40], [255, 250, 200], [128, 0, 0], [170, 255, 195], [128, 128, 0],
     [255, 215, 180], [0, 0, 128], [128, 128, 128]], np.uint8)  # fmt: skip


def sh(cmd, cwd, log: Path, env=None):
    t0 = time.time()
    with open(log, "w") as f:
        f.write("$ " + " ".join(map(str, cmd)) + "\n")
        f.flush()
        r = subprocess.run(
            list(map(str, cmd)), cwd=cwd, stdout=f, stderr=subprocess.STDOUT, env=env, check=False
        )
    return r.returncode, round(time.time() - t0, 1)


def metrics_cls(y, p, k):
    cm = np.zeros((k, k), np.int64)
    np.add.at(cm, (y, p), 1)
    n = cm.sum()
    oa = np.trace(cm) / n
    recall = np.diag(cm) / np.maximum(cm.sum(1), 1)
    pe = (cm.sum(0) * cm.sum(1)).sum() / n**2
    return {"accuracy": float(oa), "balanced_accuracy": float(recall[cm.sum(1) > 0].mean()),
            "kappa": float((oa - pe) / (1 - pe)), "per_class_recall": recall.round(4).tolist()}, cm  # fmt: skip


def miou(y, p, k):
    m = y >= 0
    inter = np.bincount(y[m] * k + p[m], minlength=k * k).reshape(k, k)
    tp = np.diag(inter)
    union = inter.sum(0) + inter.sum(1) - tp
    present = inter.sum(1) > 0
    iou = tp / np.maximum(union, 1)
    return {"miou": float(iou[present].mean()), "pixel_accuracy": float(tp.sum() / max(m.sum(), 1)),
            "per_class_iou": [round(float(v), 4) if c else None for v, c in zip(iou, present)]}  # fmt: skip


def save_png(arr, path, scale=1):
    from PIL import Image

    im = Image.fromarray(arr)
    if scale != 1:
        im = im.resize((im.width * scale, im.height * scale), Image.NEAREST)
    im.save(path)


def to_rgb(x):
    """(H, W) or (H, W, C) image of any range -> uint8 RGB for a picture."""
    x = np.asarray(x, np.float64)
    if x.ndim == 2:
        x = np.repeat(x[..., None], 3, -1)
    lo, hi = np.percentile(x, 1), np.percentile(x, 99)
    return (np.clip((x - lo) / max(hi - lo, 1e-9), 0, 1) * 255).astype(np.uint8)


def colour(mask):
    return PALETTE[np.where(mask < 0, 0, mask % (len(PALETTE) - 1) + 1)]


def confusion_png(cm, path):
    z = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    img = (255 - z * 255).astype(np.uint8)
    save_png(np.repeat(img[..., None], 3, -1), path, scale=max(4, 240 // len(cm)))


# ------------------------------------------------------------------------------ QC per domain
def qc_hsi(case, pkg, cfg, inp, out):
    from dlnet import hsi_data

    dc = cfg["data"]
    cube, gt = hsi_data.load(str(inp / case["data"]), str(inp / case["gt"]), None, None)
    pred = np.load(out / "predictions.npy")
    rows, cols, labels, sid, keep, _ = hsi_data.make_split(
        gt, dc["split"], cfg["split_seed"], dc["frac"], dc["patch"], dc["block"]
    )
    classes = np.unique(gt[gt > 0])
    te = (sid == 2) & keep
    y = labels[te]
    p = np.searchsorted(classes, pred[rows[te], cols[te]])
    m, cm = metrics_cls(y, p, len(classes))
    confusion_png(cm, out / "confusion_test.png")
    s = max(1, 400 // max(gt.shape))
    save_png(colour(np.where(gt > 0, gt - 1, -1)), out / "map_ground_truth.png", s)
    save_png(colour(np.searchsorted(classes, pred)), out / "map_predicted_all_pixels.png", s)
    save_png(
        colour(np.where(gt > 0, np.searchsorted(classes, pred), -1)),
        out / "map_predicted_labelled.png",
        s,
    )
    split_map = np.full(gt.shape, -1)
    split_map[rows[keep], cols[keep]] = sid[keep]
    save_png(colour(split_map), out / "map_split_train_val_test.png", s)
    sets = [set(zip(rows[(sid == i) & keep], cols[(sid == i) & keep])) for i in range(3)]
    disjoint = not (sets[0] & sets[2] or sets[0] & sets[1] or sets[1] & sets[2])
    return (
        m,
        "accuracy",
        disjoint,
        {"n_test": int(te.sum()), "cube": list(cube.shape), "classes": len(classes)},
    )


def split_of(case, cfg, y):
    from dlnet import core

    if case["plugin"] == "image_segmentation":
        perm = np.random.default_rng(cfg["split_seed"]).permutation(len(y))
        a, b = int(0.6 * len(y)), int(0.8 * len(y))
        return perm[:a], perm[a:b], perm[b:]
    return core.stratified_split(y, "classification", cfg["split_seed"])


def qc_array(case, pkg, cfg, inp, out):
    z = np.load(inp / case["data"])
    X, y = z["X"], z["y"]
    pred = np.load(out / "predictions.npy")
    tr, va, te = split_of(case, cfg, y)
    disjoint = not (set(tr) & set(te) or set(tr) & set(va) or set(va) & set(te))
    names = (inp / "classes.txt").read_text().split("\n")
    if case["plugin"] == "image_segmentation":
        m = miou(y[te].ravel(), pred[te].ravel(), len(names) - 1)
        rows = []
        for i in te[:6]:
            img = to_rgb(X[i])
            rows.append(np.concatenate([img, colour(y[i]), colour(pred[i])], 1))
        save_png(np.concatenate(rows, 0), out / "test_overlays_image_truth_prediction.png", 2)
        return m, "miou", disjoint, {"n_test_images": len(te), "image": list(X.shape[1:])}
    k = len(names) - 1
    m, cm = metrics_cls(y[te], pred[te], k)
    confusion_png(cm, out / "confusion_test.png")
    grid = [to_rgb(X[i]) for i in te[:32]]
    save_png(
        np.concatenate(
            [np.concatenate(grid[r * 8 : r * 8 + 8], 1) for r in range(len(grid) // 8)], 0
        ),
        out / "test_samples.png",
        2,
    )
    return m, "accuracy", disjoint, {"n_test": len(te), "image": list(X.shape[1:]), "classes": k}


def first_float(pattern, text):
    m = re.search(pattern, text)
    return float(m.group(1)) if m else None


def run(name, budget, epochs, threads):
    case = CASES[name]
    base = HERE / name
    inp, out, qc_dir = base / "input", base / "output", base / "qc"
    if not (inp / case["data"]).exists():
        raise SystemExit(f"{inp / case['data']} missing: run examples/prepare_data.py first")
    qc_dir.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "OMP_NUM_THREADS": str(threads),
        "MKL_NUM_THREADS": str(threads),
        "PYTHONPATH": str(ROOT),
    }
    data = ["--data", inp / case["data"]] + (["--gt", inp / case["gt"]] if "gt" in case else [])
    steps = {}

    rc, t = sh([sys.executable, "-m", "dlnet", "design", *data, "--plugin", case["plugin"], "--out", out,
                "--budget", budget, "--topk", 3, "--seed", 0, "--threads", threads, "--no-cache", *case["args"]],
               ROOT, qc_dir / "1_design.log", env)  # fmt: skip
    steps["1_design"] = {"ok": rc == 0, "seconds": t}
    pkgs = sorted(p for p in (out.iterdir() if out.exists() else []) if p.is_dir() and p.name.startswith("top"))
    steps["1_design"]["packages"] = [p.name for p in pkgs]
    if rc or not pkgs:
        return finish(name, qc_dir, steps, None)

    ok = True
    for p in pkgs:
        rc, t = sh(
            [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"],
            p,
            qc_dir / f"2_tests_{p.name}.log",
            env,
        )
        ok &= rc == 0
    steps["2_package_tests"] = {"ok": ok}

    top1 = next(p for p in pkgs if p.name.startswith("top1_"))
    rc, t = sh(
        [sys.executable, "train.py", *data, "--epochs", epochs, "--seed", 0],
        top1,
        qc_dir / "3_train_top1.log",
        env,
    )
    log = (qc_dir / "3_train_top1.log").read_text()
    steps["3_train_top1"] = {"ok": rc == 0 and (top1 / "model.pt").exists(), "seconds": t, "epochs": epochs,
                             "package": top1.name, "weights_kb": round((top1 / "model.pt").stat().st_size / 1024) if (top1 / "model.pt").exists() else None}  # fmt: skip
    if not steps["3_train_top1"]["ok"]:
        return finish(name, qc_dir, steps, None)

    rc, t = sh([sys.executable, "predict.py", "--data", inp / case["data"], "--out", qc_dir / "predictions.npy", "--export", top1 / "model.pt2"],
               top1, qc_dir / "4_predict_top1.log", env)  # fmt: skip
    steps["4_predict_export"] = {
        "ok": rc == 0 and (qc_dir / "predictions.npy").exists() and (top1 / "model.pt2").exists(),
        "seconds": t,
    }
    if not steps["4_predict_export"]["ok"]:
        return finish(name, qc_dir, steps, None)

    cfg = json.loads((top1 / "config.yaml").read_text())
    qc = qc_hsi if case["plugin"] == "hsi" else qc_array
    m, key, disjoint, extra = qc(case, top1, cfg, inp, qc_dir)
    test_line = re.search(r"^test\s*:\s*(\{.*\})", log, re.MULTILINE)
    printed = (
        first_float(rf"'{key}': (?:np\.float64\()?([0-9.]+)", test_line.group(1))
        if test_line
        else None
    )
    lb = json.loads((out / "leaderboard.json").read_text())
    steps["5_qc"] = {"ok": disjoint and printed is not None and abs(printed - m[key]) < 2e-3,
                     "metric": key, "test_from_predictions": round(m[key], 4), "test_printed_by_train_py": printed,
                     "splits_disjoint": disjoint, **extra, "metrics": m}  # fmt: skip
    return finish(name, qc_dir, steps, lb)


def finish(name, out, steps, lb):
    verdict = (lb or {}).get("verdict", {})
    ref = (lb or {}).get("reference", [])
    res = {"case": name, "steps": steps, "verdict": verdict, "reference": ref,
           "top": [{k: r.get(k) for k in ("rank", "name", "params", "val_mean", "val_std", "test", "group")} for r in (lb or {}).get("top_k", [])]}  # fmt: skip
    (out / "qc.json").write_text(json.dumps(res, indent=1, default=str))
    lines = [f"# QA/QC: `{name}`", "", "| Bước | Kết quả | Chi tiết |", "|---|---|---|"]
    for k, v in steps.items():
        detail = ", ".join(f"{a}={b}" for a, b in v.items() if a not in ("ok", "metrics"))
        lines.append(f"| {k} | {'ĐẠT' if v.get('ok') else 'KHÔNG ĐẠT'} | {detail} |")
    if "5_qc" in steps:
        m = steps["5_qc"]["metrics"]
        lines += ["", "Chỉ số trên tập TEST (tính lại từ predictions.npy của gói top-1 đã train đầy đủ):", "",
                  "```", json.dumps(m, indent=1), "```"]  # fmt: skip
    if verdict:
        lines += [
            "",
            f"So với baseline cổ điển cùng split (leaderboard): `{json.dumps(verdict, default=str)}`",
        ]
    lines += [
        "",
        "Xem `leaderboard.md` (xếp hạng tìm kiếm), các thư mục `top*_*/` (gói PyTorch), các log `1_..5_`.",
    ]
    (out / "QC.md").write_text("\n".join(lines) + "\n")
    print(
        f"[{name}] " + " ".join(f"{k}={'OK' if v.get('ok') else 'FAIL'}" for k, v in steps.items()),
        flush=True,
    )
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "case", help="a case name, 'all', or a domain prefix (hsi, image_classification, ...)"
    )
    ap.add_argument("--budget", default="10m")
    ap.add_argument("--epochs", type=int, default=30, help="full training of the top-1 package")
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    names = [c for c in CASES if a.case == "all" or c == a.case or c.startswith(a.case + "/")]
    for n in names:
        run(n, a.budget, a.epochs, a.threads)
