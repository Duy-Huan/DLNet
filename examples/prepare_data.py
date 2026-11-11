"""Download the standard datasets of examples/, check their sha256, write every case's input/.

    python examples/prepare_data.py --cache ~/.cache/dlnet_examples          # all cases
    python examples/prepare_data.py --cache DIR --only hsi/indian_pines camvid

Every archive is the file distributed by the dataset's publisher (or the mirror named in SOURCES)
and is checked against the sha256 recorded here, so input/ is reproducible byte for byte. Subsets are
deterministic (first N per class in the publisher's file order, or every image), documented per case
in examples/README.md. Needs numpy, scipy, Pillow.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import shutil
import tarfile
import urllib.request
from functools import partial
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
GH = "https://media.githubusercontent.com/media"
HSI_A = f"{GH}/hxlnwpu/Hyperspectral_image_classification/master/Data/matData"  # GIC mirror
HSI_B = "cu4oral6/hsi_datasets_unmixing/main"  # GIC mirror (Salinas, KSC, Botswana)

# name -> (url, sha256). HSI: files of the Grupo de Inteligencia Computacional (UPV/EHU) page
# "Hyperspectral Remote Sensing Scenes" (ehu.eus/ccwintco), via GitHub mirrors (ehu.eus is not
# reachable from CI); contents checked against the published sizes and per-class counts.
SOURCES = {
    "Indian_pines_corrected.mat": (f"{HSI_A}/Indian_pines_corrected.mat", "ec2f8808710919d566f70f0d4aa885aae1ddfd42b734aba71c5e12ca65450939"),
    "Indian_pines_gt.mat": (f"{HSI_A}/Indian_pines_gt.mat", "65c4687a8ab04f6da4789799bc3bc4f6e88bccac3ed6a2e6ae367e5e6b9e429c"),
    "PaviaU.mat": (f"{HSI_A}/PaviaU.mat", "28447fa87f7a5797845e9a189c0da85e23b1d06a4ba7361e5ff44efbf834d2fb"),
    "PaviaU_gt.mat": (f"{HSI_A}/PaviaU_gt.mat", "23f6a426928f9b32984adffe659e29f554f9fb6c93b5a107528d308d5087a829"),
    "Salinas_corrected.mat": (f"{GH}/{HSI_B}/Salinas/raw/Salinas_corrected.mat", "5ec1c0d22f56d18ecd336f8e35735863c0f160682e04e0c18ef3f89a3334d87d"),
    "Salinas_gt.mat": (f"https://raw.githubusercontent.com/{HSI_B}/Salinas/gt/Salinas_gt.mat", "ecfab4d31ef5553f097943235d8ea502038eb4a2067b2ad10b33e37c949955e2"),
    "KSC.mat": (f"{GH}/{HSI_B}/Kennedy_Space_Center/raw/KSC.mat", "b1ad011cfdb65c853e4f9f6108ca4774467d87f90a5c23b74ff3a2984a3b4786"),
    "KSC_gt.mat": (f"https://raw.githubusercontent.com/{HSI_B}/Kennedy_Space_Center/gt/KSC_gt.mat", "a1d6ab9293691006bd4d9742d1a1e1c141b1aaa5fbc5fa128b33c1d09038510b"),
    "Botswana.mat": (f"{GH}/{HSI_B}/Botswana/raw/Botswana.mat", "f1603903c844cdc2980550b0180688e8e1a72d4292595d1120e1dec2a80a91c7"),
    "Botswana_gt.mat": (f"https://raw.githubusercontent.com/{HSI_B}/Botswana/gt/Botswana_gt.mat", "668394905e10e629c16584bfd02b0f533b96d6ba18a63274a94ff3a77126a887"),
    # MNIST: the torchvision mirror of LeCun's files; Fashion-MNIST: Zalando's files on the Keras bucket
    "mnist-train-images-idx3-ubyte.gz": ("https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz", "440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609"),
    "mnist-train-labels-idx1-ubyte.gz": ("https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz", "3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c"),
    "fashion-train-images-idx3-ubyte.gz": ("https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-images-idx3-ubyte.gz", "3aede38d61863908ad78613f6a32ed271626dd12800ba2636569512369268a84"),
    "fashion-train-labels-idx1-ubyte.gz": ("https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-labels-idx1-ubyte.gz", "a04f17134ac03560a47e3764e11b92fc97de4d1bfaf8ba1a3aa29af54cc90845"),
    # fast.ai dataset bucket (fastai URLs.CIFAR, URLs.IMAGENETTE_160, URLs.PETS, URLs.CAMVID);
    # Imagenette is published there by its authors; Pets / CamVid are the original files repackaged
    "cifar10.tgz": ("https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz", "637c5814e11aefcb6ee76d5f59c67ddc8de7f5b5077502a195b0833d1e3e4441"),
    "imagenette2-160.tgz": ("https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz", "64d0c4859f35a461889e0147755a999a48b49bf38a7e0f9bd27003f10db02fe5"),
    "oxford-iiit-pet.tgz": ("https://s3.amazonaws.com/fast-ai-imageclas/oxford-iiit-pet.tgz", "3f1b76957366427dd5c12e3d48488be28dabd99a403bdb2b5100a83b9fd162d4"),
    "camvid.tgz": ("https://s3.amazonaws.com/fast-ai-imagelocal/camvid.tgz", "11db05fc3ee727fb17de7499380b20258a41beeb1002a2aee2c2244a472a4a45"),
}  # fmt: skip
# ISBI 2012 EM segmentation challenge training set (30 slices), as shipped by the U-Net Keras repo
ISBI = "https://raw.githubusercontent.com/zhixuhao/unet/master/data/membrane/train"
ISBI_SHA256 = (
    "a6036caf93768d8ec1c099fa735bf6e9fa33f9093c08f7e94ad3ba6945c3deac"  # image_0..29 then label_0..29, numeric order
)

# CamVid 32 -> the 11 classes of the SegNet benchmark (Brostow et al. 2009; Badrinarayanan et al.
# 2017); Void and anything unlisted -> ignored (-100)
CAMVID11 = {
    "Sky": ["Sky"],
    "Building": ["Building", "Archway", "Bridge", "Tunnel", "Wall"],
    "Pole": ["Column_Pole", "TrafficCone"],
    "Road": ["Road", "LaneMkgsDriv", "LaneMkgsNonDriv", "RoadShoulder"],
    "Sidewalk": ["Sidewalk", "ParkingBlock"],
    "Tree": ["Tree", "VegetationMisc"],
    "SignSymbol": ["SignSymbol", "Misc_Text", "TrafficLight"],
    "Fence": ["Fence"],
    "Car": ["Car", "SUVPickupTruck", "Truck_Bus", "Train", "OtherMoving"],
    "Pedestrian": ["Pedestrian", "Child", "CartLuggagePram", "Animal"],
    "Bicyclist": ["Bicyclist", "MotorcycleScooter"],
}


class Archive:
    """A .tgz read ONCE, front to back (random access re-decompresses from the start for every
    file: hours on an 800 MB archive). Keeps the files whose name passes `keep` in memory."""

    class _Member:
        def __init__(self, name):
            self.name = name

        def isfile(self):
            return True

    def __init__(self, path, keep=lambda name: True):
        self.files = {}
        with tarfile.open(path, "r|gz") as t:
            for m in t:
                if m.isfile() and keep(m.name):
                    self.files[m.name] = t.extractfile(m).read()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def getmembers(self):
        return [self._Member(n) for n in self.files]

    def extractfile(self, member):
        return io.BytesIO(self.files[getattr(member, "name", member)])


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(name: str, cache: Path) -> Path:
    url, digest = SOURCES[name]
    p = cache / name
    if not p.exists():
        print(f"download {url}")
        with urllib.request.urlopen(url) as r, open(p.with_suffix(".part"), "wb") as f:
            shutil.copyfileobj(r, f)
        p.with_suffix(".part").rename(p)
    got = sha256(p)
    if got != digest:
        raise SystemExit(f"{name}: sha256 {got} != {digest} (corrupt or changed upstream)")
    return p


def idx(path: Path) -> np.ndarray:
    with gzip.open(path) as f:
        data = f.read()
    ndim = data[3]
    shape = [int.from_bytes(data[4 + 4 * i : 8 + 4 * i], "big") for i in range(ndim)]
    return np.frombuffer(data, np.uint8, offset=4 + 4 * ndim).reshape(shape)


def first_per_class(y: np.ndarray, n: int) -> np.ndarray:
    """Indices of the first n samples of every class, in file order (deterministic subset)."""
    return np.sort(np.concatenate([np.flatnonzero(y == c)[:n] for c in np.unique(y)]))


def image(data: bytes, size: tuple[int, int], mode="RGB", resample=None) -> np.ndarray:
    from PIL import Image

    im = Image.open(io.BytesIO(data)).convert(mode)
    return np.asarray(im.resize(size, Image.BILINEAR if resample is None else resample))


def save_npz(case: str, X: np.ndarray, y: np.ndarray, classes: list[str], note: str) -> None:
    d = HERE / case / "input"
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / "data.npz", X=X, y=y)
    (d / "classes.txt").write_text("\n".join(classes) + "\n")
    (d / "SOURCE.md").write_text(
        note + f"\n\nArrays: X {X.shape} {X.dtype}, y {y.shape} {y.dtype}.\n"
    )
    print(f"{case}: X {X.shape} y {y.shape} -> {d / 'data.npz'}")


# ------------------------------------------------------------------------------------------- HSI
HSI_CASES = {
    "hsi/indian_pines": ("Indian_pines_corrected.mat", "Indian_pines_gt.mat"),
    "hsi/indian_pines_block": ("Indian_pines_corrected.mat", "Indian_pines_gt.mat"),
    "hsi/pavia_university": ("PaviaU.mat", "PaviaU_gt.mat"),
    "hsi/salinas": ("Salinas_corrected.mat", "Salinas_gt.mat"),
    "hsi/ksc": ("KSC.mat", "KSC_gt.mat"),
    "hsi/botswana": ("Botswana.mat", "Botswana_gt.mat"),
}


def hsi(case: str, cache: Path) -> None:
    d = HERE / case / "input"
    d.mkdir(parents=True, exist_ok=True)
    for name in HSI_CASES[case]:
        shutil.copyfile(fetch(name, cache), d / name)  # the publisher's file, unchanged
    url = {n: SOURCES[n][0] for n in HSI_CASES[case]}
    (d / "SOURCE.md").write_text(
        "Original .mat files (unchanged) of the GIC UPV/EHU 'Hyperspectral Remote Sensing Scenes' page;\n"
        + "\n".join(f"- `{n}`: {u} (sha256 `{SOURCES[n][1]}`)" for n, u in url.items())
        + "\n"
    )
    print(f"{case}: {', '.join(HSI_CASES[case])}")


# ------------------------------------------------------------------------- image classification
def mnist_like(case: str, prefix: str, classes: list[str], cache: Path, n=600) -> None:
    X = idx(fetch(f"{prefix}-train-images-idx3-ubyte.gz", cache))
    y = idx(fetch(f"{prefix}-train-labels-idx1-ubyte.gz", cache)).astype(np.int64)
    keep = first_per_class(y, n)
    save_npz(case, X[keep], y[keep], classes,
             f"First {n} images per class (file order) of the official TRAIN files "
             f"`{prefix}-train-*-ubyte.gz` ({SOURCES[prefix + '-train-images-idx3-ubyte.gz'][0]}).")  # fmt: skip


def folders(tgz: Path, split: str, size, n: int):
    """ImageFolder-style archive: <root>/<split>/<class>/<file>; first n files per class (sorted)."""
    with Archive(tgz, lambda n: f"/{split}/" in n) as t:
        members = sorted(
            (m for m in t.getmembers() if m.isfile() and f"/{split}/" in m.name and not m.name.split("/")[-1].startswith(".")),
            key=lambda m: m.name,
        )  # fmt: skip
        classes = sorted({m.name.split("/")[-2] for m in members})
        X, y = [], []
        for c, cls in enumerate(classes):
            for m in [m for m in members if m.name.split("/")[-2] == cls][:n]:
                X.append(image(t.extractfile(m).read(), size))
                y.append(c)
    return np.stack(X), np.array(y, np.int64), classes


def pets_list(t: tarfile.TarFile) -> list[tuple[str, int, str]]:
    rows = t.extractfile("oxford-iiit-pet/annotations/list.txt").read().decode().splitlines()
    out = []
    for r in rows:
        if r and not r.startswith("#"):
            name, cls = r.split()[:2]
            out.append((name, int(cls) - 1, name.rsplit("_", 1)[0]))
    return out


def pets_breeds(case: str, cache: Path, n=100, size=(64, 64)) -> None:
    with Archive(
        fetch("oxford-iiit-pet.tgz", cache), lambda n: n.endswith((".jpg", ".png", "list.txt"))
    ) as t:
        rows = pets_list(t)
        classes = [b for _, b in sorted({(c, b) for _, c, b in rows})]
        X, y = [], []
        for c in range(len(classes)):
            for name, _, _ in [r for r in rows if r[1] == c][:n]:
                X.append(image(t.extractfile(f"oxford-iiit-pet/images/{name}.jpg").read(), size))
                y.append(c)
    save_npz(case, np.stack(X), np.array(y, np.int64), classes,
             f"Oxford-IIIT Pet (Parkhi et al., CVPR 2012), 37 breeds: first {n} images per breed in "
             f"annotations/list.txt order, resized to {size[0]}x{size[1]} (bilinear). Archive: "
             f"{SOURCES['oxford-iiit-pet.tgz'][0]}.")  # fmt: skip


# --------------------------------------------------------------------------- image segmentation
def isbi(case: str, cache: Path, size=(256, 256)) -> None:
    from PIL import Image

    raw = {}
    for kind in ("image", "label"):
        for i in range(30):
            p = cache / "isbi" / f"{kind}_{i}.png"
            if not p.exists():
                p.parent.mkdir(parents=True, exist_ok=True)
                urllib.request.urlretrieve(f"{ISBI}/{kind}/{i}.png", p)
            raw[kind, i] = p.read_bytes()
    got = hashlib.sha256(
        b"".join(raw["image", i] for i in range(30)) + b"".join(raw["label", i] for i in range(30))
    ).hexdigest()
    if got != ISBI_SHA256:
        raise SystemExit(f"ISBI 2012: sha256 {got} != {ISBI_SHA256}")
    X = np.stack([image(raw["image", i], size, "L") for i in range(30)])
    lab = np.stack([image(raw["label", i], size, "L", Image.NEAREST) for i in range(30)])
    y = (lab > 127).astype(np.int64)  # 0 = membrane (black), 1 = cell interior (white)
    save_npz(case, X, y, ["membrane", "cell"],
             f"ISBI 2012 EM segmentation challenge (Arganda-Carreras et al., Front. Neuroanat. 2015), "
             f"the 30 training slices of the Drosophila ssTEM stack, 512x512 -> {size[0]}x{size[1]} "
             f"(bilinear image, nearest label). Files: {ISBI}/(image|label)/<i>.png")  # fmt: skip


def pets_trimap(case: str, cache: Path, n=40, size=(96, 96)) -> None:
    from PIL import Image

    with Archive(
        fetch("oxford-iiit-pet.tgz", cache), lambda n: n.endswith((".jpg", ".png", "list.txt"))
    ) as t:
        rows = pets_list(t)
        X, y = [], []
        for c in range(37):
            for name, _, _ in [r for r in rows if r[1] == c][:n]:
                X.append(image(t.extractfile(f"oxford-iiit-pet/images/{name}.jpg").read(), size))
                tri = image(
                    t.extractfile(f"oxford-iiit-pet/annotations/trimaps/{name}.png").read(),
                    size,
                    "L",
                    Image.NEAREST,
                )
                y.append(
                    np.select([tri == 1, tri == 2, tri == 3], [1, 0, 2], -100)
                )  # pet, background, border
    save_npz(case, np.stack(X), np.stack(y).astype(np.int64), ["background", "pet", "border"],
             f"Oxford-IIIT Pet trimaps (1 pet, 2 background, 3 border -> 1, 0, 2): first {n} images "
             f"of each of the 37 breeds, {size[0]}x{size[1]} (bilinear image, nearest trimap). "
             f"Archive: {SOURCES['oxford-iiit-pet.tgz'][0]}.")  # fmt: skip


def camvid(case: str, cache: Path, size=(120, 90)) -> None:
    from PIL import Image

    with Archive(fetch("camvid.tgz", cache)) as t:
        codes = t.extractfile("camvid/codes.txt").read().decode().split()
        lut = np.full(256, -100, np.int64)
        for k, (_, members) in enumerate(CAMVID11.items()):
            for m in members:
                lut[codes.index(m)] = k
        names = sorted(
            m.name for m in t.getmembers() if m.name.startswith("camvid/images/") and m.isfile()
        )
        X, y = [], []
        for name in names:
            stem = name.split("/")[-1][:-4]
            X.append(image(t.extractfile(name).read(), size))
            lab = image(
                t.extractfile(f"camvid/labels/{stem}_P.png").read(), size, "L", Image.NEAREST
            )
            y.append(lut[lab])
    save_npz(case, np.stack(X), np.stack(y), list(CAMVID11),
             f"CamVid (Brostow et al., PRL 2009), all {len(names)} labelled frames, 960x720 -> "
             f"{size[0]}x{size[1]} (bilinear image, nearest label); the 32 classes grouped into the 11 of "
             f"the SegNet benchmark (prepare_data.py CAMVID11), Void -> -100 (ignored). Archive: "
             f"{SOURCES['camvid.tgz'][0]}.")  # fmt: skip


CASES = {
    **{c: partial(hsi, c) for c in HSI_CASES},
    "image_classification/mnist": lambda cache: mnist_like("image_classification/mnist", "mnist", [str(i) for i in range(10)], cache),
    "image_classification/fashion_mnist": lambda cache: mnist_like("image_classification/fashion_mnist", "fashion",
        ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"], cache),
    "image_classification/cifar10": lambda cache: save_npz("image_classification/cifar10", *folders(fetch("cifar10.tgz", cache), "train", (32, 32), 600),
        "CIFAR-10 (Krizhevsky 2009), first 600 images per class (sorted file names) of the TRAIN split, 32x32 RGB. "
        f"Archive: {SOURCES['cifar10.tgz'][0]} (the official images as PNG folders)."),
    "image_classification/imagenette": lambda cache: save_npz("image_classification/imagenette", *folders(fetch("imagenette2-160.tgz", cache), "train", (64, 64), 300),
        "Imagenette v2 (fast.ai; 10 easy ImageNet classes), 160 px version: first 300 images per class of the TRAIN "
        f"split, resized to 64x64 RGB. Archive: {SOURCES['imagenette2-160.tgz'][0]}."),
    "image_classification/oxford_pets_breeds": lambda cache: pets_breeds("image_classification/oxford_pets_breeds", cache),
    "image_segmentation/isbi2012_em": lambda cache: isbi("image_segmentation/isbi2012_em", cache),
    "image_segmentation/oxford_pets_trimap": lambda cache: pets_trimap("image_segmentation/oxford_pets_trimap", cache),
    "image_segmentation/camvid": lambda cache: camvid("image_segmentation/camvid", cache),
}  # fmt: skip


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="download folder (archives are kept for reruns)")
    ap.add_argument("--only", nargs="+", help="case names or substrings")
    a = ap.parse_args()
    cache = Path(a.cache).expanduser()
    cache.mkdir(parents=True, exist_ok=True)
    for case, make in CASES.items():
        if not a.only or any(s in case for s in a.only):
            make(cache)
