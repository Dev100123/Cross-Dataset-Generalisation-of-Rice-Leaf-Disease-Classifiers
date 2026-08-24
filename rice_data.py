"""
rice_data.py

Reads manifest.csv from prep_rice_cv.py.

Dataset A (dhan_shomadhan) = 5-fold CV. For run k:
    test  = fold k
    val   = fold (k+1) % 5
    train = the other 3 folds
Dataset B (bact_fungal) = fixed train/val/test.

Every split is cluster-safe: the same leaf never crosses train/val/test.
"""

import csv
import random
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

MANIFEST = Path(r"/ibiscostorage/rchandraghosh/genera/prepared_cv/manifest.csv")

CLASSES = ["brown_spot", "blast", "leaf_scald", "sheath_blight"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}

N_FOLDS = 5
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4          # set to 0 if the Windows DataLoader hangs

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

TRAIN_TF = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

EVAL_TF = transforms.Compose([
    transforms.Resize(int(IMG_SIZE * 1.14)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# ----------------------------------------------------------------------------

def load_manifest(path=MANIFEST):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            r["fold"] = int(r["fold"])
            r["cluster"] = int(r["cluster"])
            rows.append(r)
    if not rows:
        raise SystemExit(f"manifest empty: {path}")
    return rows


class RiceDataset(Dataset):
    def __init__(self, rows, transform):
        self.rows = rows
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = Image.open(r["path"]).convert("RGB")
        return self.transform(img), CLS2IDX[r["cls"]]


def select(rows, dataset, run=None):
    """(train_rows, val_rows, test_rows) for one dataset."""
    rows = [r for r in rows if r["dataset"] == dataset]
    if not rows:
        raise SystemExit(f"no rows for dataset '{dataset}'")

    if rows[0]["split"]:                       # dataset B: fixed split
        return ([r for r in rows if r["split"] == "train"],
                [r for r in rows if r["split"] == "val"],
                [r for r in rows if r["split"] == "test"])

    if run is None:                            # dataset A: rotate folds
        raise SystemExit(f"'{dataset}' uses {N_FOLDS}-fold CV. Pass run=0..{N_FOLDS-1}.")
    te_f, va_f = run % N_FOLDS, (run + 1) % N_FOLDS
    return ([r for r in rows if r["fold"] not in (te_f, va_f)],
            [r for r in rows if r["fold"] == va_f],
            [r for r in rows if r["fold"] == te_f])


def subsample(rows, pct, seed=0):
    """
    Few-shot budget. Keep `pct` of rows, stratified by class, whole clusters only.
    Always keeps at least 1 cluster per class.
    """
    if pct >= 1.0:
        return rows
    rng = random.Random(seed)

    by_cls_cluster = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_cls_cluster[r["cls"]][r["cluster"]].append(r)

    out = []
    for cls, clusters in by_cls_cluster.items():
        cids = list(clusters.keys())
        rng.shuffle(cids)
        n_keep = max(1, int(round(len(cids) * pct)))
        for cid in cids[:n_keep]:
            out.extend(clusters[cid])
    return out


def check_leak(*groups):
    """check_leak(('train', tr), ('val', va), ('test', te))"""
    names = [g[0] for g in groups]
    sets = [{r["cluster"] for r in g[1]} for g in groups]
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            shared = sets[i] & sets[j]
            if shared:
                raise SystemExit(f"LEAK: {len(shared)} clusters in both {names[i]} and {names[j]}")


def make_loader(rows, train, batch_size=BATCH_SIZE, workers=NUM_WORKERS):
    if not rows:
        raise SystemExit("make_loader got 0 rows")
    return DataLoader(
        RiceDataset(rows, TRAIN_TF if train else EVAL_TF),
        batch_size=batch_size, shuffle=train,
        num_workers=workers, pin_memory=torch.cuda.is_available(),
        drop_last=(train and len(rows) > batch_size),
    )


def class_weights(rows):
    """Inverse-frequency weights for CrossEntropyLoss. Your classes are imbalanced."""
    counts = defaultdict(int)
    for r in rows:
        counts[CLS2IDX[r["cls"]]] += 1
    total = sum(counts.values())
    return torch.tensor(
        [total / (len(CLASSES) * max(counts.get(i, 0), 1)) for i in range(len(CLASSES))],
        dtype=torch.float,
    )


if __name__ == "__main__":
    rows = load_manifest()
    print(f"manifest: {len(rows)} rows\n")
    for ds in ("bact_fungal", "dhan_shomadhan"):
        for run in ([None] if ds == "bact_fungal" else range(N_FOLDS)):
            tr, va, te = select(rows, ds, run)
            check_leak(("train", tr), ("val", va), ("test", te))
            tag = ds if run is None else f"{ds} run={run}"
            per = defaultdict(int)
            for r in te:
                per[r["cls"]] += 1
            print(f"{tag:26s} train={len(tr):4d} val={len(va):3d} test={len(te):3d}"
                  f"  test/class={dict(sorted(per.items()))}")
    print("\nno leaks.")
