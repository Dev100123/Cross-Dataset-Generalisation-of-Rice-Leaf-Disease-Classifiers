"""
prep_rice_cv.py

Prepares the two rice datasets for a cross-dataset generalization study.

  Dataset A (dhan_shomadhan, 261 imgs) -> 5-fold CV   (too small for a fixed test set)
  Dataset B (bact_fungal,   1032 imgs) -> fixed train/val/test

Key points:
  * Splits by DUPLICATE CLUSTER, not by file. Same leaf never lands in train and test.
  * Stratified by class in every fold.
  * Writes a manifest.csv instead of copying images 5 times. Saves disk, no data moved.
  * Automatically dumps duplicate contact sheets so you can eyeball DUP_THRESHOLD.

Install:
    pip install pillow imagehash tqdm scikit-learn numpy
Run:
    python prep_rice_cv.py
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
import imagehash
from tqdm import tqdm
from sklearn.model_selection import StratifiedGroupKFold

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------

A_ROOT = Path(r"C:\Users\rajib\Desktop\dataset\Dhan-Shomadhan\Field Background")
B_ROOT = Path(r"C:\Users\rajib\Desktop\dataset\Rice Leaf Bacterial and Fungal Disease Dataset\Original")
OUT_ROOT = Path(r"C:\Users\rajib\Desktop\dataset\prepared_cv")

A_NAME = "dhan_shomadhan"
B_NAME = "bact_fungal"

N_FOLDS = 5          # for dataset A
B_SPLITS = 7         # for dataset B: 1 fold test, 1 fold val, 5 folds train (~71/14/14)
SEED = 42

# >>> THE NUMBER THAT DECIDES EVERYTHING <<<
# Look at OUT_ROOT/dup_check/*.jpg after the first run.
#   same leaf in a sheet  -> threshold is correct
#   different leaves      -> lower to 4, then 2, and re-run
DUP_THRESHOLD = 8

DROP_LABEL_CONFLICTS = True   # drop clusters where the same leaf has 2 different diseases
MAKE_DUP_SHEETS = True
N_DUP_SHEETS = 40

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

CLASS_MAP = {
    "brown spot": "brown_spot", "browon spot": "brown_spot", "brown_spot": "brown_spot",
    "leaf blast": "blast", "rice blast": "blast", "blast": "blast",
    "leaf scaled": "leaf_scald", "leaf scald": "leaf_scald", "leaf_scald": "leaf_scald",
    "sheath blight": "sheath_blight", "steath blight": "sheath_blight",
    "rice sheath blight": "sheath_blight", "sheath_blight": "sheath_blight",
}

CLASSES = ["brown_spot", "blast", "leaf_scald", "sheath_blight"]


# ----------------------------------------------------------------------------
# scan
# ----------------------------------------------------------------------------

def scan(root: Path, ds_name: str):
    if not root.exists():
        raise SystemExit(f"ERROR: path does not exist:\n  {root}")

    records, unmapped = [], set()
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        canon = CLASS_MAP.get(class_dir.name.strip().lower())
        if canon is None:
            unmapped.add(class_dir.name)
            continue
        for p in sorted(class_dir.rglob("*")):
            if p.is_file() and p.suffix.lower() in IMG_EXT:
                records.append({"path": p, "dataset": ds_name, "cls": canon})

    if unmapped:
        print(f"  [!] IGNORED folders in {ds_name} (not in CLASS_MAP): {sorted(unmapped)}")
    return records


# ----------------------------------------------------------------------------
# dedup
# ----------------------------------------------------------------------------

class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def hash_all(records):
    ok = []
    for r in tqdm(records, desc="  hashing", unit="img"):
        try:
            with Image.open(r["path"]) as im:
                r["phash"] = imagehash.phash(im.convert("RGB"))
            ok.append(r)
        except Exception as e:
            print(f"  [!] unreadable, skipped: {r['path']} ({e})")
    return ok


def cluster_duplicates(records, threshold):
    n = len(records)
    uf = UnionFind(n)
    hashes = [r["phash"] for r in records]
    for i in tqdm(range(n), desc="  clustering", unit="img"):
        hi = hashes[i]
        for j in range(i + 1, n):
            if hi - hashes[j] <= threshold:
                uf.union(i, j)
    for i, r in enumerate(records):
        r["cluster"] = uf.find(i)
    return records


def dup_sheets(records, out_dir, limit):
    """Write one jpg per duplicate cluster so you can SEE whether they are the same leaf."""
    out_dir.mkdir(parents=True, exist_ok=True)
    groups = defaultdict(list)
    for r in records:
        groups[r["cluster"]].append(r)
    multi = sorted((g for g in groups.values() if len(g) > 1),
                   key=len, reverse=True)[:limit]

    for i, g in enumerate(multi):
        ims = []
        for r in g[:6]:
            try:
                with Image.open(r["path"]) as im:
                    ims.append(im.convert("RGB").resize((200, 200)))
            except Exception:
                pass
        if not ims:
            continue
        sheet = Image.new("RGB", (200 * len(ims), 200), (20, 20, 20))
        for j, im in enumerate(ims):
            sheet.paste(im, (200 * j, 0))
        ds = "_".join(sorted({r["dataset"] for r in g}))
        cl = "_".join(sorted({r["cls"] for r in g}))
        sheet.save(out_dir / f"n{len(g):02d}_{ds}_{cl}_{i:03d}.jpg")

    print(f"  wrote {len(multi)} duplicate sheets -> {out_dir}")
    print(f"  >>> OPEN THEM. Same leaf = threshold ok. Different leaves = lower DUP_THRESHOLD.")


def find_label_conflicts(records):
    groups = defaultdict(list)
    for r in records:
        groups[r["cluster"]].append(r)
    conflicts = []
    for cid, g in groups.items():
        labels = {x["cls"] for x in g}
        if len(labels) > 1:
            conflicts.append({"cluster": cid, "labels": sorted(labels),
                              "files": [str(x["path"]) for x in g]})
    return conflicts


# ----------------------------------------------------------------------------
# splitting
# ----------------------------------------------------------------------------

def assign_folds_A(records, n_folds, seed):
    """Stratified by class, grouped by cluster. Each image gets fold 0..n_folds-1."""
    recs = [r for r in records if r["dataset"] == A_NAME]
    y = np.array([CLASSES.index(r["cls"]) for r in recs])
    g = np.array([r["cluster"] for r in recs])
    X = np.zeros(len(recs))

    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold_id, (_, test_idx) in enumerate(sgkf.split(X, y, groups=g)):
        for i in test_idx:
            recs[i]["fold"] = fold_id
            recs[i]["split"] = ""
    return recs


def assign_split_B(records, n_splits, seed):
    """fold 0 -> test, fold 1 -> val, rest -> train. Stratified + grouped."""
    recs = [r for r in records if r["dataset"] == B_NAME]
    y = np.array([CLASSES.index(r["cls"]) for r in recs])
    g = np.array([r["cluster"] for r in recs])
    X = np.zeros(len(recs))

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold_id, (_, test_idx) in enumerate(sgkf.split(X, y, groups=g)):
        name = "test" if fold_id == 0 else "val" if fold_id == 1 else "train"
        for i in test_idx:
            recs[i]["split"] = name
            recs[i]["fold"] = -1
    return recs


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    print("\n=== 1. INVENTORY ===")
    a = scan(A_ROOT, A_NAME)
    b = scan(B_ROOT, B_NAME)
    for name, recs in ((A_NAME, a), (B_NAME, b)):
        per = defaultdict(int)
        for r in recs:
            per[r["cls"]] += 1
        print(f"  {name}: {len(recs)} images  {dict(sorted(per.items()))}")

    print("\n=== 2. DEDUP ===")
    allrecs = hash_all(a + b)
    allrecs = cluster_duplicates(allrecs, DUP_THRESHOLD)

    n_clusters = len({r["cluster"] for r in allrecs})
    print(f"  {len(allrecs)} images -> {n_clusters} unique clusters (threshold={DUP_THRESHOLD})")

    if MAKE_DUP_SHEETS:
        dup_sheets(allrecs, OUT_ROOT / "dup_check", N_DUP_SHEETS)

    conflicts = find_label_conflicts(allrecs)
    if conflicts:
        print(f"  [!!] {len(conflicts)} clusters have CONFLICTING labels")
        if DROP_LABEL_CONFLICTS:
            bad = {c["cluster"] for c in conflicts}
            before = len(allrecs)
            allrecs = [r for r in allrecs if r["cluster"] not in bad]
            print(f"       dropped {before - len(allrecs)} images from those clusters")

    print("\n=== 3. FOLDS ===")
    a_recs = assign_folds_A(allrecs, N_FOLDS, SEED)
    b_recs = assign_split_B(allrecs, B_SPLITS, SEED)

    print(f"  {A_NAME}: {N_FOLDS}-fold CV")
    for f in range(N_FOLDS):
        per = defaultdict(int)
        for r in a_recs:
            if r["fold"] == f:
                per[r["cls"]] += 1
        print(f"    fold {f}: n={sum(per.values()):3d}  {dict(sorted(per.items()))}")

    print(f"  {B_NAME}: fixed split")
    for sp in ("train", "val", "test"):
        per = defaultdict(int)
        for r in b_recs:
            if r["split"] == sp:
                per[r["cls"]] += 1
        print(f"    {sp:5s}: n={sum(per.values()):4d}  {dict(sorted(per.items()))}")

    print("\n=== 4. MANIFEST ===")
    man = OUT_ROOT / "manifest.csv"
    with open(man, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "dataset", "cls", "cluster", "fold", "split"])
        for r in a_recs + b_recs:
            w.writerow([str(r["path"]), r["dataset"], r["cls"],
                        r["cluster"], r.get("fold", -1), r.get("split", "")])

    report = {
        "seed": SEED, "n_folds": N_FOLDS, "dup_threshold": DUP_THRESHOLD,
        "classes": CLASSES,
        "source_paths": {A_NAME: str(A_ROOT), B_NAME: str(B_ROOT)},
        "n_images": len(allrecs), "n_clusters": n_clusters,
        "label_conflicts": conflicts[:50],
        "dropped_conflicts": DROP_LABEL_CONFLICTS,
    }
    (OUT_ROOT / "report.json").write_text(json.dumps(report, indent=2, default=str))

    print(f"  manifest -> {man}")
    print(f"  report   -> {OUT_ROOT / 'report.json'}")
    print(f"\nNEXT: open {OUT_ROOT / 'dup_check'} and check the sheets before you train.")


if __name__ == "__main__":
    main()
