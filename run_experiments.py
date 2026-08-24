"""
run_experiments.py  (v3 - multi-backbone)

Same cross-dataset design as before, now looped over several backbones so you can
show whether the domain gap is architecture-independent.

Backbones come from models.py (build_model): resnet50, vit, convnext_tiny,
convnext_large, segformer_v0 (MiT-B0), segformer_v5 (MiT-B5).

Everything else is unchanged:
  * both datasets k-fold (in-domain ceiling has error bars)
  * few-shot budget sweep
  * multiple seeds
  * full metric suite (accuracy, P/R/F1 macro+weighted, per-class, confusion)
  * checkpoint / resume (now keyed by MODEL too)

COMPUTE WARNING
---------------
The grid is per-backbone. Running all budgets x all seeds for all 6 models is
thousands of trainings. The DEFAULT below is the sane staged plan:
    Stage 1 (this default): ALL models, but only budget=100%  -> the architecture
             comparison table. ~756 trainings.
    Stage 2: pick the winner, add it to FULL_SWEEP_MODELS, rerun. Resume skips
             everything already done and only fills the missing budgets.

Run:
    pip install torch torchvision timm transformers scikit-learn numpy pillow
    python models.py            # verify all backbones build
    python run_experiments.py   # smoke test first (see CONFIG)
"""

import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score,
                             precision_recall_fscore_support,
                             precision_score, recall_score)
from sklearn.model_selection import StratifiedGroupKFold

from rice_data import (CLASSES, CLS2IDX, class_weights, load_manifest,
                       make_loader, subsample)
from rice_model import adabn, fit                      # generic; work on any ClsNet
from models import ALL_MODELS, build_model, has_bn

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------

OUT = Path("results")
CKPT = OUT / "ckpt"

A = "dhan_shomadhan"     # small
B = "bact_fungal"        # large

# which backbones to run
MODELS = ["vit", "convnext_tiny", "convnext_large",
          "segformer_v0", "segformer_v5"]

# These models get the full few-shot budget sweep; the rest run at 100% only
# (that gives the architecture-comparison table + one full few-shot curve in a
# single launch). segformer_v0 (MiT-B0) is the cheapest, so it carries the curve.
# Add more names here if you want their curves too.
FULL_SWEEP_MODELS = {"segformer_v0"}

SEEDS = [32]                      # single seed; error bars are fold-only (5 per number)
BUDGETS = [0.01, 0.05, 0.10, 0.25, 0.50, 1.00]
N_FOLDS = 5
FOLDS = list(range(N_FOLDS))
EPOCHS = {A: 250, B: 80}

PATIENCE = {A: 30, B: 12}
# per-(model,dataset) batch. Big models get smaller batches to fit 32GB.
def batch_for(model, ds):
    big = model in ("convnext_large", "segformer_v5")
    base = 16 if ds == A else 32
    return base // 2 if big else base

LR = 1e-4
VERBOSE = True

AGG = ["accuracy", "balanced_accuracy",
       "precision_macro", "recall_macro", "f1_macro",
       "precision_weighted", "recall_weighted", "f1_weighted"]

TRANSFER = [(B, A), (A, B)]


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def set_seed(s):
    import random
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def build_folds(rows, seed):
    y = np.array([CLS2IDX[r["cls"]] for r in rows])
    g = np.array([r["cluster"] for r in rows])
    X = np.zeros(len(rows))
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    fold_of = {}
    for fid, (_, test_idx) in enumerate(sgkf.split(X, y, groups=g)):
        for i in test_idx:
            fold_of[int(i)] = fid
    return fold_of


def split_for_fold(rows, fold_of, k):
    va = (k + 1) % N_FOLDS
    tr = [rows[i] for i, f in fold_of.items() if f not in (k, va)]
    vl = [rows[i] for i, f in fold_of.items() if f == va]
    te = [rows[i] for i, f in fold_of.items() if f == k]
    return tr, vl, te


def no_leak(train, *others):
    ct = {r["cluster"] for r in train}
    for name, grp in others:
        bad = ct & {r["cluster"] for r in grp}
        if bad:
            raise SystemExit(f"LEAK: {len(bad)} clusters in train and {name}")


@torch.no_grad()
def predict(model, loader):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        x = x.to("cuda" if torch.cuda.is_available() else "cpu", non_blocking=True)
        with torch.autocast("cuda", enabled=torch.cuda.is_available()):
            out = model(x)
        ps.append(out.float().argmax(1).cpu().numpy()); ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def metrics(y, p):
    labels = list(range(len(CLASSES)))
    pr, rc, f1, sup = precision_recall_fscore_support(y, p, labels=labels, zero_division=0)
    agg = {
        "accuracy": accuracy_score(y, p),
        "balanced_accuracy": balanced_accuracy_score(y, p),
        "precision_macro": precision_score(y, p, average="macro", zero_division=0),
        "recall_macro": recall_score(y, p, average="macro", zero_division=0),
        "f1_macro": f1_score(y, p, average="macro", zero_division=0),
        "precision_weighted": precision_score(y, p, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y, p, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y, p, average="weighted", zero_division=0),
    }
    per_class = {c: {"precision": float(pr[i]), "recall": float(rc[i]),
                     "f1": float(f1[i]), "support": int(sup[i])}
                 for i, c in enumerate(CLASSES)}
    cm = confusion_matrix(y, p, labels=labels).tolist()
    return {k: float(v) for k, v in agg.items()}, per_class, cm


# ----------------------------------------------------------------------------
# checkpoint / resume  (now keyed by model)
# ----------------------------------------------------------------------------

COLS = (["model", "direction", "regime", "budget", "fold", "seed", "n_test", "n_train_used"]
        + AGG + ["per_class", "confusion"])

def model_csv(mname):
    return OUT / mname / "results.csv"

def load_done(root):
    """Scan every results/<model>/results.csv so resume works across all models."""
    done = set()
    for path in root.glob("*/results.csv"):
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                done.add((r["model"], r["direction"], r["regime"],
                          f'{float(r["budget"]):.4f}', str(r["fold"]), str(r["seed"])))
    return done

def key(model, direction, regime, budget, fold, seed):
    return (model, direction, regime, f"{float(budget):.4f}", str(fold), str(seed))

def append_row(path, row):
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


# ----------------------------------------------------------------------------
# training units  (model-aware)
# ----------------------------------------------------------------------------

def train_from_scratch(mname, train, val, ds):
    model = build_model(mname, pretrained=True)
    model.set_trainable("full")
    bs = batch_for(mname, ds)
    model, vf1 = fit(model,
                     make_loader(train, True, batch_size=bs),
                     make_loader(val, False, batch_size=bs),
                     epochs=EPOCHS[ds], lr=LR,
                     weights=class_weights(train), patience=PATIENCE[ds],
                     verbose=VERBOSE)
    return model, vf1


def source_model(mname, rows, fold_of, ds, seed):
    p = CKPT / f"source_{mname}_{ds}_seed{seed}.pt"
    if p.exists():
        model = build_model(mname, pretrained=False)
        model.load_state_dict(torch.load(p, map_location="cpu"))
        model.to("cuda" if torch.cuda.is_available() else "cpu")
        print(f"    [cached] {p.name}")
        return {k: v.clone() for k, v in model.state_dict().items()}
    tr = [rows[i] for i, f in fold_of.items() if f != 0]
    vl = [rows[i] for i, f in fold_of.items() if f == 0]
    set_seed(seed)
    model, vf1 = train_from_scratch(mname, tr, vl, ds)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), p)
    print(f"    trained {p.name}  val_f1={vf1:.4f}")
    return {k: v.clone() for k, v in model.state_dict().items()}


def load_state(mname, state):
    model = build_model(mname, pretrained=False)
    model.load_state_dict(state)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    return model


def adapt(mname, src_state, train, val, ds, mode, seed):
    model = load_state(mname, src_state)
    model.set_trainable(mode)
    set_seed(seed)
    lr = 1e-3 if mode == "head" else LR
    bs = batch_for(mname, ds)
    model, _ = fit(model,
                   make_loader(train, True, batch_size=bs),
                   make_loader(val, False, batch_size=bs),
                   epochs=EPOCHS[ds], lr=lr,
                   weights=class_weights(train), patience=PATIENCE[ds],
                   verbose=VERBOSE)
    return model


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def budgets_for(mname):
    return BUDGETS if mname in FULL_SWEEP_MODELS else [1.00]


def main(models=None, full_sweep=False, budgets=None):
    global MODELS, BUDGETS, FULL_SWEEP_MODELS
    if models:
        bad = [m for m in models if m not in ALL_MODELS]
        if bad:
            raise SystemExit(f"unknown model(s) {bad}. options: {ALL_MODELS}")
        MODELS = models
    if budgets:                       # explicit budget list -> sweep for these models
        BUDGETS = budgets
        FULL_SWEEP_MODELS = set(MODELS)
    elif full_sweep:                  # all 6 default budgets
        FULL_SWEEP_MODELS = set(MODELS)
    else:                             # 100% only (architecture comparison)
        FULL_SWEEP_MODELS = set()

    from models import model_info
    OUT.mkdir(exist_ok=True); CKPT.mkdir(parents=True, exist_ok=True)
    done = load_done(OUT)
    rows_all = load_manifest()
    DATA = {ds: [r for r in rows_all if r["dataset"] == ds] for ds in (A, B)}

    # rough plan
    n = 0
    for mname in MODELS:
        nb = len(budgets_for(mname))
        n += len(SEEDS) * len(FOLDS) * 2                       # in-domain
        n += len(SEEDS) * 2                                    # source
        n += len(SEEDS) * len(FOLDS) * 2 * 3 * nb              # adaptations
    print(f"device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"models={MODELS}")
    print(f"full-sweep models={sorted(FULL_SWEEP_MODELS) or '(none: 100% budget only)'}")
    print(f"seeds={SEEDS} folds={FOLDS} budgets={BUDGETS}")
    print(f"planned trainings across all models: ~{n}  (finished cells skipped)")
    print(f"output: {OUT}/<model>/   ({len(done)} cells already recorded)\n")

    t0 = time.time()
    folds = {ds: {s: build_folds(DATA[ds], s) for s in SEEDS} for ds in (A, B)}

    def emit(mname, direction, regime, budget, fold, seed, y, p, n_used):
        agg, pc, cm = metrics(y, p)
        row = {"model": mname, "direction": direction, "regime": regime,
               "budget": budget, "fold": fold, "seed": seed, "n_test": int(len(y)),
               "n_train_used": n_used, **agg,
               "per_class": json.dumps(pc), "confusion": json.dumps(cm)}
        append_row(model_csv(mname), row)
        done.add(key(mname, direction, regime, budget, fold, seed))
        print(f"  [{mname:14s}] {direction:26s} {regime:11s} b={budget:<4} f{fold} s{seed}  "
              f"F1m={agg['f1_macro']:.4f} acc={agg['accuracy']:.4f}")

    for mname in MODELS:
        print(f"\n{'#'*74}\n# MODEL: {mname}\n{'#'*74}")
        BUD = budgets_for(mname)
        # write model_info.json (size + pretrained weights) into the model folder
        (OUT / mname).mkdir(parents=True, exist_ok=True)
        info = model_info(mname)
        (OUT / mname / "model_info.json").write_text(json.dumps(info, indent=2))
        print(f"  {info['params_M']}M params · {info['size_mb_fp32']}MB fp32 · "
              f"{info['pretrained_checkpoint']} ({info['pretrain_data']})")

        for seed in SEEDS:
            print(f"\n{'='*66}\n{mname} | SEED {seed}\n{'='*66}")

            # ---------- in-domain CV ceilings ----------
            for ds in (A, B):
                for k in FOLDS:
                    if key(mname, f"{ds}->{ds}", "in_domain", 1.0, k, seed) in done:
                        continue
                    tr, vl, te = split_for_fold(DATA[ds], folds[ds][seed], k)
                    no_leak(tr, ("val", vl), ("test", te))
                    set_seed(seed)
                    model, _ = train_from_scratch(mname, tr, vl, ds)
                    y, p = predict(model, make_loader(te, False, batch_size=batch_for(mname, ds)))
                    emit(mname, f"{ds}->{ds}", "in_domain", 1.0, k, seed, y, p, len(tr))
                    del model; torch.cuda.empty_cache()

            # ---------- source models ----------
            state = {}
            for ds in (A, B):
                print(f"  source model: {mname} on {ds}")
                state[ds] = source_model(mname, DATA[ds], folds[ds][seed], ds, seed)

            # ---------- transfer ----------
            for src, tgt in TRANSFER:
                direction = f"{src}->{tgt}"
                for k in FOLDS:
                    tr_pool, vl, te = split_for_fold(DATA[tgt], folds[tgt][seed], k)
                    te_loader = make_loader(te, False, batch_size=batch_for(mname, tgt))

                    if key(mname, direction, "zero_shot", 0.0, k, seed) not in done:
                        m = load_state(mname, state[src])
                        y, p = predict(m, te_loader)
                        emit(mname, direction, "zero_shot", 0.0, k, seed, y, p, 0)
                        del m; torch.cuda.empty_cache()

                    # AdaBN only meaningful for BatchNorm nets (resnet50)
                    if key(mname, direction, "adabn", 0.0, k, seed) not in done:
                        m = load_state(mname, state[src])
                        if has_bn(m):
                            m = adabn(m, make_loader(tr_pool, False, batch_size=batch_for(mname, tgt)))
                            y, p = predict(m, te_loader)
                            emit(mname, direction, "adabn", 0.0, k, seed, y, p, 0)
                        del m; torch.cuda.empty_cache()

                    for budget in BUD:
                        sub = subsample(tr_pool, budget, seed=seed)
                        no_leak(sub, ("val", vl), ("test", te))
                        for mode in ("head", "encoder", "full"):
                            if key(mname, direction, f"ft_{mode}", budget, k, seed) in done:
                                continue
                            m = adapt(mname, state[src], sub, vl, tgt, mode, seed)
                            y, p = predict(m, te_loader)
                            emit(mname, direction, f"ft_{mode}", budget, k, seed, y, p, len(sub))
                            del m; torch.cuda.empty_cache()

    summarize(OUT)
    print(f"\ntotal time this session: {(time.time()-t0)/60:.1f} min")


# ----------------------------------------------------------------------------

def summarize(root):
    """Read every results/<model>/results.csv; write a per-model summary.txt in
    each folder and one combined summary_all.txt at the top level."""
    all_rows = []
    for path in sorted(root.glob("*/results.csv")):
        rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
        all_rows += rows
        # per-model summary
        agg = defaultdict(list)
        for r in rows:
            agg[(r["direction"], r["regime"], f'{float(r["budget"]):.2f}')].append(float(r["f1_macro"]))
        lines = [f"{'direction':26s} {'regime':11s} {'bud':>5s} {'f1_macro':>20s} {'n':>4s}", "-" * 72]
        for (d, rg, b), v in sorted(agg.items()):
            v = np.array(v)
            lines.append(f"{d:26s} {rg:11s} {b:>5s} {v.mean():>11.4f} +- {v.std():.4f} {len(v):4d}")
        (path.parent / "summary.txt").write_text("\n".join(lines), encoding="utf-8")

    if not all_rows:
        return
    agg = defaultdict(list)
    for r in all_rows:
        agg[(r["model"], r["direction"], r["regime"], f'{float(r["budget"]):.2f}')].append(float(r["f1_macro"]))
    lines = [f"{'model':15s} {'direction':26s} {'regime':11s} {'bud':>5s} {'f1_macro':>20s} {'n':>4s}",
             "-" * 88]
    for (mn, d, rg, b), v in sorted(agg.items()):
        v = np.array(v)
        lines.append(f"{mn:15s} {d:26s} {rg:11s} {b:>5s} {v.mean():>11.4f} +- {v.std():.4f} {len(v):4d}")
    txt = "\n".join(lines)
    print("\n" + "=" * 88 + "\nSUMMARY (macro-F1, all models)\n" + "=" * 88)
    print(txt)
    (root / "summary_all.txt").write_text(txt, encoding="utf-8")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="cross-dataset multi-backbone runner")
    ap.add_argument("--models", nargs="+", default=None,
                    help=f"models to run (default: all). options: {ALL_MODELS}")
    ap.add_argument("--full-sweep", action="store_true",
                    help="run all 6 budgets (few-shot curve). default: 100%% only")
    ap.add_argument("--budgets", nargs="+", type=float, default=None,
                    help="custom budgets, e.g. --budgets 0.01 0.05 0.1 0.25 0.5 1.0")
    a = ap.parse_args()
    main(models=a.models, full_sweep=a.full_sweep, budgets=a.budgets)