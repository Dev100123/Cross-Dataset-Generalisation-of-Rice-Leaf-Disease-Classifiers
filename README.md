# Cross-Dataset Generalisation of Rice Leaf Disease Classifiers

Code for the paper **“Cross-Dataset Generalisation of Rice Leaf Disease Classifiers: Evaluating
Intra-Crop Domain Shift and Target-Domain Adaptation”**
Rajib Chandra Ghosh, Semanto Mondal, Alberto Moccardi, Flora Amato — DIETI, University of Naples Federico II.

Deep models for rice leaf disease classification routinely report >90% accuracy, but almost always on a
held-out split of the *same* dataset. This repository contains the full pipeline for a leakage-controlled,
**bidirectional cross-dataset** study on two independently collected Bangladeshi rice datasets that share
exactly four disease classes (brown spot, blast, leaf scald, sheath blight).

Headline results: in-domain macro-F1 is 0.87–0.91 on SIP and 0.66–0.71 on DHA, **zero-shot transfer
collapses below the 0.25 chance level for all five backbones**, model capacity does not help, and supervised
adaptation of the *encoder* recovers most of the loss — much of it from a small fraction of target labels.

---

## 1. What is in this repository

| File | Purpose |
|---|---|
| `prep_rice_cv.py` | Scans both dataset roots, canonicalises class names, runs perceptual-hash de-duplication, clusters near-duplicates, drops label-conflicting clusters, assigns grouped + class-stratified folds, and writes `manifest.csv`. **Run this first.** |
| `rice_data.py` | Manifest loading, `Dataset`/`DataLoader` construction, train/eval transforms, cluster-safe leakage checks, few-shot `subsample()`, inverse-frequency class weights. |
| `models.py` | One `build_model(name)` factory for every backbone, with a uniform `set_trainable("full"/"encoder"/"head")` API and `model_info()` (params, checkpoint id). |
| `rice_model.py` | The training engine: `fit()` (AdamW + cosine schedule + AMP + early stopping on val macro-F1 + best-weight restore), `evaluate()`, and `adabn()`. |
| `run_experiments.py` | The main experiment runner. In-domain CV, source-model training, zero-shot, linear probe / encoder-FT / full-FT, few-shot budget sweep. Resume-aware. |
| `train_one.py` | Convenience wrapper: run one backbone at the 100% budget only (the architecture-comparison row). |
| `run_budget_sweep.py` | Convenience wrapper: run one backbone across all label budgets (the few-shot curve). |
| `make_paper_outputs.py` | Reads `results/*/results.csv` and emits the LaTeX tables and PDF/PNG figures used in the paper. |

Outputs are written to `results/<model>/results.csv` (one row per direction × regime × budget × fold × seed,
including per-class metrics and confusion matrices as JSON) and to `paper/` for the tables and figures.

---

## 2. Datasets

Both datasets are public and are **not** redistributed here.

| Short name | Source | Subset used | Region |
|---|---|---|---|
| **DHA** | Dhan-Shomadhan (Hossain et al., 2021), [10.17632/znsxdctwtt.1](https://doi.org/10.17632/znsxdctwtt.1) | *Field Background* | Dhaka Division |
| **SIP** | Rice Leaf Bacterial and Fungal Disease (Hasan et al., 2023), [10.17632/hx6f852hw4.2](https://doi.org/10.17632/hx6f852hw4.2) | *Original* (non-augmented) | Sirajganj–Pabna |

Only the four classes shared by both taxonomies are kept; all other classes are discarded.
Class-name spelling differences between the two sources are normalised by `CLASS_MAP` in `prep_rice_cv.py`.

Expected layout for each root:

```
<dataset root>/
  Brown Spot/      *.jpg
  Rice Blast/      *.jpg
  Leaf Scald/      *.jpg
  Sheath Blight/   *.jpg
```

---

## 3. Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

A CUDA GPU is strongly recommended. The reported results were produced on a single **NVIDIA V100 (32 GB)**;
batch sizes in `run_experiments.py` are tuned for that memory budget.

---

## 4. Reproducing the study

### Step 1 — prepare the data

Edit the three paths at the top of `prep_rice_cv.py` (`A_ROOT`, `B_ROOT`, `OUT_ROOT`), then:

```bash
python prep_rice_cv.py
```

This writes `<OUT_ROOT>/manifest.csv`, `<OUT_ROOT>/report.json`, and `<OUT_ROOT>/dup_check/*.jpg`.

**Inspect `dup_check/` before training.** Each sheet shows one near-duplicate cluster. If the images in a
sheet are the same leaf, `DUP_THRESHOLD` (default: pHash Hamming distance ≤ 8) is correct; if they are
different leaves, lower it to 4 or 2 and re-run. Clusters whose members carry conflicting labels are dropped
(`DROP_LABEL_CONFLICTS`), and the count is recorded in `report.json`.

### Step 2 — point the loader at the manifest

Set `MANIFEST` at the top of `rice_data.py` to the `manifest.csv` produced above.

### Step 3 — sanity checks

```bash
python rice_data.py      # prints split sizes, asserts no cluster crosses train/val/test
python models.py         # builds every backbone, checks the freeze modes
```

### Step 4 — the architecture comparison (100% target labels)

```bash
python train_one.py segformer_v0     # MiT-B0
python train_one.py convnext_tiny    # ConvNeXt-T
python train_one.py segformer_v5     # MiT-B5
python train_one.py vit              # ViT-B/16
python train_one.py convnext_large   # ConvNeXt-L
```

Every cell is checkpointed to `results/<model>/results.csv`; re-running skips finished cells, so the jobs are
safe to kill and restart.

### Step 5 — the few-shot label-budget curve

```bash
python run_budget_sweep.py segformer_v0   # 1%, 5%, 10%, 25%, 50%, 100%
```

The 100% cells from step 4 are reused; only the missing budgets are computed.

### Step 6 — tables and figures

```bash
python make_paper_outputs.py
```

---

## 5. Evaluation protocol

**Leakage control.** Images are hashed with a 64-bit perceptual hash and transitively grouped by
Hamming distance ≤ 8 using union-find, so near-duplicate shots of the same leaf form one *cluster*.
All splitting is done with `StratifiedGroupKFold` over these clusters, so the same leaf can never appear in
both training and test. `check_leak()` / `no_leak()` re-assert this at run time and abort on violation.

**Folds.** Both datasets use 5-fold cross-validation (seed 32). In each iteration one fold is the test set,
one is the validation set (early stopping and best-weight selection only), and the remaining three are
training data. Every fold serves as test exactly once; all reported numbers are mean ± std over the five folds.

**Backbones.** All five are ImageNet-1k pretrained — pretraining data is pinned deliberately so that the
capacity comparison is not confounded by in21k/in22k checkpoints.

| Name in code | Paper name | Checkpoint | Params |
|---|---|---|---|
| `segformer_v0` | MiT-B0 | `nvidia/mit-b0` | 3.3 M |
| `convnext_tiny` | ConvNeXt-T | `convnext_tiny.fb_in1k` | 27.8 M |
| `segformer_v5` | MiT-B5 | `nvidia/mit-b5` | 81.5 M |
| `vit` | ViT-B/16 | `vit_base_patch16_224.augreg_in1k` | 85.8 M |
| `convnext_large` | ConvNeXt-L | `convnext_large.fb_in1k` | 196.2 M |

**Regimes**, evaluated in both directions (SIP→DHA and DHA→SIP):

| Regime in CSV | Meaning |
|---|---|
| `in_domain` | Trained and tested on the same dataset — the upper reference. |
| `zero_shot` | Source-trained model applied to the target test fold, no adaptation. |
| `ft_head` | Linear probe: encoder frozen, classifier head trained (lr 1e-3). |
| `ft_encoder` | Encoder fine-tuned, classifier head frozen. |
| `ft_full` | All parameters fine-tuned. |
| `adabn` | BatchNorm-statistic recalibration on target images. Emitted **only** for BatchNorm backbones; see §7. |

**Training configuration** (identical across architectures unless noted):

| | DHA | SIP |
|---|---|---|
| Train / val / test images per fold | ≈157 / 52 / 52 | ≈612 / 204 / 204 |
| Max epochs | 250 | 80 |
| Early-stopping patience | 30 | 12 |
| Batch size (standard backbones) | 16 | 32 |
| Batch size (ConvNeXt-L, MiT-B5) | 8 | 16 |

Optimiser AdamW, weight decay 1e-4, cosine annealing, lr 1e-4 (1e-3 for linear probing), class-weighted
cross-entropy with inverse-frequency weights, label smoothing 0.05, AMP throughout, model selection on
validation macro-F1.

**Preprocessing** — held constant across all backbones (`rice_data.py`), 224×224 ImageNet-normalised input:

* train: `RandomResizedCrop(224, scale=(0.7, 1.0))`, horizontal flip, vertical flip, `ColorJitter(0.2, 0.2, 0.2, 0.05)`
* eval: resize to 256, centre crop 224

---

## 6. Reported results

Macro-F1, mean over five folds. Chance level for four balanced classes is 0.25.

| Model | Params | In-domain DHA | In-domain SIP | ZS SIP→DHA | ZS DHA→SIP | Full-FT SIP→DHA | Full-FT DHA→SIP |
|---|---|---|---|---|---|---|---|
| MiT-B0 | 3.3 M | 0.656 | 0.872 | 0.097 | 0.186 | 0.658 | 0.896 |
| ConvNeXt-T | 27.8 M | 0.665 | 0.902 | 0.117 | 0.189 | 0.642 | 0.896 |
| MiT-B5 | 81.5 M | 0.713 | 0.902 | 0.146 | 0.180 | 0.700 | 0.877 |
| ViT-B/16 | 85.8 M | 0.672 | 0.900 | 0.143 | 0.162 | 0.693 | 0.894 |
| ConvNeXt-L | 196.2 M | 0.691 | 0.909 | 0.202 | 0.212 | 0.674 | 0.912 |

Label efficiency (MiT-B0, full fine-tuning, macro-F1):

| Target labels | SIP→DHA | DHA→SIP |
|---|---|---|
| 0% (zero-shot) | 0.097 | 0.186 |
| 1% | 0.312 | 0.301 |
| 5% | 0.380 | 0.540 |
| 10% | 0.459 | 0.616 |
| 25% | 0.500 | 0.760 |
| 50% | 0.525 | 0.854 |
| 100% | 0.658 | 0.896 |
| In-domain ceiling | 0.656 | 0.872 |

---

## 7. Notes, scope, and known caveats

Read this section before drawing conclusions from the code.

* **Single seed.** All reported runs use seed 32. The ± values are the spread over the five
  cross-validation folds, **not** over random restarts. `make_paper_outputs.py` prints a warning when fewer
  than three seeds are present. Widen `SEEDS` in `run_experiments.py` to add seed variance.
* **The label budget applies to the training pool only.** `subsample()` reduces the target *training* rows;
  the validation fold used for early stopping and best-weight selection is **not** subsampled. A run labelled
  “1%” therefore still consults a full-size labelled validation fold. `n_train_used` in `results.csv` records
  the true number of training images behind every budget point.
* **Budgets are fractions of clusters, not of images.** `subsample()` keeps whole duplicate clusters and always
  retains at least one cluster per class, so a nominal 1% is a lower bound rather than an exact image count.
  Use `n_train_used` for the exact figure.
* **De-duplication is run jointly over both datasets**, so a cluster can in principle span DHA and SIP.
  Splitting is then performed per dataset. Check `manifest.csv` for cross-dataset clusters if this matters
  for your use of the code:
  ```bash
  python -c "import csv,collections; d=collections.defaultdict(set); [d[r['cluster']].add(r['dataset']) for r in csv.DictReader(open('manifest.csv'))]; print(sum(len(v)>1 for v in d.values()), 'clusters span both datasets')"
  ```
* **`ResNet-50` and `AdaBN` are present in the code but are not part of the five reported backbones.**
  `models.py` can build `resnet50`, `rice_model.py` contains a standalone ResNet-50 builder, and `adabn()`
  recalibrates BatchNorm statistics on unlabelled target images. All five reported backbones use
  LayerNorm, so `has_bn()` is false for them and the `adabn` regime is skipped — those cells are simply
  absent from the results. These paths are exploratory leftovers, kept for completeness.
* **`train_from_scratch()` is a misnomer.** It builds with `pretrained=True` and fine-tunes end to end from
  ImageNet-1k weights. Nothing in the study is trained from random initialisation.
* **`rice_data.select()` implements an alternative protocol for SIP** (the fixed train/val/test split written
  by `prep_rice_cv.py`'s `assign_split_B`). The reported experiments do **not** use it — `run_experiments.py`
  rebuilds 5-fold cross-validation for both datasets via `build_folds()`, and the manifest's `split` column is
  unused there. `rice_data.py`'s `__main__` block is a standalone sanity check.
* **Paths are hard-coded.** `prep_rice_cv.py` and `rice_data.py` carry absolute paths from the machines the
  study was run on. Edit them before running.
* **De-duplication is O(n²)** in the number of images (~1.3k here, a few seconds). It will not scale to large
  corpora without an index.

---

## 8. Citation

```bibtex
@article{ghosh2025crossdataset,
  title   = {Cross-Dataset Generalisation of Rice Leaf Disease Classifiers:
             Evaluating Intra-Crop Domain Shift and Target-Domain Adaptation},
  author  = {Ghosh, Rajib Chandra and Mondal, Semanto and Moccardi, Alberto and Amato, Flora},
  journal = {Smart Agricultural Technology},
  year    = {2025}
}
```



