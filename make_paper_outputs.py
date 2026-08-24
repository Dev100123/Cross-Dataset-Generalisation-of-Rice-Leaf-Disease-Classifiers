"""
make_paper_outputs.py

Reads results/<model>/results.csv + results/<model>/model_info.json and produces
paper-ready tables and figures in paper/.

OUTPUTS (paper/)
  table_architectures.tex     models x [params, ceilings, zero-shot, full-FT]  (Table 1)
  table_metrics_<model>.tex   full metric suite for one model (both directions)
  fig_arch_bars.pdf/.png      zero-shot vs full-FT per model, both directions
  fig_size_vs_perf.pdf/.png   params (log x) vs zero-shot F1 and full-FT F1
  fig_fewshot_<model>.pdf     macro-F1 vs label budget (models with >1 budget)
  fig_confusion_<model>.pdf   zero-shot confusion matrices (the failure mode)
  summary.md                  everything, readable

Run:
    pip install pandas matplotlib numpy
    python make_paper_outputs.py
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS = Path("results")
OUT = Path("paper")

A = "dhan_shomadhan"
B = "bact_fungal"
DS_SHORT = {A: "DHA", B: "SIP"}

CLASSES = ["brown_spot", "blast", "leaf_scald", "sheath_blight"]
CLASS_NICE = ["Brown spot", "Blast", "Leaf scald", "Sheath blight"]
CHANCE = 1.0 / len(CLASSES)

# nice display order + labels for models
MODEL_ORDER = ["segformer_v0", "resnet50", "convnext_tiny", "segformer_v5", "vit", "convnext_large"]
MODEL_NICE = {"resnet50": "ResNet-50", "vit": "ViT-B/16",
              "convnext_tiny": "ConvNeXt-T", "convnext_large": "ConvNeXt-L",
              "segformer_v0": "MiT-B0", "segformer_v5": "MiT-B5"}

PAIRS = [(f"{B}->{A}", f"{A}->{A}", f"{DS_SHORT[B]}$\\to${DS_SHORT[A]}"),
         (f"{A}->{B}", f"{B}->{B}", f"{DS_SHORT[A]}$\\to${DS_SHORT[B]}")]

C_ZS = "#D85A30"
C_FT = "#1D9E75"
C_CE = "#5F5E5A"
C_CH = "#A32D2D"


# ----------------------------------------------------------------------------

def load_all():
    frames, info = [], {}
    for csv_path in sorted(RESULTS.glob("*/results.csv")):
        frames.append(pd.read_csv(csv_path))
        ij = csv_path.parent / "model_info.json"
        if ij.exists():
            d = json.loads(ij.read_text())
            info[d["model"]] = d
    if not frames:
        raise SystemExit(f"no results found under {RESULTS}/*/results.csv")
    df = pd.concat(frames, ignore_index=True)
    df["budget"] = df["budget"].astype(float)
    return df, info


def agg(df, model, direction, regime, budget=None, metric="f1_macro"):
    m = (df["model"] == model) & (df["direction"] == direction) & (df["regime"] == regime)
    if budget is not None:
        m &= np.isclose(df["budget"], budget)
    v = df.loc[m, metric].values
    if len(v) == 0:
        return np.nan, np.nan, 0
    return float(v.mean()), float(v.std()), len(v)


def models_present(df):
    present = [m for m in MODEL_ORDER if m in set(df["model"])]
    present += [m for m in df["model"].unique() if m not in present]
    return present


def cell(mean, std, n):
    return "--" if n == 0 else f"{mean:.3f}$\\pm${std:.3f}"


# ----------------------------------------------------------------------------
# Table 1: architecture comparison
# ----------------------------------------------------------------------------

def table_architectures(df, info):
    models = models_present(df)
    L = ["% requires \\usepackage{booktabs}",
         "\\begin{table*}[t]\\centering\\small",
         "\\caption{Cross-dataset generalisation across backbones (macro-F1, "
         "mean$\\pm$std over seeds$\\times$folds). All models ImageNet-1k pretrained. "
         f"Chance = {CHANCE:.2f}. ZS = zero-shot, FT = full fine-tuning at 100\\% "
         "target labels.}",
         "\\label{tab:arch}",
         "\\begin{tabular}{lr cc cc cc}",
         "\\toprule",
         "& & \\multicolumn{2}{c}{In-domain} & \\multicolumn{2}{c}{Zero-shot} "
         "& \\multicolumn{2}{c}{Full FT} \\\\",
         "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}",
         f"Model & Params & {DS_SHORT[A]} & {DS_SHORT[B]} & "
         f"{DS_SHORT[B]}$\\to${DS_SHORT[A]} & {DS_SHORT[A]}$\\to${DS_SHORT[B]} & "
         f"{DS_SHORT[B]}$\\to${DS_SHORT[A]} & {DS_SHORT[A]}$\\to${DS_SHORT[B]} \\\\",
         "\\midrule"]
    for mn in models:
        pm = info.get(mn, {}).get("params_M", float("nan"))
        cA = cell(*agg(df, mn, f"{A}->{A}", "in_domain", 1.0))
        cB = cell(*agg(df, mn, f"{B}->{B}", "in_domain", 1.0))
        zBA = cell(*agg(df, mn, f"{B}->{A}", "zero_shot", 0.0))
        zAB = cell(*agg(df, mn, f"{A}->{B}", "zero_shot", 0.0))
        fBA = cell(*agg(df, mn, f"{B}->{A}", "ft_full", 1.0))
        fAB = cell(*agg(df, mn, f"{A}->{B}", "ft_full", 1.0))
        L.append(f"{MODEL_NICE.get(mn, mn)} & {pm:.1f}M & {cA} & {cB} & "
                 f"{zBA} & {zAB} & {fBA} & {fAB} \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    (OUT / "table_architectures.tex").write_text("\n".join(L), encoding="utf-8")
    print("  table_architectures.tex")


# ----------------------------------------------------------------------------
# per-model full metric table
# ----------------------------------------------------------------------------

METRICS = [("accuracy", "Acc"), ("balanced_accuracy", "Bal.Acc"),
           ("precision_macro", "Prec$_M$"), ("recall_macro", "Rec$_M$"),
           ("f1_macro", "F1$_M$"), ("f1_weighted", "F1$_w$")]

def table_metrics(df, model):
    regimes = ["zero_shot", "adabn", "ft_head", "ft_encoder", "ft_full", "in_domain"]
    rn = {"zero_shot": "Zero-shot", "adabn": "AdaBN", "ft_head": "Linear probe",
          "ft_encoder": "Encoder-FT", "ft_full": "Full FT", "in_domain": "Target-only"}
    L = ["% requires \\usepackage{booktabs}",
         "\\begin{table}[t]\\centering\\small",
         f"\\caption{{Full metrics for {MODEL_NICE.get(model, model)} "
         "(mean over seeds$\\times$folds).}}",
         f"\\label{{tab:metrics_{model}}}",
         "\\begin{tabular}{ll" + "c" * len(METRICS) + "}", "\\toprule",
         "Dir. & Method & " + " & ".join(m[1] for m in METRICS) + " \\\\", "\\midrule"]
    for di, (d, ind, nice) in enumerate(PAIRS):
        for r in regimes:
            src = d if r != "in_domain" else ind
            b = 0.0 if r in ("zero_shot", "adabn") else 1.0
            vals = []
            for mkey, _ in METRICS:
                mean, _, n = agg(df, model, src, r, b, mkey)
                vals.append("--" if n == 0 else f"{mean:.3f}")
            first = nice if r == regimes[0] else ""
            L.append(f"{first} & {rn[r]} & " + " & ".join(vals) + " \\\\")
        L.append("\\midrule")
    L[-1] = "\\bottomrule"
    L += ["\\end{tabular}", "\\end{table}"]
    (OUT / f"table_metrics_{model}.tex").write_text("\n".join(L), encoding="utf-8")
    print(f"  table_metrics_{model}.tex")


# ----------------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------------

def fig_arch_bars(df):
    models = models_present(df)
    x = np.arange(len(models))
    w = 0.2
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, (d, ind, nice) in zip(axes, PAIRS):
        zs = [agg(df, m, d, "zero_shot", 0.0) for m in models]
        ft = [agg(df, m, d, "ft_full", 1.0) for m in models]
        ce = [agg(df, m, ind, "in_domain", 1.0) for m in models]
        ax.bar(x - w, [a[0] for a in zs], w, yerr=[a[1] for a in zs], capsize=2,
               color=C_ZS, label="Zero-shot", zorder=3)
        ax.bar(x, [a[0] for a in ft], w, yerr=[a[1] for a in ft], capsize=2,
               color=C_FT, label="Full FT", zorder=3)
        ax.bar(x + w, [a[0] for a in ce], w, yerr=[a[1] for a in ce], capsize=2,
               color=C_CE, label="Target-only", zorder=3)
        ax.axhline(CHANCE, ls=":", lw=1, color=C_CH)
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_NICE.get(m, m) for m in models], rotation=25, ha="right")
        ax.set_title(nice.replace("$\\to$", "→"))
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.3, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Macro-F1")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(OUT / f"fig_arch_bars.{e}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  fig_arch_bars.pdf")


def fig_size_vs_perf(df, info):
    models = [m for m in models_present(df) if m in info]
    xs = [info[m]["params_M"] for m in models]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (d, ind, nice) in zip(axes, PAIRS):
        zs = [agg(df, m, d, "zero_shot", 0.0)[0] for m in models]
        ft = [agg(df, m, d, "ft_full", 1.0)[0] for m in models]
        ax.scatter(xs, zs, color=C_ZS, s=55, zorder=3, label="Zero-shot")
        ax.scatter(xs, ft, color=C_FT, s=55, zorder=3, label="Full FT")
        for xi, mn in zip(xs, models):
            ax.annotate(MODEL_NICE.get(mn, mn), (xi, ft[models.index(mn)]),
                        fontsize=7, xytext=(0, 6), textcoords="offset points", ha="center")
        ax.axhline(CHANCE, ls=":", lw=1, color=C_CH)
        ax.set_xscale("log")
        ax.set_xlabel("Parameters (M, log scale)")
        ax.set_title(nice.replace("$\\to$", "→"))
        ax.set_ylim(0, 1.0)
        ax.grid(alpha=0.3, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Macro-F1")
    axes[0].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(OUT / f"fig_size_vs_perf.{e}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  fig_size_vs_perf.pdf")


def fig_fewshot(df):
    for mn in models_present(df):
        budgets = sorted(b for b in df.loc[(df["model"] == mn) &
                         df["regime"].str.startswith("ft_"), "budget"].unique() if b > 0)
        if len(budgets) < 2:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
        for ax, (d, ind, nice) in zip(axes, PAIRS):
            for r, lbl in [("ft_head", "Linear probe"), ("ft_encoder", "Encoder-FT"),
                           ("ft_full", "Full FT")]:
                ms = [agg(df, mn, d, r, b)[0] for b in budgets]
                ss = [agg(df, mn, d, r, b)[1] for b in budgets]
                ms, ss = np.array(ms), np.array(ss)
                ax.plot(budgets, ms, marker="o", ms=4, label=lbl)
                ax.fill_between(budgets, ms - ss, ms + ss, alpha=0.15)
            zs = agg(df, mn, d, "zero_shot", 0.0)[0]
            ce = agg(df, mn, ind, "in_domain", 1.0)[0]
            if not np.isnan(zs):
                ax.axhline(zs, ls="-.", lw=1, color=C_ZS, label="Zero-shot")
            if not np.isnan(ce):
                ax.axhline(ce, ls="--", lw=1.2, color=C_CE, label="Target-only")
            ax.axhline(CHANCE, ls=":", lw=1, color=C_CH)
            ax.set_xscale("log")
            ax.set_xticks(budgets)
            ax.set_xticklabels([f"{int(b*100)}%" for b in budgets])
            ax.set_xlabel("Labelled target (% of train pool)")
            ax.set_title(nice.replace("$\\to$", "→"))
            ax.grid(alpha=0.3)
            ax.spines[["top", "right"]].set_visible(False)
        axes[0].set_ylabel("Macro-F1")
        axes[0].legend(fontsize=8)
        fig.suptitle(MODEL_NICE.get(mn, mn), fontsize=11)
        fig.tight_layout()
        for e in ("pdf", "png"):
            fig.savefig(OUT / f"fig_fewshot_{mn}.{e}", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  fig_fewshot_{mn}.pdf")


def fig_confusion(df, model):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    drew = False
    for ax, (d, ind, nice) in zip(axes, PAIRS):
        sub = df.loc[(df["model"] == model) & (df["direction"] == d) &
                     (df["regime"] == "zero_shot"), "confusion"]
        if sub.empty:
            ax.axis("off"); continue
        cm = np.sum([np.array(json.loads(s)) for s in sub], axis=0).astype(float)
        cmn = cm / np.clip(cm.sum(1, keepdims=True), 1, None)
        ax.imshow(cmn, cmap="Oranges", vmin=0, vmax=1)
        for i in range(len(CLASSES)):
            for j in range(len(CLASSES)):
                ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if cmn[i, j] > 0.5 else "#4A1B0C")
        ax.set_xticks(range(len(CLASSES))); ax.set_xticklabels(CLASS_NICE, rotation=35, ha="right", fontsize=8)
        ax.set_yticks(range(len(CLASSES))); ax.set_yticklabels(CLASS_NICE, fontsize=8)
        ax.set_xlabel("Predicted"); ax.set_title(f"Zero-shot {nice}".replace("$\\to$", "→"), fontsize=10)
        drew = True
    if not drew:
        plt.close(fig); return
    axes[0].set_ylabel("True")
    fig.suptitle(MODEL_NICE.get(model, model), fontsize=11)
    fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(OUT / f"fig_confusion_{model}.{e}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  fig_confusion_{model}.pdf")


def summary_md(df, info):
    L = ["# Results summary", ""]
    L.append("| Model | Params | ZS B→A | ZS A→B | FT B→A | FT A→B | ceil A | ceil B |")
    L.append("|---|---|---|---|---|---|---|---|")
    for mn in models_present(df):
        pm = info.get(mn, {}).get("params_M", "?")
        def c(d, r, b): 
            m, s, n = agg(df, mn, d, r, b); return "--" if n == 0 else f"{m:.3f}±{s:.3f}"
        L.append(f"| {MODEL_NICE.get(mn, mn)} | {pm}M "
                 f"| {c(f'{B}->{A}','zero_shot',0.0)} | {c(f'{A}->{B}','zero_shot',0.0)} "
                 f"| {c(f'{B}->{A}','ft_full',1.0)} | {c(f'{A}->{B}','ft_full',1.0)} "
                 f"| {c(f'{A}->{A}','in_domain',1.0)} | {c(f'{B}->{B}','in_domain',1.0)} |")
    txt = "\n".join(L).replace("$\\to$", "→")
    (OUT / "summary.md").write_text(txt, encoding="utf-8")
    print("  summary.md"); print("\n" + txt)


# ----------------------------------------------------------------------------

def main():
    OUT.mkdir(exist_ok=True)
    df, info = load_all()
    print(f"loaded {len(df)} rows, {df['model'].nunique()} models\n")

    n_seeds = df["seed"].nunique()
    if n_seeds < 3:
        print(f"  [!] only {n_seeds} seed(s); error bars are fold-only. Widen SEEDS.\n")

    table_architectures(df, info)
    fig_arch_bars(df)
    fig_size_vs_perf(df, info)
    fig_fewshot(df)
    for mn in models_present(df):
        table_metrics(df, mn)
    # confusion for the reference model if present, else the first
    ref = "resnet50" if "resnet50" in set(df["model"]) else models_present(df)[0]
    fig_confusion(df, ref)
    summary_md(df, info)
    print(f"\n-> {OUT.resolve()}")


if __name__ == "__main__":
    main()
