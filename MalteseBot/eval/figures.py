"""figures.py — publication figures for Chapter 4 (resubmission, Sep 2026).

Reads stats_summary.json / results CSVs written by evaluate_multi.py, ablation.py
and stats.py, and writes PNGs into the dissertation Figures/ folder.

Palette: two categorical hues only (blue = retrieval-augmented, orange = baseline),
validated for colour-vision deficiency; text in ink colours, recessive grid,
legend always present, values labelled directly because a printed figure has no
hover layer.

Run (from MalteseBot, inside the venv):
    .venv\\Scripts\\python.exe eval\\figures.py [--out <dir>]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_OUT = EVAL_DIR.parent.parent.parent / "Dissertation_LaTeX" / "Figures"
MODEL_ORDER = ["gpt-4o-2024-11-20", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-nano", "gpt-5-mini", "gpt-5.5"]
LABEL = {"gpt-4o-2024-11-20": "GPT-4o", "gpt-4o-mini": "GPT-4o\nmini", "gpt-4.1": "GPT-4.1",
         "gpt-4.1-nano": "GPT-4.1\nnano", "gpt-5-mini": "GPT-5\nmini", "gpt-5.5": "GPT-5.5"}
CONFIG_ORDER = ["bm25", "dense", "hybrid", "hybrid+rerank"]
CONFIG_LABEL = {"bm25": "BM25", "dense": "Dense", "hybrid": "Hybrid", "hybrid+rerank": "Hybrid +\nrerank"}

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
SEED = 20260906
rng = np.random.default_rng(SEED)

plt.rcParams.update({
    "font.family": ["Arial", "DejaVu Sans"], "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "legend.frameon": False, "figure.dpi": 200, "savefig.dpi": 200,
})


def fnum(v):
    try:
        return None if v in ("", None) else float(v)
    except ValueError:
        return None


def boot_ci(vals: np.ndarray, B: int = 10_000) -> tuple[float, float]:
    n = len(vals)
    if n == 0:
        return (math.nan, math.nan)
    idx = rng.integers(0, n, size=(B, n))
    means = vals[idx].mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def load_rows():
    with open(EVAL_DIR / "results_multimodel.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if r["run"] == "1"]


def grouped_bars(ax, cats, a, b, a_ci, b_ci, la, lb, ylabel, fmt="{:.2f}", ylim=(0, 1.0)):
    x = np.arange(len(cats))
    w = 0.36
    for off, vals, cis, col, lab in ((-w / 2, a, a_ci, BLUE, la), (w / 2, b, b_ci, ORANGE, lb)):
        err = np.array([[v - lo, hi - v] for v, (lo, hi) in zip(vals, cis)]).T
        err = np.nan_to_num(err, nan=0.0)
        ax.bar(x + off, vals, w, color=col, label=lab, linewidth=0, zorder=3)
        ax.errorbar(x + off, vals, yerr=err, fmt="none", ecolor=INK2, elinewidth=0.9, capsize=2, zorder=4)
        for xi, v, (lo, hi) in zip(x + off, vals, cis):
            top = hi if not math.isnan(hi) else v
            ax.text(xi, top + 0.02, fmt.format(v), ha="center", va="bottom", fontsize=7.5, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)


def shared_legend(fig):
    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.07, 1, 1))


def fig_multimodel(rows, out: Path):
    models = [m for m in MODEL_ORDER if any(r["model"] == m for r in rows)]
    cats = [LABEL[m] for m in models]
    prec = {c: [] for c in ("RAG", "Baseline")}
    prec_ci = {c: [] for c in ("RAG", "Baseline")}
    hal = {c: [] for c in ("RAG", "Baseline")}
    hal_ci = {c: [] for c in ("RAG", "Baseline")}
    for m in models:
        for c in ("RAG", "Baseline"):
            sub = [r for r in rows if r["model"] == m and r["condition"] == c]
            kp = np.array([fnum(r["kp_alias"]) for r in sub if fnum(r["kp_alias"]) is not None])
            prec[c].append(float(kp.mean()) if len(kp) else math.nan)
            prec_ci[c].append(boot_ci(kp))
            h = [int(float(r["halluc_j1"])) for r in sub if r["halluc_j1"] != ""]
            hal[c].append(sum(h) / len(h) if h else math.nan)
            hal_ci[c].append(wilson(sum(h), len(h)))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 3.4))
    grouped_bars(ax1, cats, prec["RAG"], prec["Baseline"], prec_ci["RAG"], prec_ci["Baseline"],
                 "Retrieval-augmented", "No-retrieval baseline", "Fact precision (alias matcher)", ylim=(0, 1.18))
    grouped_bars(ax2, cats, hal["RAG"], hal["Baseline"], hal_ci["RAG"], hal_ci["Baseline"],
                 "Retrieval-augmented", "No-retrieval baseline", "Hallucination rate (judge 1)", fmt="{:.0%}",
                 ylim=(0, 1.18))
    ax1.set_title("(a) Fact precision, 95% bootstrap CI", loc="left")
    ax2.set_title("(b) Incorrect-claim rate, 95% Wilson CI", loc="left")
    shared_legend(fig)
    fig.savefig(out / "fig_multimodel.png")
    plt.close(fig)
    print("wrote fig_multimodel.png")


def fig_gain(S: dict, out: Path):
    pm = S["per_model_run1"]
    models = [m for m in MODEL_ORDER if m in pm]
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.grid(axis="x", color=GRID)
    ax.grid(axis="y", visible=False)
    y = np.arange(len(models))[::-1]
    for dy, metric, mk, lab, fill in ((0.16, "kp_strict", "o", "Strict matcher (pre-registered)", "white"),
                                      (-0.16, "kp_alias", "o", "Alias-tolerant matcher", BLUE)):
        xs = [100 * pm[m][metric]["diff"] for m in models]
        lo = [100 * pm[m][metric]["ci_lo"] for m in models]
        hi = [100 * pm[m][metric]["ci_hi"] for m in models]
        ax.errorbar(xs, y + dy, xerr=[np.array(xs) - np.array(lo), np.array(hi) - np.array(xs)], fmt="none",
                    ecolor=BLUE, elinewidth=1.2, capsize=2.5, zorder=3)
        ax.scatter(xs, y + dy, s=42, marker=mk, facecolor=fill, edgecolor=BLUE, linewidth=1.4, zorder=4, label=lab)
        for xv, yv in zip(xs, y + dy):
            ax.text(xv, yv + 0.13 if dy > 0 else yv - 0.36, f"{xv:+.0f}", fontsize=7.5, ha="center", color=INK)
    ax.axvline(0, color=INK2, linewidth=0.9)
    ax.axvline(30, color=INK2, linewidth=0.9, linestyle="--")
    ax.text(30, len(models) - 0.35, "H1 threshold\n(+30 pp)", fontsize=7.5, ha="left", va="top", color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels([LABEL[m].replace("\n", " ") for m in models])
    ax.set_xlabel("Retrieval gain in fact precision, percentage points (95% bootstrap CI, N = 24)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "fig_gain_by_model.png")
    plt.close(fig)
    print("wrote fig_gain_by_model.png")


def fig_ablation(S: dict, out: Path):
    ab = S.get("ablation", {})
    ret = ab.get("retrieval", {})
    ds = ab.get("downstream", {})
    if not ret:
        return
    embs = [e for e in ("text-embedding-3-large", "text-embedding-3-small") if e in ret]
    cats = [CONFIG_LABEL[c] for c in CONFIG_ORDER]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 3.3))
    x = np.arange(len(CONFIG_ORDER))
    w = 0.36
    for i, (e, col) in enumerate(zip(embs, (BLUE, ORANGE))):
        vals = [ret[e].get(c, {}).get("mrr", math.nan) for c in CONFIG_ORDER]
        off = (-w / 2, w / 2)[i] if len(embs) == 2 else 0
        ax1.bar(x + off, vals, w, color=col, linewidth=0, zorder=3, label=e.replace("text-embedding-3-", "embedding-3-"))
        for xi, v in zip(x + off, vals):
            ax1.text(xi, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(cats)
    ax1.set_ylim(0, 1.15)
    ax1.set_ylabel("Mean reciprocal rank of expected provision")
    ax1.set_title("(a) Retrieval quality by configuration", loc="left")
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2)
    if ds:
        kp = [ds.get(c, {}).get("kp_alias", math.nan) for c in CONFIG_ORDER]
        hl = [ds.get(c, {}).get("hallucination", math.nan) for c in CONFIG_ORDER]
        ax2.bar(x - w / 2, kp, w, color=BLUE, linewidth=0, zorder=3, label="Fact precision (alias)")
        ax2.bar(x + w / 2, hl, w, color=ORANGE, linewidth=0, zorder=3, label="Hallucination rate")
        for xi, v in zip(x - w / 2, kp):
            ax2.text(xi, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5)
        for xi, v in zip(x + w / 2, hl):
            ax2.text(xi, v + 0.015, f"{v:.0%}", ha="center", va="bottom", fontsize=7.5)
        ax2.set_xticks(x)
        ax2.set_xticklabels(cats)
        ax2.set_ylim(0, 1.15)
        ax2.set_ylabel("Proportion")
        ax2.set_title("(b) Downstream GPT-4o answer quality", loc="left")
        ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out / "fig_ablation.png")
    plt.close(fig)
    print("wrote fig_ablation.png")


def fig_oos(S: dict, out: Path):
    oo = S.get("oos", {})
    models = [m for m in MODEL_ORDER if m in oo and "RAG" in oo[m]]
    if not models:
        return
    cats = [LABEL[m] for m in models]
    ref = {c: [oo[m][c]["refusal"] for m in models] for c in ("RAG", "Baseline")}
    ref_ci = {c: [tuple(oo[m][c]["refusal_ci"]) for m in models] for c in ("RAG", "Baseline")}
    ass = {c: [oo[m][c]["asserts_j1"] if oo[m][c]["asserts_j1"] is not None else math.nan for m in models]
           for c in ("RAG", "Baseline")}
    n = oo[models[0]]["RAG"]["n"]
    ass_ci = {c: [wilson(int(round(v * n)), n) if not math.isnan(v) else (math.nan, math.nan) for v in ass[c]]
              for c in ("RAG", "Baseline")}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 3.4))
    grouped_bars(ax1, cats, ref["RAG"], ref["Baseline"], ref_ci["RAG"], ref_ci["Baseline"],
                 "Retrieval-augmented", "No-retrieval baseline", "Share declining for lack of coverage",
                 fmt="{:.0%}", ylim=(0, 1.18))
    grouped_bars(ax2, cats, ass["RAG"], ass["Baseline"], ass_ci["RAG"], ass_ci["Baseline"],
                 "Retrieval-augmented", "No-retrieval baseline", "Share asserting specifics as fact",
                 fmt="{:.0%}", ylim=(0, 1.18))
    ax1.set_title("(a) Honest refusal on out-of-scope questions", loc="left")
    ax2.set_title("(b) Fabricated specifics (judge 1), 95% Wilson CI", loc="left")
    shared_legend(fig)
    fig.savefig(out / "fig_oos.png")
    plt.close(fig)
    print("wrote fig_oos.png")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    S = json.loads((EVAL_DIR / "stats_summary.json").read_text(encoding="utf-8"))
    rows = load_rows()
    fig_multimodel(rows, out)
    fig_gain(S, out)
    fig_ablation(S, out)
    fig_oos(S, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
