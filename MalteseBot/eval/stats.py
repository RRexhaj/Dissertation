"""stats.py — inferential statistics for the technical evaluation (resubmission, Sep 2026).

Reads the long-format results written by evaluate_multi.py (and, when present,
results_oos.csv, results_ablation_*.csv) and produces stats_summary.json plus
LaTeX table fragments in eval/tables/ that the dissertation \\input{}s, so that
no number is retyped by hand.

Analysis plan (pre-specified in Chapter 3 of the resubmission)
--------------------------------------------------------------
Primary unit = the 24 gold questions, paired across conditions (RAG vs Baseline)
within each generator model. Run 1 is the primary run; runs 2-3 are replications.

Per model
  * fact precision (strict = pre-registered June matcher; alias = sensitivity):
    Wilcoxon signed-rank test, matched-pairs rank-biserial r, percentile
    bootstrap 95% CI (10 000 resamples) of the mean paired difference.
    H1 (>= 30 pp advantage) is assessed on the point estimate and on the CI.
  * hallucination (incorrect claim, judge 1 primary / judge 2 secondary),
    omission, citation validity: exact McNemar test on discordant pairs,
    risk difference with bootstrap CI.
Across models
  * Friedman test (questions as blocks) + Kendall's W on precision within each
    condition and on the per-question gain; pairwise Wilcoxon with Holm
    correction where the omnibus test is significant; Cochran's Q on the
    binary hallucination flag.
Exploratory
  * gain by language (Mann-Whitney U, 12 vs 12) and by difficulty
    (Kruskal-Wallis, 8/10/6).
Stability
  * mean +/- SD of cell means across the three runs; per-question consistency.
Judges, out-of-scope set, retrieval ablation and cost are summarised too.

Run (from MalteseBot, inside the venv):
    .venv\\Scripts\\python.exe eval\\stats.py
"""
from __future__ import annotations

import csv
import json
import math
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

EVAL_DIR = Path(__file__).resolve().parent
TABLES = EVAL_DIR / "tables"
RESULTS = EVAL_DIR / "results_multimodel.csv"
OOS = EVAL_DIR / "results_oos.csv"
ABL_RET = EVAL_DIR / "results_ablation_retrieval.csv"
ABL_DS = EVAL_DIR / "results_ablation_downstream.csv"
PRICES = EVAL_DIR / "prices.json"

MODEL_ORDER = ["gpt-4o-2024-11-20", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-nano", "gpt-5-mini", "gpt-5.5"]
LABEL = {"gpt-4o-2024-11-20": "GPT-4o", "gpt-4o-mini": "GPT-4o mini", "gpt-4.1": "GPT-4.1",
         "gpt-4.1-nano": "GPT-4.1 nano", "gpt-5-mini": "GPT-5 mini", "gpt-5.5": "GPT-5.5"}
PRIMARY = "gpt-4o-2024-11-20"
CONDS = ("RAG", "Baseline")
H1_THRESHOLD = 0.30
B = 10_000
SEED = 20260906
NUMERIC = ["kp_strict", "kp_alias", "cite_ok", "invalid_cite", "mirror", "halluc_j1", "omission_j1",
           "halluc_j2", "omission_j2", "rouge_l", "latency_s", "tokens_in", "tokens_out"]
CONFIG_ORDER = ["bm25", "dense", "hybrid", "hybrid+rerank"]
CONFIG_LABEL = {"bm25": "BM25 only", "dense": "Dense only", "hybrid": "Hybrid (0.6/0.4)",
                "hybrid+rerank": "Hybrid + rerank (deployed)"}

rng = np.random.default_rng(SEED)


# ---------------------------------------------------------------------------
# Generic statistics helpers
# ---------------------------------------------------------------------------

def fnum(v):
    if v is None or v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def label(m: str) -> str:
    return LABEL.get(m, m)


def p_str(p) -> str:
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "--"
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def wilcoxon_rb(x: np.ndarray, y: np.ndarray) -> dict:
    d = x - y
    nz = d[d != 0]
    out = {"n": int(len(d)), "n_nonzero": int(len(nz))}
    if len(nz) == 0:
        out.update({"W": 0.0, "p": 1.0, "r_rb": 0.0})
        return out
    try:
        res = stats.wilcoxon(x, y)
        out["W"], out["p"] = float(res.statistic), float(res.pvalue)
    except ValueError:
        out["W"], out["p"] = float("nan"), 1.0
    ranks = stats.rankdata(np.abs(nz))
    rp, rn = ranks[nz > 0].sum(), ranks[nz < 0].sum()
    out["r_rb"] = float((rp - rn) / (rp + rn)) if (rp + rn) else 0.0
    return out


def boot_mean_diff(x: np.ndarray, y: np.ndarray) -> dict:
    d = x - y
    n = len(d)
    if n == 0:
        return {"mean": None, "ci_lo": None, "ci_hi": None}
    idx = rng.integers(0, n, size=(B, n))
    means = d[idx].mean(axis=1)
    return {"mean": float(d.mean()), "ci_lo": float(np.percentile(means, 2.5)),
            "ci_hi": float(np.percentile(means, 97.5))}


def mcnemar_exact(x: np.ndarray, y: np.ndarray) -> dict:
    b = int(((x == 1) & (y == 0)).sum())
    c = int(((x == 0) & (y == 1)).sum())
    p = float(stats.binomtest(min(b, c), b + c, 0.5).pvalue) if (b + c) else 1.0
    return {"b": b, "c": c, "p": p}


def friedman(matrix: np.ndarray) -> dict:
    n, k = matrix.shape
    if n < 3 or k < 2 or np.allclose(matrix, matrix[0]):
        return {"n": int(n), "k": int(k), "chi2": None, "p": None, "W": None}
    try:
        chi2, p = stats.friedmanchisquare(*[matrix[:, j] for j in range(k)])
    except ValueError:
        return {"n": int(n), "k": int(k), "chi2": None, "p": None, "W": None}
    return {"n": int(n), "k": int(k), "chi2": float(chi2), "p": float(p), "W": float(chi2 / (n * (k - 1)))}


def cochran_q(matrix: np.ndarray) -> dict:
    n, k = matrix.shape
    G = matrix.sum(axis=0)
    L = matrix.sum(axis=1)
    denom = k * L.sum() - (L ** 2).sum()
    if denom == 0:
        return {"n": int(n), "k": int(k), "Q": None, "p": None}
    Q = (k - 1) * (k * (G ** 2).sum() - G.sum() ** 2) / denom
    return {"n": int(n), "k": int(k), "Q": float(Q), "p": float(stats.chi2.sf(Q, k - 1))}


def holm(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (m - rank) * pvals[i])
        running = max(running, val)
        adj[i] = running
    return adj


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def kappa(a: list[int], b: list[int]) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def mean(vals) -> float | None:
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None


def sd(vals) -> float | None:
    vals = [v for v in vals if v is not None]
    return float(np.std(vals, ddof=1)) if len(vals) > 1 else None


# ---------------------------------------------------------------------------
# LaTeX table writer (matches the style of the June tables)
# ---------------------------------------------------------------------------

# Tables whose natural width exceeds the text block (measured on the compiled PDF);
# they are scaled to \textwidth with \resizebox instead of overflowing the margin.
WIDE_TABLES = {"tab_primary_gpt4o", "tab_across_models", "tab_stability", "tab_ablation",
               "tab_oos", "tab_multimodel_halluc"}


def tex_table(name: str, caption: str, tlabel: str, colspec: str, header: list[str],
              rows: list[list[str]], wide: bool = False, note: str | None = None) -> None:
    TABLES.mkdir(exist_ok=True)
    wide = wide or name in WIDE_TABLES
    lines = ["\\begin{table}[H]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{tlabel}}}",
             "{\\footnotesize\\setstretch{1.0}\\setlength{\\tabcolsep}{4pt}\\def\\arraystretch{1.2}"]
    if wide:
        lines.append("\\resizebox{\\textwidth}{!}{%")
    lines += [f"\\begin{{tabular}}{{{colspec}}}", "\\toprule",
              " & ".join(f"\\textbf{{{h}}}" for h in header) + " \\\\", "\\midrule"]
    lines += [" & ".join(r) + " \\\\" for r in rows]
    lines += ["\\bottomrule", "\\end{tabular}" + ("}" if wide else ""), "}"]
    if note:
        lines.append(f"\\par\\vspace{{2pt}}\\footnotesize{{{note}}}")
    lines.append("\\end{table}")
    (TABLES / f"{name}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def pct(v, digits: int = 1) -> str:
    return "--" if v is None else f"{100 * v:.{digits}f}\\%"


def dec(v, digits: int = 2) -> str:
    return "--" if v is None else f"{v:.{digits}f}"


def pp(v, digits: int = 1) -> str:
    return "--" if v is None else f"{100 * v:+.{digits}f}"


def ci_pp(lo, hi) -> str:
    return "--" if lo is None else f"[{100 * lo:+.1f}, {100 * hi:+.1f}]"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in NUMERIC:
            if k in r:
                r[k] = fnum(r[k])
    return rows


class Data:
    def __init__(self, rows: list[dict]):
        self.rows = rows
        self.models = [m for m in MODEL_ORDER if any(r["model"] == m for r in rows)]
        self.models += sorted({r["model"] for r in rows} - set(self.models))
        self.runs = sorted({r["run"] for r in rows}, key=int)
        self.by: dict[tuple, dict[str, dict]] = defaultdict(dict)   # (model, cond, run) -> id -> row
        for r in rows:
            self.by[(r["model"], r["condition"], r["run"])][r["id"]] = r
        self.ids = sorted({r["id"] for r in rows})
        self.info = {r["id"]: (r["lang"], r["difficulty"], r["topic"]) for r in rows}

    def cell(self, model: str, cond: str, run: str) -> dict[str, dict]:
        return self.by.get((model, cond, run), {})

    def paired(self, model: str, metric: str, run: str) -> tuple[list[str], np.ndarray, np.ndarray]:
        a, b = self.cell(model, "RAG", run), self.cell(model, "Baseline", run)
        ids = [i for i in self.ids if i in a and i in b and a[i].get(metric) is not None and b[i].get(metric) is not None]
        return ids, np.array([a[i][metric] for i in ids], float), np.array([b[i][metric] for i in ids], float)

    def avg_cell(self, model: str, cond: str) -> dict[str, dict]:
        """Per-question values averaged over runs."""
        acc: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
        for run in self.runs:
            for qid, r in self.cell(model, cond, run).items():
                for k in NUMERIC:
                    if r.get(k) is not None:
                        acc[qid][k].append(r[k])
        return {qid: {k: float(np.mean(v)) for k, v in d.items()} for qid, d in acc.items()}

    def paired_avg(self, model: str, metric: str) -> tuple[list[str], np.ndarray, np.ndarray]:
        a, b = self.avg_cell(model, "RAG"), self.avg_cell(model, "Baseline")
        ids = [i for i in self.ids if i in a and i in b and metric in a[i] and metric in b[i]]
        return ids, np.array([a[i][metric] for i in ids], float), np.array([b[i][metric] for i in ids], float)


# ---------------------------------------------------------------------------
# Analyses
# ---------------------------------------------------------------------------

def per_model(d: Data, run: str | None) -> dict:
    """run=None -> run-averaged values."""
    out = {}
    for m in d.models:
        res = {}
        for metric in ("kp_strict", "kp_alias", "rouge_l"):
            ids, x, y = d.paired(m, metric, run) if run else d.paired_avg(m, metric)
            if not ids:
                continue
            w = wilcoxon_rb(x, y)
            bo = boot_mean_diff(x, y)
            res[metric] = {"n": len(ids), "rag": float(x.mean()), "base": float(y.mean()),
                           "diff": bo["mean"], "ci_lo": bo["ci_lo"], "ci_hi": bo["ci_hi"],
                           "wilcoxon_W": w["W"], "p": w["p"], "r_rb": w["r_rb"], "n_nonzero": w["n_nonzero"]}
            if metric.startswith("kp"):
                res[metric]["h1_point_met"] = bool(bo["mean"] is not None and bo["mean"] >= H1_THRESHOLD)
                res[metric]["ci_excludes_zero"] = bool(bo["ci_lo"] is not None and bo["ci_lo"] > 0)
                res[metric]["ci_lower_clears_threshold"] = bool(bo["ci_lo"] is not None and bo["ci_lo"] >= H1_THRESHOLD)
        for metric in ("halluc_j1", "halluc_j2", "omission_j1", "omission_j2", "cite_ok", "invalid_cite", "mirror"):
            ids, x, y = d.paired(m, metric, run) if run else d.paired_avg(m, metric)
            if not ids:
                continue
            bo = boot_mean_diff(x, y)
            entry = {"n": len(ids), "rag": float(x.mean()), "base": float(y.mean()),
                     "diff": bo["mean"], "ci_lo": bo["ci_lo"], "ci_hi": bo["ci_hi"]}
            if run:  # binary flags -> exact McNemar
                entry.update(mcnemar_exact(x.round(), y.round()))
            res[metric] = entry
        out[m] = res
    return out


def across_models(d: Data, run: str | None) -> dict:
    out = {}
    if len(d.models) < 2:
        return out
    for cond in CONDS:
        for metric in ("kp_strict", "kp_alias"):
            cells = {m: (d.cell(m, cond, run) if run else d.avg_cell(m, cond)) for m in d.models}
            ids = [i for i in d.ids if all(i in cells[m] and cells[m][i].get(metric) is not None for m in d.models)]
            if len(ids) < 3:
                continue
            M = np.array([[cells[m][i][metric] for m in d.models] for i in ids], float)
            fr = friedman(M)
            mean_rank = stats.rankdata(M, axis=1).mean(axis=0)
            entry = {"friedman": fr, "means": {m: float(M[:, j].mean()) for j, m in enumerate(d.models)},
                     "mean_rank": {m: float(mean_rank[j]) for j, m in enumerate(d.models)}, "pairwise": {}}
            if fr["p"] is not None and fr["p"] < 0.05:
                pairs = list(combinations(range(len(d.models)), 2))
                raw = [wilcoxon_rb(M[:, i], M[:, j])["p"] for i, j in pairs]
                adj = holm(raw)
                for (i, j), pr, pa in zip(pairs, raw, adj):
                    entry["pairwise"][f"{d.models[i]} vs {d.models[j]}"] = {
                        "p_raw": pr, "p_holm": pa, "significant": bool(pa < 0.05),
                        "diff": float(M[:, i].mean() - M[:, j].mean())}
            out[f"{cond}_{metric}"] = entry
        # binary hallucination across models: Cochran's Q
        for metric in ("halluc_j1", "halluc_j2"):
            if not run:
                continue
            cells = {m: d.cell(m, cond, run) for m in d.models}
            ids = [i for i in d.ids if all(i in cells[m] and cells[m][i].get(metric) is not None for m in d.models)]
            if len(ids) < 3:
                continue
            M = np.array([[cells[m][i][metric] for m in d.models] for i in ids], float).round()
            out[f"{cond}_{metric}_cochranQ"] = cochran_q(M) | {"rates": {m: float(M[:, j].mean()) for j, m in enumerate(d.models)}}
    # gains
    for metric in ("kp_strict", "kp_alias"):
        gains = {}
        for m in d.models:
            ids, x, y = d.paired(m, metric, run) if run else d.paired_avg(m, metric)
            gains[m] = dict(zip(ids, x - y))
        ids = [i for i in d.ids if all(i in gains[m] for m in d.models)]
        if len(ids) < 3:
            continue
        M = np.array([[gains[m][i] for m in d.models] for i in ids], float)
        out[f"gain_{metric}"] = {"friedman": friedman(M), "means": {m: float(M[:, j].mean()) for j, m in enumerate(d.models)}}
    return out


def subgroup(d: Data, run: str) -> dict:
    out = {"by_language": {}, "by_difficulty": {}}
    for m in d.models:
        ids, x, y = d.paired(m, "kp_alias", run)
        if not ids:
            continue
        gain = x - y
        lang = np.array([d.info[i][0] for i in ids])
        diff = np.array([d.info[i][1] for i in ids])
        hal_ids, hx, hy = d.paired(m, "halluc_j1", run)
        hlang = np.array([d.info[i][0] for i in hal_ids])
        entry = {}
        for lg in ("en", "mt"):
            sel = lang == lg
            hsel = hlang == lg
            entry[lg] = {"n": int(sel.sum()), "rag": float(x[sel].mean()) if sel.any() else None,
                         "base": float(y[sel].mean()) if sel.any() else None,
                         "gain": float(gain[sel].mean()) if sel.any() else None,
                         "halluc_rag": float(hx[hsel].mean()) if hsel.any() else None,
                         "halluc_base": float(hy[hsel].mean()) if hsel.any() else None}
        if (lang == "en").sum() and (lang == "mt").sum():
            u = stats.mannwhitneyu(gain[lang == "en"], gain[lang == "mt"], alternative="two-sided")
            entry["mannwhitney_gain_en_vs_mt"] = {"U": float(u.statistic), "p": float(u.pvalue)}
        out["by_language"][m] = entry
        dentry = {}
        groups = []
        for lv in ("easy", "medium", "hard"):
            sel = diff == lv
            if sel.any():
                groups.append(gain[sel])
                dentry[lv] = {"n": int(sel.sum()), "rag": float(x[sel].mean()), "base": float(y[sel].mean()),
                              "gain": float(gain[sel].mean())}
        if len(groups) >= 2 and all(len(g) > 1 for g in groups):
            try:
                kw = stats.kruskal(*groups)
                dentry["kruskal_gain"] = {"H": float(kw.statistic), "p": float(kw.pvalue)}
            except ValueError:
                pass
        out["by_difficulty"][m] = dentry
    return out


def stability(d: Data) -> dict:
    out = {}
    for m in d.models:
        out[m] = {}
        for cond in CONDS:
            per_run = {metric: [mean([r.get(metric) for r in d.cell(m, cond, run).values()]) for run in d.runs]
                       for metric in ("kp_strict", "kp_alias", "halluc_j1", "cite_ok")}
            # per-question consistency of the hallucination flag across runs
            flags = defaultdict(list)
            for run in d.runs:
                for qid, r in d.cell(m, cond, run).items():
                    if r.get("halluc_j1") is not None:
                        flags[qid].append(int(r["halluc_j1"]))
            consistent = [len(set(v)) == 1 for v in flags.values() if len(v) == len(d.runs)]
            kp_sd = []
            for qid in d.ids:
                vals = [d.cell(m, cond, run).get(qid, {}).get("kp_alias") for run in d.runs]
                vals = [v for v in vals if v is not None]
                if len(vals) == len(d.runs) and len(vals) > 1:
                    kp_sd.append(float(np.std(vals, ddof=1)))
            out[m][cond] = {metric: {"per_run": vals, "mean": mean(vals), "sd": sd(vals)}
                            for metric, vals in per_run.items()}
            out[m][cond]["halluc_flag_consistency"] = (sum(consistent) / len(consistent)) if consistent else None
            out[m][cond]["kp_alias_within_question_sd"] = mean(kp_sd)
    return out


def judge_llm_agreement(d: Data) -> dict:
    out = {}
    for metric in ("halluc", "omission"):
        rows = [r for r in d.rows if r.get(f"{metric}_j1") is not None and r.get(f"{metric}_j2") is not None]
        a = [int(r[f"{metric}_j1"]) for r in rows]
        b = [int(r[f"{metric}_j2"]) for r in rows]
        entry = {"n": len(rows), "kappa": kappa(a, b) if rows else None,
                 "agreement": (sum(1 for x, y in zip(a, b) if x == y) / len(rows)) if rows else None,
                 "rate_j1": mean(a), "rate_j2": mean(b)}
        for cond in CONDS:
            sub = [r for r in rows if r["condition"] == cond]
            aa = [int(r[f"{metric}_j1"]) for r in sub]
            bb = [int(r[f"{metric}_j2"]) for r in sub]
            entry[cond] = {"n": len(sub), "kappa": kappa(aa, bb) if sub else None,
                           "agreement": (sum(1 for x, y in zip(aa, bb) if x == y) / len(sub)) if sub else None,
                           "rate_j1": mean(aa), "rate_j2": mean(bb)}
        out[metric] = entry
    return out


def oos_stats(rows: list[dict], models: list[str]) -> dict:
    out = {}
    if not rows:
        return out
    for r in rows:
        for k in ("refusal", "mirror", "asserts_j1", "asserts_j2"):
            r[k] = fnum(r.get(k))
    for m in models:
        out[m] = {}
        by = {c: {r["id"]: r for r in rows if r["model"] == m and r["condition"] == c} for c in CONDS}
        for c in CONDS:
            sub = list(by[c].values())
            if not sub:
                continue
            k = int(sum(r["refusal"] or 0 for r in sub))
            lo, hi = wilson(k, len(sub))
            out[m][c] = {"n": len(sub), "refusal": k / len(sub), "refusal_ci": [lo, hi],
                         "asserts_j1": mean([r["asserts_j1"] for r in sub]),
                         "asserts_j2": mean([r["asserts_j2"] for r in sub]),
                         "mirror": mean([r["mirror"] for r in sub])}
        ids = [i for i in by["RAG"] if i in by["Baseline"]]
        if ids:
            for metric in ("refusal", "asserts_j1"):
                x = np.array([by["RAG"][i][metric] or 0 for i in ids]).round()
                y = np.array([by["Baseline"][i][metric] or 0 for i in ids]).round()
                out[m][f"mcnemar_{metric}"] = mcnemar_exact(x, y) | {"rag": float(x.mean()), "base": float(y.mean())}
    return out


def ablation_stats(ret: list[dict], ds: list[dict]) -> dict:
    out = {"retrieval": {}, "downstream": {}}
    for r in ret:
        for k in ("hit1", "hit5", "rr", "latency_s"):
            r[k] = fnum(r[k])
    for emb in sorted({r["embedder"] for r in ret}):
        out["retrieval"][emb] = {}
        cfgs = [c for c in CONFIG_ORDER if any(r["config"] == c and r["embedder"] == emb for r in ret)]
        mats = {}
        for c in cfgs:
            sub = [r for r in ret if r["embedder"] == emb and r["config"] == c]
            n = len(sub)
            k1 = int(sum(r["hit1"] for r in sub))
            out["retrieval"][emb][c] = {"n": n, "hit1": k1 / n, "hit1_ci": list(wilson(k1, n)),
                                        "hit5": mean([r["hit5"] for r in sub]), "mrr": mean([r["rr"] for r in sub]),
                                        "latency_s": mean([r["latency_s"] for r in sub]),
                                        "mrr_en": mean([r["rr"] for r in sub if r["lang"] == "en"]),
                                        "mrr_mt": mean([r["rr"] for r in sub if r["lang"] == "mt"])}
            mats[c] = {r["id"]: r["rr"] for r in sub}
        ids = [i for i in sorted(mats[cfgs[0]]) if all(i in mats[c] for c in cfgs)] if cfgs else []
        if len(ids) >= 3 and len(cfgs) >= 2:
            M = np.array([[mats[c][i] for c in cfgs] for i in ids], float)
            out["retrieval"][emb]["friedman_rr"] = friedman(M)
            if "hybrid+rerank" in cfgs:
                j = cfgs.index("hybrid+rerank")
                raw, names = [], []
                for k, c in enumerate(cfgs):
                    if c == "hybrid+rerank":
                        continue
                    raw.append(wilcoxon_rb(M[:, k], M[:, j])["p"])
                    names.append(c)
                for c, pr, pa in zip(names, raw, holm(raw)):
                    out["retrieval"][emb][c]["vs_deployed_p_raw"] = pr
                    out["retrieval"][emb][c]["vs_deployed_p_holm"] = pa
    if ds:
        for r in ds:
            for k in ("kp_strict", "kp_alias", "cite_ok", "invalid_cite", "mirror", "halluc", "omission", "rouge_l"):
                r[k] = fnum(r.get(k))
        cfgs = [c for c in CONFIG_ORDER if any(r["config"] == c for r in ds)]
        mats = {}
        for c in cfgs:
            sub = [r for r in ds if r["config"] == c]
            out["downstream"][c] = {"n": len(sub), "kp_strict": mean([r["kp_strict"] for r in sub]),
                                    "kp_alias": mean([r["kp_alias"] for r in sub]),
                                    "citation_rate": mean([r["cite_ok"] for r in sub]),
                                    "hallucination": mean([r["halluc"] for r in sub]),
                                    "omission": mean([r["omission"] for r in sub]),
                                    "mirror": mean([r["mirror"] for r in sub])}
            mats[c] = {r["id"]: r["kp_alias"] for r in sub}
        ids = [i for i in sorted(mats[cfgs[0]]) if all(i in mats[c] for c in cfgs)] if cfgs else []
        if len(ids) >= 3 and "hybrid+rerank" in cfgs:
            M = np.array([[mats[c][i] for c in cfgs] for i in ids], float)
            out["downstream"]["friedman_kp_alias"] = friedman(M)
            j = cfgs.index("hybrid+rerank")
            raw, names = [], []
            for k, c in enumerate(cfgs):
                if c != "hybrid+rerank":
                    raw.append(wilcoxon_rb(M[:, k], M[:, j])["p"])
                    names.append(c)
            for c, pr, pa in zip(names, raw, holm(raw)):
                out["downstream"][c]["vs_deployed_p_raw"] = pr
                out["downstream"][c]["vs_deployed_p_holm"] = pa
    return out


def cost(d: Data) -> dict:
    prices = json.loads(PRICES.read_text(encoding="utf-8")) if PRICES.exists() else {}
    out = {}
    for m in d.models:
        rows = [r for r in d.rows if r["model"] == m]
        snap = Counter(r["snapshot"] for r in rows if r.get("snapshot")).most_common(1)
        entry = {"snapshot": snap[0][0] if snap else "", "price": prices.get(m)}
        for c in CONDS:
            sub = [r for r in rows if r["condition"] == c]
            tin, tout = mean([r["tokens_in"] for r in sub]), mean([r["tokens_out"] for r in sub])
            usd = None
            pr = prices.get(m) or {}
            if tin is not None and pr.get("input") is not None and pr.get("output") is not None:
                usd = (tin / 1e6 * pr["input"] + tout / 1e6 * pr["output"]) * 1000
            entry[c] = {"n": len(sub), "latency_mean": mean([r["latency_s"] for r in sub]),
                        "latency_sd": sd([r["latency_s"] for r in sub]),
                        "tokens_in": tin, "tokens_out": tout, "usd_per_1000_queries": usd}
        out[m] = entry
    return out


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def write_tables(S: dict, d: Data) -> None:
    pm = S["per_model_run1"]
    # Table: fact precision per model (strict and alias), Wilcoxon + CI
    for metric, name, cap in (("kp_strict", "tab_multimodel_strict",
                               "Fact precision (strict matcher, pre-registered) by generator model, "
                               "retrieval-augmented versus no-retrieval baseline on the 24-item gold set (run 1). "
                               "$\\Delta$ in percentage points with percentile-bootstrap 95\\% CI; Wilcoxon signed-rank test; "
                               "$r$ = matched-pairs rank-biserial correlation."),
                              ("kp_alias", "tab_multimodel_alias",
                               "Fact precision (alias-tolerant matcher, sensitivity analysis) by generator model, "
                               "retrieval-augmented versus no-retrieval baseline (run 1), with bootstrap 95\\% CI, "
                               "Wilcoxon signed-rank test and rank-biserial $r$.")):
        rows = []
        for m in d.models:
            e = pm.get(m, {}).get(metric)
            if not e:
                continue
            h1 = "Yes" if e["h1_point_met"] else "No"
            rows.append([label(m), dec(e["rag"]), dec(e["base"]), pp(e["diff"]), ci_pp(e["ci_lo"], e["ci_hi"]),
                         p_str(e["p"]), dec(e["r_rb"]), h1])
        tex_table(name, cap, f"tab:{name.replace('tab_', '').replace('_', '-')}", "lccccccc",
                  ["Model", "RAG", "Baseline", "$\\Delta$ (pp)", "95\\% CI", "$p$", "$r$", "H1 ($\\geq$30 pp)"], rows)

    # Table: hallucination + omission + citation per model
    rows = []
    for m in d.models:
        h = pm.get(m, {}).get("halluc_j1")
        o = pm.get(m, {}).get("omission_j1")
        c = pm.get(m, {}).get("cite_ok")
        if not h:
            continue
        rows.append([label(m), pct(h["rag"]), pct(h["base"]), pp(h["diff"]), p_str(h["p"]),
                     pct(o["rag"]) if o else "--", pct(o["base"]) if o else "--",
                     pct(c["rag"], 0) if c else "--"])
    tex_table("tab_multimodel_halluc",
              "Hallucination (incorrect or fabricated legal claim, judge 1 = GPT-4.1) and omission of a key fact by "
              "generator model, retrieval-augmented versus baseline (run 1, N = 24 paired questions), with the share of "
              "grounded answers citing the expected instrument (no baseline answer cited one). "
              "$p$ from the exact McNemar test on discordant pairs.",
              "tab:multimodel-halluc", "lccccccc",
              ["Model", "Halluc. RAG", "Halluc. base", "$\\Delta$ (pp)", "$p$", "Omit. RAG", "Omit. base",
               "Cited RAG"], rows)

    # Table: primary model, all metrics (June-style table with inference)
    e = pm.get(PRIMARY, {})
    rows = []
    spec = [("kp_strict", "Fact precision (strict)", "Wilcoxon"),
            ("kp_alias", "Fact precision (alias)", "Wilcoxon"),
            ("cite_ok", "Citation validity", "McNemar"),
            ("halluc_j1", "Incorrect claim (judge 1, GPT-4.1)", "McNemar"),
            ("halluc_j2", "Incorrect claim (judge 2, GPT-4o)", "McNemar"),
            ("omission_j1", "Omission (judge 1)", "McNemar"),
            ("rouge_l", "ROUGE-L", "Wilcoxon")]
    for metric, nm, test in spec:
        x = e.get(metric)
        if not x:
            continue
        if metric in ("kp_strict", "kp_alias", "rouge_l"):
            rows.append([nm, dec(x["rag"]), dec(x["base"]), pp(x["diff"]), ci_pp(x["ci_lo"], x["ci_hi"]),
                         test, p_str(x["p"]), dec(x["r_rb"])])
        else:
            rows.append([nm, pct(x["rag"]), pct(x["base"]), pp(x["diff"]), ci_pp(x["ci_lo"], x["ci_hi"]),
                         test, p_str(x["p"]), "--"])
    tex_table("tab_primary_gpt4o",
              "Primary comparison (GPT-4o, run 1, N = 24 paired questions): retrieval-augmented system versus "
              "no-retrieval baseline on every metric, with paired difference, bootstrap 95\\% CI, test and effect size.",
              "tab:primary-gpt4o", "lccccccc",
              ["Metric", "RAG", "Baseline", "$\\Delta$ (pp)", "95\\% CI", "Test", "$p$", "$r$"], rows)

    # Table: stability across runs
    st = S["stability"]
    rows = []
    for m in d.models:
        for c in CONDS:
            x = st.get(m, {}).get(c)
            if not x:
                continue
            kp = x["kp_alias"]
            hl = x["halluc_j1"]
            rows.append([label(m) if c == "RAG" else "", c,
                         f"{dec(kp['mean'])} $\\pm$ {dec(kp['sd'], 3)}" if kp["sd"] is not None else dec(kp["mean"]),
                         f"{pct(hl['mean'])} $\\pm$ {pct(hl['sd'])}" if hl["sd"] is not None else pct(hl["mean"]),
                         pct(x["halluc_flag_consistency"], 0), dec(x["kp_alias_within_question_sd"], 3)])
    tex_table("tab_stability",
              f"Run-to-run stability over {len(d.runs)} independent runs: mean $\\pm$ SD of the per-run cell means "
              "for fact precision (alias) and hallucination (judge 1), the share of questions whose hallucination "
              "flag was identical in every run, and the mean within-question SD of fact precision.",
              "tab:stability", "llcccc",
              ["Model", "Condition", "Fact precision", "Hallucination", "Flag consistency", "Within-question SD"], rows)

    # Table: by language (alias precision + hallucination) per model
    rows = []
    for m in d.models:
        x = S["subgroup_run1"]["by_language"].get(m)
        if not x:
            continue
        en, mt = x.get("en", {}), x.get("mt", {})
        mw = x.get("mannwhitney_gain_en_vs_mt", {})
        rows.append([label(m), dec(en.get("rag")), dec(en.get("base")), pct(en.get("halluc_base")),
                     dec(mt.get("rag")), dec(mt.get("base")), pct(mt.get("halluc_base")), p_str(mw.get("p"))])
    tex_table("tab_bylang",
              "Fact precision (alias) and baseline hallucination rate by language and model (run 1; 12 English and "
              "12 Maltese questions). Last column: Mann--Whitney $U$ test of the per-question retrieval gain, "
              "English versus Maltese.",
              "tab:bylang", "lccccccc",
              ["Model", "EN RAG", "EN base", "EN halluc.", "MT RAG", "MT base", "MT halluc.", "$p$ (gain)"],
              rows)

    # Table: across-model omnibus tests
    am = S["across_models_run1"]
    rows = []
    for key, nm in (("RAG_kp_alias", "Fact precision (alias), RAG condition"),
                    ("Baseline_kp_alias", "Fact precision (alias), baseline condition"),
                    ("RAG_kp_strict", "Fact precision (strict), RAG condition"),
                    ("Baseline_kp_strict", "Fact precision (strict), baseline condition"),
                    ("gain_kp_alias", "Retrieval gain (alias)"), ("gain_kp_strict", "Retrieval gain (strict)")):
        x = am.get(key)
        if not x:
            continue
        fr = x["friedman"]
        rows.append([nm, str(fr["n"]), str(fr["k"]), dec(fr["chi2"]), p_str(fr["p"]), dec(fr["W"])])
    for key, nm in (("Baseline_halluc_j1_cochranQ", "Hallucination (judge 1), baseline condition"),
                    ("RAG_halluc_j1_cochranQ", "Hallucination (judge 1), RAG condition")):
        x = am.get(key)
        if not x:
            continue
        rows.append([nm + " (Cochran's $Q$)", str(x["n"]), str(x["k"]), dec(x["Q"]), p_str(x["p"]), "--"])
    tex_table("tab_across_models",
              "Omnibus tests of differences between the generator models (questions as blocks, run 1): Friedman "
              "$\\chi^2$ with Kendall's $W$, and Cochran's $Q$ for the binary hallucination flag.",
              "tab:across-models", "lccccc", ["Comparison", "$n$", "$k$", "$\\chi^2$ / $Q$", "$p$", "$W$"], rows)

    # Table: OOS
    oo = S.get("oos", {})
    rows = []
    for m in d.models:
        x = oo.get(m)
        if not x or "RAG" not in x:
            continue
        mc = x.get("mcnemar_refusal", {})
        rows.append([label(m), pct(x["RAG"]["refusal"], 0), pct(x["Baseline"]["refusal"], 0), p_str(mc.get("p")),
                     pct(x["RAG"]["asserts_j1"], 0), pct(x["Baseline"]["asserts_j1"], 0),
                     pct(x["RAG"]["mirror"], 0), pct(x["Baseline"]["mirror"], 0)])
    if rows:
        tex_table("tab_oos",
                  "Out-of-scope robustness set (10 questions outside the corpus): share of answers that explicitly "
                  "declined for lack of source coverage (refusal), share asserting specific legal figures or rules "
                  "as fact (judge 1), and language mirroring, by model and condition. $p$ from the exact McNemar test "
                  "on refusal, RAG versus baseline.",
                  "tab:oos", "lccccccc",
                  ["Model", "Refuse RAG", "Refuse base", "$p$", "Assert RAG", "Assert base", "Mirror RAG", "Mirror base"],
                  rows)

    # Table: ablation (retrieval metrics) + separate downstream table
    ab = S.get("ablation", {})
    rows = []
    for emb in ("text-embedding-3-large", "text-embedding-3-small"):
        x = ab.get("retrieval", {}).get(emb)
        if not x:
            continue
        for c in CONFIG_ORDER:
            y = x.get(c)
            if not y:
                continue
            rows.append([emb.replace("text-embedding-3-", "3-") if c == CONFIG_ORDER[0] else "", CONFIG_LABEL[c],
                         dec(y["hit1"]), dec(y["hit5"]), dec(y["mrr"]), dec(y["mrr_en"]), dec(y["mrr_mt"]),
                         dec(y["latency_s"])])
    if rows:
        tex_table("tab_ablation",
                  "Retrieval ablation over the 24 gold queries: Hit@1, Hit@5 and mean reciprocal rank (MRR) of the "
                  "expected statutory provision for four retrieval configurations and two embedding models, with mean "
                  "retrieval latency.",
                  "tab:ablation", "llcccccc",
                  ["Embedder", "Configuration", "Hit@1", "Hit@5", "MRR", "MRR EN", "MRR MT", "Latency (s)"], rows)
    rows = []
    for c in CONFIG_ORDER:
        ds = ab.get("downstream", {}).get(c)
        if not ds:
            continue
        rows.append([CONFIG_LABEL[c], dec(ds["kp_strict"]), dec(ds["kp_alias"]), pct(ds["citation_rate"], 0),
                     pct(ds["hallucination"]), pct(ds["omission"]),
                     "--" if ds.get("vs_deployed_p_holm") is None else p_str(ds["vs_deployed_p_holm"])])
    if rows:
        fr = ab.get("downstream", {}).get("friedman_kp_alias") or {}
        tex_table("tab_ablation_downstream",
                  "Downstream answer quality when each retrieval configuration (large embedder) supplies the context "
                  "for a GPT-4o answer to the 24 gold questions: fact precision under both matchers, citation validity, "
                  "incorrect-claim and omission rates (judge 1), and the Holm-corrected Wilcoxon $p$ of each "
                  "configuration against the deployed one on alias precision"
                  + (f" (Friedman $\\chi^2$(3) = {fr['chi2']:.2f}, $p$ = {fr['p']:.2f})." if fr.get("chi2") else "."),
                  "tab:ablation-downstream", "lcccccc",
                  ["Configuration", "Strict", "Alias", "Cited", "Halluc.", "Omission", "$p$ vs deployed"], rows)

    # Table: cost/latency
    co = S["cost"]
    rows = []
    for m in d.models:
        x = co.get(m)
        if not x or "RAG" not in x:
            continue
        r_, b_ = x["RAG"], x["Baseline"]
        usd = r_["usd_per_1000_queries"]
        rows.append([label(m), x["snapshot"], dec(r_["latency_mean"]), dec(b_["latency_mean"]),
                     f"{int(r_['tokens_in'] or 0)}/{int(r_['tokens_out'] or 0)}",
                     f"{int(b_['tokens_in'] or 0)}/{int(b_['tokens_out'] or 0)}",
                     "--" if usd is None else f"{usd:.2f}"])
    tex_table("tab_cost",
              "Served model snapshot, mean response latency, mean prompt/completion tokens per query and indicative "
              "API cost per 1000 grounded queries (list prices at the time of the evaluation), by model.",
              "tab:cost", "llccccc",
              ["Model", "Snapshot", "Latency RAG (s)", "Latency base (s)", "Tokens RAG (in/out)",
               "Tokens base (in/out)", "USD / 1000 RAG queries"], rows, wide=True)

    # Table: LLM judge agreement
    ja = S["judge_llm_agreement"]
    rows = []
    for metric, nm in (("halluc", "Incorrect claim"), ("omission", "Omission")):
        x = ja.get(metric)
        if not x or not x["n"]:
            continue
        rows.append([nm, "All answers", str(x["n"]), dec(x["kappa"]), pct(x["agreement"]), pct(x["rate_j1"]), pct(x["rate_j2"])])
        for c in CONDS:
            y = x.get(c, {})
            if y.get("n"):
                rows.append(["", c, str(y["n"]), dec(y["kappa"]), pct(y["agreement"]), pct(y["rate_j1"]), pct(y["rate_j2"])])
    tex_table("tab_judge_llm",
              "Agreement between the two LLM judges (judge 1 = GPT-4.1, judge 2 = GPT-4o) over every generated answer: "
              "Cohen's $\\kappa$, percentage agreement and the rate flagged by each judge.",
              "tab:judge-llm", "llccccc",
              ["Criterion", "Subset", "$n$", "$\\kappa$", "Agreement", "Rate judge 1", "Rate judge 2"], rows)


# ---------------------------------------------------------------------------

def main() -> int:
    rows = load(RESULTS)
    if not rows:
        sys.exit(f"no results at {RESULTS}")
    d = Data(rows)
    print(f"{len(rows)} rows; models={d.models}; runs={d.runs}; questions={len(d.ids)}")
    S = {
        "meta": {"n_rows": len(rows), "models": d.models, "runs": d.runs, "n_questions": len(d.ids),
                 "primary_model": PRIMARY, "h1_threshold": H1_THRESHOLD, "bootstrap_B": B, "seed": SEED},
        "per_model_run1": per_model(d, "1"),
        "per_model_runavg": per_model(d, None),
        "across_models_run1": across_models(d, "1"),
        "across_models_runavg": across_models(d, None),
        "subgroup_run1": subgroup(d, "1"),
        "stability": stability(d),
        "judge_llm_agreement": judge_llm_agreement(d),
        "oos": oos_stats(load(OOS), d.models),
        "ablation": ablation_stats(load(ABL_RET), load(ABL_DS)),
        "cost": cost(d),
    }
    (EVAL_DIR / "stats_summary.json").write_text(json.dumps(S, indent=2, default=float), encoding="utf-8")
    write_tables(S, d)
    # console digest
    print("\nPer-model (run 1): alias precision RAG/Base, gain pp [CI], p, r | halluc j1 RAG/Base, McNemar p")
    for m in d.models:
        e = S["per_model_run1"].get(m, {})
        k, h = e.get("kp_alias"), e.get("halluc_j1")
        if k and h:
            print(f"  {label(m):<14} {k['rag']:.3f}/{k['base']:.3f} {100*k['diff']:+.1f} [{100*k['ci_lo']:+.1f},{100*k['ci_hi']:+.1f}] "
                  f"p={k['p']:.4f} r={k['r_rb']:.2f} | {h['rag']:.3f}/{h['base']:.3f} p={h['p']:.4f}")
    for key in ("RAG_kp_alias", "Baseline_kp_alias", "gain_kp_alias"):
        x = S["across_models_run1"].get(key)
        if x:
            fr = x["friedman"]
            print(f"Friedman {key}: chi2={fr['chi2']} p={fr['p']} W={fr['W']}")
    print(f"wrote stats_summary.json and {len(list(TABLES.glob('*.tex')))} table fragments to {TABLES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
