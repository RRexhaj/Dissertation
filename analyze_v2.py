"""analyze_v2.py — deeper statistics for the within-subjects trust study (resubmission, Sep 2026).

Builds on analyze.py (same loader, same composite) and adds what the first
submission lacked:
  * scale reliability: Cronbach's alpha (pooled and per condition) and corrected
    item-total correlations for the five-item composite;
  * medians and IQRs beside means; pairwise effect sizes (matched-pairs
    rank-biserial r) and percentile-bootstrap 95% CIs on every paired difference;
  * serial-position checks: composite by position irrespective of condition,
    position 1 vs 2 within the two plain conditions (paired), and whether the
    transparency answer at position 3 vs 4 differs;
  * scenario check: composite by scenario within each condition (Kruskal-Wallis);
  * exploratory moderators of the transparency premium (A - B): legal
    familiarity (Spearman), AI use (Mann-Whitney), first language, age;
  * robustness: the primary test re-run on the alternative screening files;
  * LaTeX table fragments and figures in the resubmission palette.

Run (inside the MalteseBot venv, from Dissertation_repo):
    python analyze_v2.py                      # reads cleaned_primary.csv
Outputs: analysis_summary_v2.json, tables/tab_us_*.tex, figures fig_us_*.png + regenerated
         fig_trust_by_condition.png / fig_trust_calibration.png
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analyze import COMPOSITE_ITEMS, CONDITION_ORDER, kendalls_w, load  # noqa: E402

PRIMARY_CSV = HERE / "cleaned_primary.csv"
ALT_CSVS = {"final_version_only": HERE / "cleaned_final_version.csv",
            "keep_straightliners": HERE / "cleaned_keep_straightliners.csv"}
TABLES = HERE / "tables"
FIG_DIR = HERE / "figures"
A, B, C = CONDITION_ORDER
SHORT = {A: "Transparency", B: "Plain (accurate)", C: "Baseline"}
ITEMS = ["t_reliable", "t_accurate", "t_confident", "t_wary_r", "t_rely", "t_source", "t_understand", "t_use"]
ITEM_LABEL = {"t_reliable": "Reliable", "t_accurate": "Accurate", "t_confident": "Confident to act",
              "t_wary_r": "Not suspicious (reverse-coded)", "t_rely": "Would rely on before a lawyer",
              "t_source": "Source clear", "t_understand": "Understood rights", "t_use": "Would use"}
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
COL = {A: BLUE, B: ORANGE, C: AQUA}
B_BOOT, SEED = 10_000, 20260906
rng = np.random.default_rng(SEED)

plt.rcParams.update({
    "font.family": ["Arial", "DejaVu Sans"], "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True, "legend.frameon": False,
    "figure.dpi": 200, "savefig.dpi": 200})


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def wilcoxon_rb(x, y) -> dict:
    x, y = np.asarray(x, float), np.asarray(y, float)
    d = x - y
    nz = d[d != 0]
    out = {"n": int(len(d)), "n_nonzero": int(len(nz)), "mean_diff": float(d.mean())}
    if len(nz) == 0:
        out.update({"W": 0.0, "p": 1.0, "r_rb": 0.0})
    else:
        try:
            res = stats.wilcoxon(x, y)
            out["W"], out["p"] = float(res.statistic), float(res.pvalue)
        except ValueError:
            out["W"], out["p"] = float("nan"), 1.0
        ranks = stats.rankdata(np.abs(nz))
        rp, rn = ranks[nz > 0].sum(), ranks[nz < 0].sum()
        out["r_rb"] = float((rp - rn) / (rp + rn)) if (rp + rn) else 0.0
    idx = rng.integers(0, len(d), size=(B_BOOT, len(d)))
    means = d[idx].mean(axis=1)
    out["ci_lo"], out["ci_hi"] = float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
    return out


def boot_ci_mean(vals) -> tuple[float, float]:
    vals = np.asarray(vals, float)
    idx = rng.integers(0, len(vals), size=(B_BOOT, len(vals)))
    means = vals[idx].mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def cronbach_alpha(df_items: pd.DataFrame) -> float:
    k = df_items.shape[1]
    item_var = df_items.var(axis=0, ddof=1).sum()
    total_var = df_items.sum(axis=1).var(ddof=1)
    return float(k / (k - 1) * (1 - item_var / total_var)) if total_var > 0 else float("nan")


def item_total(df_items: pd.DataFrame) -> dict:
    out = {}
    for c in df_items.columns:
        rest = df_items.drop(columns=c).mean(axis=1)
        out[c] = float(stats.pearsonr(df_items[c], rest)[0])
    return out


def p_str(p) -> str:
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "--"
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def tex_table(name: str, caption: str, label: str, colspec: str, header: list[str], rows: list[list[str]],
              wide: bool = False) -> None:
    TABLES.mkdir(exist_ok=True)
    lines = ["\\begin{table}[H]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             "{\\def\\arraystretch{1.3}"]
    if wide:
        lines.append("\\resizebox{\\textwidth}{!}{%")
    lines += [f"\\begin{{tabular}}{{{colspec}}}", "\\toprule",
              " & ".join(f"\\textbf{{{h}}}" for h in header) + " \\\\", "\\midrule"]
    lines += [" & ".join(r) + " \\\\" for r in rows]
    lines += ["\\bottomrule", "\\end{tabular}" + ("}" if wide else ""), "}", "\\end{table}"]
    (TABLES / f"{name}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def wide_table(scen: pd.DataFrame, value: str) -> pd.DataFrame:
    core = scen[scen["is_probe"] == 0]
    w = core.pivot_table(index="participant_id", columns="condition", values=value, aggfunc="mean")
    return w.reindex(columns=CONDITION_ORDER).dropna()


# ---------------------------------------------------------------------------
# analyses
# ---------------------------------------------------------------------------

def omnibus(scen: pd.DataFrame, value: str) -> dict:
    w = wide_table(scen, value)
    out = {"n": int(len(w)), "descriptives": {}}
    for c in CONDITION_ORDER:
        col = w[c]
        lo, hi = boot_ci_mean(col.values)
        out["descriptives"][c] = {"mean": float(col.mean()), "sd": float(col.std(ddof=1)),
                                  "median": float(col.median()), "iqr": [float(col.quantile(.25)), float(col.quantile(.75))],
                                  "ci_lo": lo, "ci_hi": hi}
    chi2, p = stats.friedmanchisquare(*[w[c] for c in CONDITION_ORDER])
    out["friedman"] = {"chi2": float(chi2), "df": 2, "p": float(p), "kendalls_w": kendalls_w(float(chi2), len(w), 3)}
    pairs = {}
    for i, j in ((0, 1), (0, 2), (1, 2)):
        a, b = CONDITION_ORDER[i], CONDITION_ORDER[j]
        r = wilcoxon_rb(w[a], w[b])
        r["p_bonferroni"] = min(1.0, r["p"] * 3)
        pairs[f"{a} vs {b}"] = r
    out["pairwise"] = pairs
    return out


def reliability(scen: pd.DataFrame) -> dict:
    core = scen[scen["is_probe"] == 0]
    items = core[COMPOSITE_ITEMS]
    out = {"pooled": {"n_ratings": int(len(items)), "alpha": cronbach_alpha(items), "item_total": item_total(items)}}
    for c in CONDITION_ORDER:
        sub = core.loc[core["condition"] == c, COMPOSITE_ITEMS]
        out[c] = {"n_ratings": int(len(sub)), "alpha": cronbach_alpha(sub)}
    return out


def calibration(scen: pd.DataFrame) -> dict:
    high = scen[(scen["condition"] == A) & (scen["is_probe"] == 0)].groupby("participant_id")["composite"].mean()
    low = scen[scen["is_probe"] == 1].groupby("participant_id")["composite"].mean()
    paired = pd.concat([high.rename("high"), low.rename("low")], axis=1).dropna()
    r = wilcoxon_rb(paired["high"], paired["low"])
    r.update({"mean_high": float(paired["high"].mean()), "mean_low": float(paired["low"].mean()),
              "sd_high": float(paired["high"].std(ddof=1)), "sd_low": float(paired["low"].std(ddof=1))})
    return r


def order_checks(scen: pd.DataFrame) -> dict:
    out = {}
    pos = pd.to_numeric(scen["order_position"], errors="coerce")
    scen = scen.assign(pos=pos)
    by_pos = scen.groupby("pos")["composite"].agg(["mean", "std", "count"])
    out["composite_by_position_all"] = {int(k): {"mean": float(v["mean"]), "sd": float(v["std"]), "n": int(v["count"])}
                                        for k, v in by_pos.iterrows()}
    # position 1 vs 2 within the plain conditions (each participant has one of each)
    plain = scen[(scen["condition"].isin([B, C])) & (scen["is_probe"] == 0)]
    w = plain.pivot_table(index="participant_id", columns="pos", values="composite", aggfunc="mean")
    if 1 in w.columns and 2 in w.columns:
        w12 = w[[1, 2]].dropna()
        out["plain_pos1_vs_pos2"] = wilcoxon_rb(w12[2], w12[1]) | {"mean_pos1": float(w12[1].mean()),
                                                                     "mean_pos2": float(w12[2].mean())}
    # does it matter which plain condition came first? B-C difference by order
    first = plain[plain["pos"] == 1].set_index("participant_id")["condition"]
    wbc = wide_table(scen, "composite")
    diff = (wbc[B] - wbc[C]).rename("b_minus_c").to_frame().join(first.rename("first"))
    g1, g2 = diff.loc[diff["first"] == B, "b_minus_c"], diff.loc[diff["first"] == C, "b_minus_c"]
    if len(g1) > 1 and len(g2) > 1:
        u = stats.mannwhitneyu(g1, g2, alternative="two-sided")
        out["b_minus_c_by_which_first"] = {"B_first": {"n": int(len(g1)), "mean": float(g1.mean())},
                                           "C_first": {"n": int(len(g2)), "mean": float(g2.mean())},
                                           "U": float(u.statistic), "p": float(u.pvalue)}
    # transparency answer at position 3 vs 4 (between participants)
    tr = scen[(scen["condition"] == A) & (scen["is_probe"] == 0)]
    g3, g4 = tr.loc[tr["pos"] == 3, "composite"], tr.loc[tr["pos"] == 4, "composite"]
    if len(g3) > 1 and len(g4) > 1:
        u = stats.mannwhitneyu(g3, g4, alternative="two-sided")
        out["transparency_pos3_vs_pos4"] = {"pos3": {"n": int(len(g3)), "mean": float(g3.mean())},
                                            "pos4": {"n": int(len(g4)), "mean": float(g4.mean())},
                                            "U": float(u.statistic), "p": float(u.pvalue)}
    return out


def scenario_checks(scen: pd.DataFrame) -> dict:
    out = {}
    core = scen[scen["is_probe"] == 0]
    for c in CONDITION_ORDER:
        sub = core[core["condition"] == c]
        groups = {k: g["composite"].values for k, g in sub.groupby("scenario_key")}
        entry = {k: {"n": int(len(v)), "mean": float(np.mean(v))} for k, v in groups.items()}
        if len(groups) >= 2 and all(len(v) > 1 for v in groups.values()):
            kw = stats.kruskal(*groups.values())
            entry["kruskal"] = {"H": float(kw.statistic), "p": float(kw.pvalue)}
        out[c] = entry
    return out


def moderators(scen: pd.DataFrame) -> dict:
    w = wide_table(scen, "composite")
    prem = (w[A] - w[B]).rename("premium")
    one = scen.drop_duplicates("participant_id").set_index("participant_id")
    df = prem.to_frame().join(one[["law_familiarity", "ai_use_freq", "native_language", "age_band"]])
    out = {"premium_mean": float(prem.mean()), "premium_sd": float(prem.std(ddof=1))}
    fam = pd.to_numeric(df["law_familiarity"], errors="coerce")
    ok = fam.notna()
    if ok.sum() > 3:
        rho, p = stats.spearmanr(fam[ok], df.loc[ok, "premium"])
        out["law_familiarity_spearman"] = {"rho": float(rho), "p": float(p), "n": int(ok.sum())}
    heavy = df["ai_use_freq"].isin(["Daily", "Weekly"])
    if heavy.sum() > 1 and (~heavy).sum() > 1:
        u = stats.mannwhitneyu(df.loc[heavy, "premium"], df.loc[~heavy, "premium"], alternative="two-sided")
        out["ai_use_heavy_vs_light"] = {"heavy": {"n": int(heavy.sum()), "mean": float(df.loc[heavy, "premium"].mean())},
                                        "light": {"n": int((~heavy).sum()), "mean": float(df.loc[~heavy, "premium"].mean())},
                                        "U": float(u.statistic), "p": float(u.pvalue)}
    eng = df["native_language"] == "English"
    if eng.sum() > 1 and (~eng).sum() > 1:
        u = stats.mannwhitneyu(df.loc[eng, "premium"], df.loc[~eng, "premium"], alternative="two-sided")
        out["first_language_english_vs_other"] = {"english": {"n": int(eng.sum()), "mean": float(df.loc[eng, "premium"].mean())},
                                                  "other": {"n": int((~eng).sum()), "mean": float(df.loc[~eng, "premium"].mean())},
                                                  "U": float(u.statistic), "p": float(u.pvalue)}
    young = df["age_band"].isin(["18–24", "18-24", "25–34", "25-34"])
    if young.sum() > 1 and (~young).sum() > 1:
        u = stats.mannwhitneyu(df.loc[young, "premium"], df.loc[~young, "premium"], alternative="two-sided")
        out["age_under35_vs_35plus"] = {"under35": {"n": int(young.sum()), "mean": float(df.loc[young, "premium"].mean())},
                                        "35plus": {"n": int((~young).sum()), "mean": float(df.loc[~young, "premium"].mean())},
                                        "U": float(u.statistic), "p": float(u.pvalue)}
    return out


def closing_items(final: pd.DataFrame) -> dict:
    """The live app wrote the two closing answers two columns to the left of their headers
    for some rows, so scan every string cell for the known option texts."""
    notice_opts = ["Yes, clearly", "Yes, somewhat", "No, I did not notice"]
    most_opts = ["Answers that showed sources and confidence", "Answers without sources or confidence",
                 "I trusted them about the same"]
    notice, most = {}, {}
    for _, row in final.iterrows():
        cells = [str(v).strip() for v in row.values if isinstance(v, str)]
        for o in notice_opts:
            if any(c == o or c.startswith(o) for c in cells):
                notice[o] = notice.get(o, 0) + 1
                break
        for o in most_opts:
            if any(c == o or c.startswith(o[:30]) for c in cells):
                most[o] = most.get(o, 0) + 1
                break
    return {"n_final_rows": int(len(final)), "noticed": notice, "most_trusted": most}


# ---------------------------------------------------------------------------
# tables + figures
# ---------------------------------------------------------------------------

def write_tables(S: dict) -> None:
    d = S["composite"]["descriptives"]
    fr = S["composite"]["friedman"]
    rows = [[SHORT[c], f"{d[c]['mean']:.2f}", f"{d[c]['sd']:.2f}", f"{d[c]['median']:.2f}",
             f"{d[c]['iqr'][0]:.2f}--{d[c]['iqr'][1]:.2f}", f"[{d[c]['ci_lo']:.2f}, {d[c]['ci_hi']:.2f}]"]
            for c in CONDITION_ORDER]
    tex_table("tab_us_descriptives",
              f"Composite trust (1--5) by condition: mean, SD, median, interquartile range and bootstrap 95\\% CI "
              f"of the mean; Friedman $\\chi^2$(2) = {fr['chi2']:.2f}, $p$ {p_str(fr['p']).replace('$<$', '$<$ ')}, "
              f"Kendall's $W$ = {fr['kendalls_w']:.2f} (N = {S['composite']['n']}).",
              "tab:us-descriptives", "lccccc", ["Condition", "Mean", "SD", "Median", "IQR", "95\\% CI"], rows)
    rows = []
    for name, r in S["composite"]["pairwise"].items():
        a, b = name.split(" vs ")
        rows.append([f"{SHORT[a]} vs {SHORT[b]}", f"{r['mean_diff']:+.2f}", f"[{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}]",
                     f"{r['W']:.1f}", p_str(r["p_bonferroni"]), f"{r['r_rb']:.2f}"])
    tex_table("tab_us_pairwise",
              "Pairwise comparisons of composite trust: mean paired difference with bootstrap 95\\% CI, Wilcoxon "
              "signed-rank $W$, Bonferroni-corrected $p$ over three comparisons, and matched-pairs rank-biserial $r$ (N = 20).",
              "tab:us-pairwise", "lccccc", ["Comparison", "$\\Delta$", "95\\% CI", "$W$", "$p$ (corr.)", "$r$"], rows)
    rel = S["reliability"]
    rows = [["All conditions pooled", str(rel["pooled"]["n_ratings"]), f"{rel['pooled']['alpha']:.2f}"]]
    rows += [[SHORT[c], str(rel[c]["n_ratings"]), f"{rel[c]['alpha']:.2f}"] for c in CONDITION_ORDER]
    it = rel["pooled"]["item_total"]
    rows += [[f"Item--total: {ITEM_LABEL[k]}", "", f"{v:.2f}"] for k, v in it.items()]
    tex_table("tab_us_reliability",
              "Internal consistency of the five-item composite trust scale: Cronbach's $\\alpha$ pooled and per "
              "condition, and corrected item--total correlations (pooled ratings, probe excluded).",
              "tab:us-reliability", "lcc", ["Scale / item", "Ratings", "$\\alpha$ / $r_{it}$"], rows)
    rows = []
    for key, label in (("source", "Source clear (manipulation check)"), ("understand", "Understood rights"),
                       ("use", "Would use")):
        o = S[key]
        dd = o["descriptives"]
        f2 = o["friedman"]
        pw = o["pairwise"]
        rows.append([label, f"{dd[A]['mean']:.2f}", f"{dd[B]['mean']:.2f}", f"{dd[C]['mean']:.2f}",
                     f"{f2['chi2']:.2f}", p_str(f2["p"]), f"{f2['kendalls_w']:.2f}",
                     p_str(pw[f'{A} vs {B}']['p_bonferroni']), p_str(pw[f'{A} vs {C}']['p_bonferroni']),
                     p_str(pw[f'{B} vs {C}']['p_bonferroni'])])
    tex_table("tab_us_secondary",
              "Single-item outcomes by condition (means, 1--5), Friedman test with Kendall's $W$, and "
              "Bonferroni-corrected pairwise $p$-values (N = 20).",
              "tab:us-secondary", "lccccccccc",
              ["Item", "Transp.", "Plain", "Base", "$\\chi^2$", "$p$", "$W$", "T vs P", "T vs B", "P vs B"], rows, wide=True)
    oc = S["order"]
    rows = [[f"Position {k}", f"{v['mean']:.2f}", f"{v['sd']:.2f}", str(v["n"])]
            for k, v in sorted(oc["composite_by_position_all"].items())]
    tex_table("tab_us_order",
              "Composite trust by serial position, all answers pooled regardless of condition (positions 1--2 "
              "always plain answers; positions 3--4 the transparency answer and the low-confidence probe).",
              "tab:us-order", "lccc", ["Position", "Mean", "SD", "Ratings"], rows)
    mod = S["moderators"]
    rows = []
    if "law_familiarity_spearman" in mod:
        m = mod["law_familiarity_spearman"]
        rows.append(["Self-rated legal familiarity (1--5)", f"Spearman $\\rho$ = {m['rho']:.2f}", p_str(m["p"])])
    for key, lab in (("ai_use_heavy_vs_light", "AI use: daily/weekly vs less"),
                     ("first_language_english_vs_other", "First language: English vs other"),
                     ("age_under35_vs_35plus", "Age: under 35 vs 35 and over")):
        if key in mod:
            m = mod[key]
            g = [k for k in m if isinstance(m[k], dict)]
            rows.append([lab, f"{m[g[0]]['mean']:+.2f} (n={m[g[0]]['n']}) vs {m[g[1]]['mean']:+.2f} (n={m[g[1]]['n']}); "
                         f"$U$ = {m['U']:.1f}", p_str(m["p"])])
    tex_table("tab_us_moderators",
              f"Exploratory moderators of the transparency premium (composite trust, transparency minus plain; "
              f"mean {mod['premium_mean']:+.2f}, SD {mod['premium_sd']:.2f}).",
              "tab:us-moderators", "lcc", ["Moderator", "Statistic", "$p$"], rows)


def figures(scen: pd.DataFrame, S: dict) -> None:
    FIG_DIR.mkdir(exist_ok=True)
    d = S["composite"]["descriptives"]
    # bars with bootstrap CI (regenerated in the resubmission palette)
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    x = np.arange(3)
    means = [d[c]["mean"] for c in CONDITION_ORDER]
    err = np.array([[d[c]["mean"] - d[c]["ci_lo"], d[c]["ci_hi"] - d[c]["mean"]] for c in CONDITION_ORDER]).T
    ax.bar(x, means, 0.55, color=[COL[c] for c in CONDITION_ORDER], linewidth=0, zorder=3)
    ax.errorbar(x, means, yerr=err, fmt="none", ecolor=INK2, elinewidth=0.9, capsize=3, zorder=4)
    for xi, m, c in zip(x, means, CONDITION_ORDER):
        ax.text(xi, d[c]["ci_hi"] + 0.05, f"{m:.2f}", ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels(["Transparency\n(A)", "Plain, accurate\n(B)", "Baseline\n(C)"])
    ax.set_ylim(1, 5.3)
    ax.set_ylabel("Mean composite trust (1–5), 95% bootstrap CI")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_trust_by_condition.png")
    plt.close(fig)
    # calibration
    cal = S["calibration"]
    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    vals = [cal["mean_high"], cal["mean_low"]]
    ax.bar([0, 1], vals, 0.55, color=[BLUE, ORANGE], linewidth=0, zorder=3)
    ax.errorbar([0, 1], vals, yerr=[cal["sd_high"], cal["sd_low"]], fmt="none", ecolor=INK2, elinewidth=0.9, capsize=3, zorder=4)
    for xi, v in zip([0, 1], vals):
        ax.text(xi, v + 0.65, f"{v:.2f}", ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["High-confidence\ntransparency answer", "Low-confidence\nprobe"])
    ax.set_ylim(1, 5.3)
    ax.set_ylabel("Mean composite trust (1–5), ±SD")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_trust_calibration.png")
    plt.close(fig)
    # per-participant paired lines
    w = wide_table(scen, "composite")
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for _, row in w.iterrows():
        ax.plot(x, [row[c] for c in CONDITION_ORDER], color=INK2, alpha=0.25, linewidth=1, zorder=2)
    ax.plot(x, means, color=INK, linewidth=2, marker="o", markersize=6, zorder=4, label="Mean")
    for xi, c in zip(x, CONDITION_ORDER):
        ax.scatter([xi], [d[c]["mean"]], color=COL[c], s=60, zorder=5)
    ax.set_xticks(x)
    ax.set_xticklabels(["Transparency (A)", "Plain, accurate (B)", "Baseline (C)"])
    ax.set_ylim(0.8, 5.2)
    ax.set_ylabel("Composite trust (1–5)")
    ax.set_title("Each line is one participant (N = 20)", loc="left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_us_paired.png")
    plt.close(fig)
    # item-level dot plot
    core = scen[scen["is_probe"] == 0]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.grid(axis="x", color=GRID)
    ax.grid(axis="y", visible=False)
    y = np.arange(len(ITEMS))[::-1]
    for c in CONDITION_ORDER:
        vals = [core.loc[core["condition"] == c, it].mean() for it in ITEMS]
        ax.scatter(vals, y, color=COL[c], s=48, zorder=4, label=SHORT[c])
    ax.set_yticks(y)
    ax.set_yticklabels([ITEM_LABEL[i] for i in ITEMS])
    ax.set_xlim(1, 5)
    ax.set_xlabel("Mean rating (1–5)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_us_items.png")
    plt.close(fig)
    # order-position plot
    oc = S["order"]["composite_by_position_all"]
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ks = sorted(oc)
    ax.errorbar(ks, [oc[k]["mean"] for k in ks], yerr=[oc[k]["sd"] for k in ks], fmt="o-", color=BLUE,
                ecolor=INK2, capsize=3, linewidth=1.6, markersize=6, zorder=4)
    for k in ks:
        ax.text(k, oc[k]["mean"] + 0.75, f"{oc[k]['mean']:.2f}", ha="center", fontsize=8.5)
    ax.set_xticks(ks)
    ax.set_xticklabels([f"{k}\n{'plain' if k <= 2 else 'transparent'}" for k in ks])
    ax.set_xlabel("Serial position of the answer")
    ax.set_ylabel("Composite trust (1–5), ±SD")
    ax.set_ylim(1, 5.4)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_us_order.png")
    plt.close(fig)
    print("wrote figures to", FIG_DIR)


# ---------------------------------------------------------------------------

def main() -> int:
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else PRIMARY_CSV
    scen, final, info = load(csv_path, keep_failed=True)
    S = {
        "meta": info | {"csv": csv_path.name, "bootstrap_B": B_BOOT, "seed": SEED},
        "composite": omnibus(scen, "composite"),
        "source": omnibus(scen, "t_source"),
        "understand": omnibus(scen, "t_understand"),
        "use": omnibus(scen, "t_use"),
        "reliability": reliability(scen),
        "calibration": calibration(scen),
        "order": order_checks(scen),
        "scenario": scenario_checks(scen),
        "moderators": moderators(scen),
        "closing": closing_items(final),
        "robustness": {},
    }
    core = scen[scen["is_probe"] == 0]
    S["items_by_condition"] = {it: {c: float(core.loc[core["condition"] == c, it].mean()) for c in CONDITION_ORDER}
                               for it in ITEMS}
    for name, path in ALT_CSVS.items():
        if path.exists():
            s2, _, i2 = load(path, keep_failed=True)
            o = omnibus(s2, "composite")
            S["robustness"][name] = {"n": o["n"], "friedman": o["friedman"],
                                     "means": {c: o["descriptives"][c]["mean"] for c in CONDITION_ORDER},
                                     "pairwise_p_bonf": {k: v["p_bonferroni"] for k, v in o["pairwise"].items()}}
    (HERE / "analysis_summary_v2.json").write_text(json.dumps(S, indent=2, ensure_ascii=False, default=float),
                                                   encoding="utf-8")
    write_tables(S)
    figures(scen, S)
    c = S["composite"]
    print(f"N={c['n']} Friedman chi2={c['friedman']['chi2']:.2f} p={c['friedman']['p']:.4f} W={c['friedman']['kendalls_w']:.2f}")
    for k, v in c["pairwise"].items():
        print(f"  {k}: diff={v['mean_diff']:+.2f} [{v['ci_lo']:+.2f},{v['ci_hi']:+.2f}] p_bonf={v['p_bonferroni']:.4f} r={v['r_rb']:.2f}")
    print("alpha pooled:", round(S["reliability"]["pooled"]["alpha"], 3))
    print("order:", json.dumps(S["order"], indent=1)[:600])
    print("closing:", S["closing"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
