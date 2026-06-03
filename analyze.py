"""analyze.py — Within-subjects analysis for the MalteseLegalBot trust study.

The survey (app.py) is a WITHIN-SUBJECTS, counterbalanced design: every
participant rates all three presentation conditions (each on a different
scenario) plus one low-confidence calibration probe. This script:

  * loads the responses CSV (local backup) or any export of the Google Sheet
    saved with the same column headers;
  * reverse-codes the suspicion item and builds a composite trust score;
  * (optionally) drops participants who failed the embedded attention check;
  * PRIMARY analysis — Friedman test across the three conditions (the
    non-parametric repeated-measures test, appropriate for ordinal Likert data
    and small samples) + Kendall's W effect size + pairwise Wilcoxon with
    Bonferroni correction;
  * CALIBRATION analysis — paired Wilcoxon comparing trust in the
    high-confidence Transparency answer vs the low-confidence probe (does
    transparency help users *appropriately* lower trust?);
  * MANIPULATION CHECK — the source-transparency item across conditions;
  * descriptive + demographic summaries; and
  * writes analysis_summary.json and figures.

Run (inside the project venv):
    python analyze.py                       # uses results/user_study_responses.csv
    python analyze.py path\\to\\responses.csv
    python analyze.py --keep-failed         # do NOT drop attention-check failures

Requires: pandas, scipy, matplotlib (already in the project venv).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
DEFAULT_CSV = HERE / "results" / "user_study_responses.csv"

CONDITION_ORDER = ["RAG + Transparency", "RAG (no transparency)", "Baseline"]

# Composite = core trust/reliance items (t_wary reverse-coded). t_source is the
# manipulation-check item and t_understand / t_use are reported separately.
COMPOSITE_ITEMS = ["t_reliable", "t_accurate", "t_confident", "t_wary_r", "t_rely"]
ALL_NUMERIC = ["t_reliable", "t_accurate", "t_confident", "t_wary",
               "t_source", "t_understand", "t_use", "t_rely"]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load(csv_path: Path, keep_failed: bool) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if not csv_path.exists():
        sys.exit(
            f"No responses file at {csv_path}.\n"
            "Collect responses first (or export the Google Sheet to that path with the "
            "same column headers), then re-run."
        )
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    if "row_type" not in df.columns:
        sys.exit(
            "This CSV looks like the OLD between-subjects schema (no 'row_type' column).\n"
            "Re-export from the current within-subjects survey, or analyse it with the "
            "previous version of this script."
        )

    scen = df[df["row_type"] == "scenario"].copy()
    final = df[df["row_type"] == "final"].copy()

    for col in ALL_NUMERIC:
        scen[col] = pd.to_numeric(scen[col], errors="coerce")
    scen = scen.dropna(subset=ALL_NUMERIC)
    scen["t_wary_r"] = 6 - scen["t_wary"]          # reverse-code suspicion
    scen["composite"] = scen[COMPOSITE_ITEMS].mean(axis=1)
    scen["condition"] = scen["condition"].astype(str).str.strip()
    scen["is_probe"] = pd.to_numeric(scen["is_probe"], errors="coerce").fillna(0).astype(int)

    # Attention-check filtering (only if the column exists in this export).
    info = {"n_participants_raw": int(scen["participant_id"].nunique())}
    if "attention_pass" in scen.columns:
        ap = pd.to_numeric(scen["attention_pass"], errors="coerce")
        failed = set(scen.loc[ap == 0, "participant_id"])
        info["n_failed_attention"] = len(failed)
        if failed and not keep_failed:
            scen = scen[~scen["participant_id"].isin(failed)]
            final = final[~final["participant_id"].isin(failed)]
            info["attention_filter"] = f"excluded {len(failed)} participant(s) who failed the attention check"
        else:
            info["attention_filter"] = "none applied" if not failed else "kept (--keep-failed)"
    else:
        info["attention_filter"] = "no attention-check column in export"
    info["n_participants_analysed"] = int(scen["participant_id"].nunique())
    return scen, final, info


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def kendalls_w(chi2: float, n: int, k: int) -> float:
    return round(chi2 / (n * (k - 1)), 3) if n and k > 1 else 0.0


def repeated_measures(scen: pd.DataFrame, value: str) -> dict:
    """Friedman across the 3 conditions on the rotated scenarios (probe excluded)."""
    core = scen[scen["is_probe"] == 0]
    wide = core.pivot_table(index="participant_id", columns="condition",
                            values=value, aggfunc="mean")
    wide = wide.reindex(columns=CONDITION_ORDER).dropna()
    out = {
        "n_complete_participants": int(len(wide)),
        "mean_per_condition": {c: round(float(wide[c].mean()), 3) for c in CONDITION_ORDER} if len(wide) else {},
        "sd_per_condition": {c: round(float(wide[c].std(ddof=1)), 3) for c in CONDITION_ORDER} if len(wide) > 1 else {},
    }
    if len(wide) < 3:
        out["friedman"] = "insufficient complete cases (need >=3 participants with all 3 conditions)"
        return out
    chi2, p = stats.friedmanchisquare(*[wide[c] for c in CONDITION_ORDER])
    out["friedman"] = {
        "chi2": round(float(chi2), 3), "df": len(CONDITION_ORDER) - 1,
        "p": round(float(p), 4), "kendalls_w": kendalls_w(float(chi2), len(wide), len(CONDITION_ORDER)),
        "significant_at_0.05": bool(p < 0.05),
    }
    # Pairwise Wilcoxon signed-rank, Bonferroni over 3 comparisons.
    pairs = {}
    comparisons = [(0, 1), (0, 2), (1, 2)]
    for i, j in comparisons:
        a, b = CONDITION_ORDER[i], CONDITION_ORDER[j]
        try:
            w, pw = stats.wilcoxon(wide[a], wide[b])
            pairs[f"{a} vs {b}"] = {
                "W": round(float(w), 3), "p": round(float(pw), 4),
                "p_bonferroni": round(min(float(pw) * len(comparisons), 1.0), 4),
                "significant_at_0.05_corrected": bool(pw * len(comparisons) < 0.05),
            }
        except Exception as e:
            pairs[f"{a} vs {b}"] = f"(unavailable: {e})"
    out["pairwise_wilcoxon"] = pairs
    return out


def calibration(scen: pd.DataFrame) -> dict:
    """Trust in the HIGH-confidence Transparency answer vs the LOW-confidence probe.

    Both are shown in the Transparency condition, so any difference reflects the
    confidence signal — i.e. whether users calibrate their trust."""
    high = (scen[(scen["condition"] == "RAG + Transparency") & (scen["is_probe"] == 0)]
            .groupby("participant_id")["composite"].mean())
    low = (scen[scen["is_probe"] == 1]
           .groupby("participant_id")["composite"].mean())
    paired = pd.concat([high.rename("high_conf"), low.rename("low_conf")], axis=1).dropna()
    out = {
        "n_pairs": int(len(paired)),
        "mean_high_confidence": round(float(paired["high_conf"].mean()), 3) if len(paired) else None,
        "mean_low_confidence_probe": round(float(paired["low_conf"].mean()), 3) if len(paired) else None,
    }
    if len(paired) >= 5:
        try:
            w, p = stats.wilcoxon(paired["high_conf"], paired["low_conf"])
            drop = float(paired["high_conf"].mean() - paired["low_conf"].mean())
            out["wilcoxon"] = {"W": round(float(w), 3), "p": round(float(p), 4),
                               "mean_trust_drop": round(drop, 3),
                               "calibrated_lower_for_low_confidence": bool(drop > 0 and p < 0.05)}
        except Exception as e:
            out["wilcoxon"] = f"(unavailable: {e})"
    else:
        out["wilcoxon"] = "insufficient pairs (need >=5)"
    return out


def per_item(scen: pd.DataFrame) -> dict:
    core = scen[scen["is_probe"] == 0]
    items = ["t_reliable", "t_accurate", "t_confident", "t_wary_r",
             "t_source", "t_understand", "t_use", "t_rely"]
    res = {}
    for it in items:
        res[it] = {c: round(float(core.loc[core["condition"] == c, it].mean()), 3)
                   for c in CONDITION_ORDER if (core["condition"] == c).any()}
    return res


def demographics(scen: pd.DataFrame, final: pd.DataFrame) -> dict:
    one = scen.drop_duplicates("participant_id")
    out = {}
    for col in ["age_band", "gender", "education", "ai_use_freq", "native_language", "language"]:
        if col in one.columns:
            out[col] = one[col].value_counts(dropna=True).to_dict()
    if "law_familiarity" in one.columns:
        fam = pd.to_numeric(one["law_familiarity"], errors="coerce").dropna()
        out["law_familiarity_mean"] = round(float(fam.mean()), 2) if len(fam) else None
    if "manip_notice" in final.columns and len(final):
        out["manip_notice"] = final["manip_notice"].value_counts(dropna=True).to_dict()
        out["manip_most_trusted"] = final["manip_most_trusted"].value_counts(dropna=True).to_dict()
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

COLOURS = {"RAG + Transparency": "#1a7a3e", "RAG (no transparency)": "#3e7a9a", "Baseline": "#a02020"}


def fig_composite(scen: pd.DataFrame, out: Path):
    core = scen[scen["is_probe"] == 0]
    wide = core.pivot_table(index="participant_id", columns="condition",
                            values="composite", aggfunc="mean").reindex(columns=CONDITION_ORDER)
    labels, means, sds = [], [], []
    for c in CONDITION_ORDER:
        col = wide[c].dropna()
        if len(col):
            labels.append(c.replace(" (", "\n("))
            means.append(col.mean())
            sds.append(col.std(ddof=1) if len(col) > 1 else 0)
    if not means:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, means, yerr=sds, capsize=6,
           color=[COLOURS[c] for c in CONDITION_ORDER][:len(means)])
    ax.set_ylim(1, 5); ax.set_ylabel("Mean composite trust (1–5)")
    ax.set_title("Trust by condition (within-subjects)")
    for i, m in enumerate(means):
        ax.text(i, min(m + 0.12, 4.9), f"{m:.2f}", ha="center", fontweight="bold")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print("wrote", out.name)


def fig_calibration(scen: pd.DataFrame, out: Path):
    high = scen[(scen["condition"] == "RAG + Transparency") & (scen["is_probe"] == 0)]["composite"]
    low = scen[scen["is_probe"] == 1]["composite"]
    if not len(high) or not len(low):
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["High-confidence\n(Transparency)", "Low-confidence\nprobe"],
           [high.mean(), low.mean()],
           yerr=[high.std(ddof=1) if len(high) > 1 else 0, low.std(ddof=1) if len(low) > 1 else 0],
           capsize=6, color=["#1a7a3e", "#b07a00"])
    ax.set_ylim(1, 5); ax.set_ylabel("Mean composite trust (1–5)")
    ax.set_title("Trust calibration: confidence signal")
    for i, m in enumerate([high.mean(), low.mean()]):
        ax.text(i, min(m + 0.12, 4.9), f"{m:.2f}", ha="center", fontweight="bold")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print("wrote", out.name)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = [a for a in sys.argv[1:]]
    keep_failed = "--keep-failed" in args
    paths = [a for a in args if not a.startswith("--")]
    csv_path = Path(paths[0]) if paths else DEFAULT_CSV

    scen, final, info = load(csv_path, keep_failed)
    print(f"Participants: {info['n_participants_raw']} raw, "
          f"{info['n_participants_analysed']} analysed ({info['attention_filter']}).")

    summary = {
        "meta": info,
        "primary_composite_trust": repeated_measures(scen, "composite"),
        "manipulation_check_source_item": repeated_measures(scen, "t_source"),
        "comprehension_item": repeated_measures(scen, "t_understand"),
        "intention_to_use_item": repeated_measures(scen, "t_use"),
        "calibration_high_vs_low_confidence": calibration(scen),
        "per_item_means_by_condition": per_item(scen),
        "demographics": demographics(scen, final),
    }

    out_dir = csv_path.parent
    (out_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                                   encoding="utf-8")
    fig_composite(scen, out_dir / "fig_trust_by_condition.png")
    fig_calibration(scen, out_dir / "fig_trust_calibration.png")

    print("\n=== Primary: composite trust across conditions ===")
    print(json.dumps(summary["primary_composite_trust"], indent=2, ensure_ascii=False))
    print("\n=== Calibration (high vs low confidence) ===")
    print(json.dumps(summary["calibration_high_vs_low_confidence"], indent=2, ensure_ascii=False))
    print(f"\nWrote analysis_summary.json + figures to {out_dir}")


if __name__ == "__main__":
    main()
