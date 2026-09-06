"""judge_agreement.py — agreement between the human rater and the two LLM judges.

Reads the completed rating_sheet.xlsx (columns incorrect_claim / omission filled
by the blind rater) and rating_key.json, then computes for hallucination and
omission separately:
    human vs judge 1 (gpt-4.1), human vs judge 2 (gpt-4o), judge 1 vs judge 2
    Cohen's kappa, percentage agreement, 2x2 confusion counts, and (treating the
    human as ground truth) each judge's sensitivity and specificity.

Run (from MalteseBot, inside the venv):
    .venv\\Scripts\\python.exe eval\\judge_agreement.py
Outputs: eval/judge_agreement.json, eval/tables/tab_judge_agreement.tex
"""
from __future__ import annotations

import json
from pathlib import Path

from openpyxl import load_workbook

EVAL_DIR = Path(__file__).resolve().parent
TABLES = EVAL_DIR / "tables"
SHEET = EVAL_DIR / "rating_sheet.xlsx"
KEY = EVAL_DIR / "rating_key.json"


def cohen_kappa(a: list[int], b: list[int]) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pa1 = sum(a) / n
    pb1 = sum(b) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def confusion(truth: list[int], pred: list[int]) -> dict:
    tp = sum(1 for t, p in zip(truth, pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(truth, pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(truth, pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(truth, pred) if t == 1 and p == 0)
    sens = tp / (tp + fn) if tp + fn else None
    spec = tn / (tn + fp) if tn + fp else None
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "sensitivity": None if sens is None else round(sens, 3),
            "specificity": None if spec is None else round(spec, 3)}


def landis_koch(k: float) -> str:
    if k != k:
        return "n/a"
    for lim, lab in [(0.0, "poor"), (0.2, "slight"), (0.4, "fair"), (0.6, "moderate"), (0.8, "substantial")]:
        if k <= lim:
            return lab
    return "almost perfect"


def main() -> int:
    key = json.loads(KEY.read_text(encoding="utf-8"))
    wb = load_workbook(SHEET, data_only=True)
    ws = wb["Ratings"]
    header = [c.value for c in ws[1]]
    col = {h: i for i, h in enumerate(header)}
    human: dict[str, dict] = {}
    for row in ws.iter_rows(min_row=2, values_only=True):
        rid = row[col["rating_id"]]
        if rid is None:
            continue
        ic, om = row[col["incorrect_claim"]], row[col["omission"]]
        if ic is None or om is None:
            continue
        human[rid] = {"halluc": int(ic), "omission": int(om)}
    rated = [rid for rid in key if rid in human]
    print(f"{len(rated)} of {len(key)} answers rated")

    out: dict = {"n_rated": len(rated), "n_total": len(key), "pairs": {}}
    for metric, jkey in (("halluc", "halluc_j"), ("omission", "omission_j")):
        h = [human[r][metric] for r in rated]
        j1 = [int(key[r][f"{jkey}1"]) for r in rated if key[r][f"{jkey}1"] != ""]
        j2 = [int(key[r][f"{jkey}2"]) for r in rated if key[r][f"{jkey}2"] != ""]
        ok = len(j1) == len(h) == len(j2)
        pairs = {}
        if ok:
            for name, a, b, truth_first in (("human_vs_judge1", h, j1, True), ("human_vs_judge2", h, j2, True),
                                            ("judge1_vs_judge2", j1, j2, False)):
                k = cohen_kappa(a, b)
                pairs[name] = {"kappa": round(k, 3), "agreement": round(sum(1 for x, y in zip(a, b) if x == y) / len(a), 3),
                               "interpretation": landis_koch(k), "n": len(a)}
                if truth_first:
                    pairs[name].update(confusion(a, b))
        out["pairs"][metric] = pairs
        out[f"{metric}_rate_human"] = round(sum(h) / len(h), 3) if h else None
        out[f"{metric}_rate_judge1"] = round(sum(j1) / len(j1), 3) if j1 else None
        out[f"{metric}_rate_judge2"] = round(sum(j2) / len(j2), 3) if j2 else None
        # Breakdown of the human verdicts by condition (baseline vs RAG) for the summary tables
        for cond in ("RAG", "Baseline"):
            sub = [human[r][metric] for r in rated if key[r]["condition"] == cond]
            out[f"{metric}_rate_human_{cond}"] = round(sum(sub) / len(sub), 3) if sub else None

    (EVAL_DIR / "judge_agreement.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    TABLES.mkdir(exist_ok=True)
    lines = [
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{Agreement between the blind human rater and the two LLM judges on the rated sample "
        f"(n = {len(rated)} answers). Sensitivity and specificity treat the human rating as ground truth.}}",
        "\\label{tab:judge-agreement}",
        "\\begin{tabular}{llcccc}",
        "\\toprule",
        "Criterion & Pair & Cohen's $\\kappa$ & Agreement & Sensitivity & Specificity \\\\",
        "\\midrule",
    ]
    names = {"human_vs_judge1": "Human vs judge 1 (GPT-4.1)", "human_vs_judge2": "Human vs judge 2 (GPT-4o)",
             "judge1_vs_judge2": "Judge 1 vs judge 2"}
    for metric, label in (("halluc", "Incorrect claim"), ("omission", "Omission")):
        for i, (name, v) in enumerate(out["pairs"].get(metric, {}).items()):
            se = "--" if v.get("sensitivity") is None else f"{v['sensitivity']:.2f}"
            sp = "--" if v.get("specificity") is None else f"{v['specificity']:.2f}"
            lines.append(f"{label if i == 0 else ''} & {names[name]} & {v['kappa']:.2f} ({v['interpretation']}) & "
                         f"{v['agreement']:.2f} & {se} & {sp} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (TABLES / "tab_judge_agreement.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
