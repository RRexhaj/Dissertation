"""make_appendix_tables.py — LaTeX fragments that put the gold set and per-question results in the thesis.

Writes to eval/tables/:
    tab_gold_questions.tex   compact table of the 24 gold questions (Chapter 3)
    app_gold_full.tex        Appendix B: every item in full (question, reference answer,
                             key facts, aliases, expected citation)
    app_oos_questions.tex    Appendix B (continued): the out-of-scope robustness set
    app_per_question.tex     Appendix C: per-question results, run 1, every model
                             (fact precision alias, hallucination judge 1) — needs results_multimodel.csv

Run (from MalteseBot, inside the venv):
    .venv\\Scripts\\python.exe eval\\make_appendix_tables.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
TABLES = EVAL_DIR / "tables"
MODEL_ORDER = ["gpt-4o-2024-11-20", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-nano", "gpt-5-mini", "gpt-5.5"]
LABEL = {"gpt-4o-2024-11-20": "4o", "gpt-4o-mini": "4o-m", "gpt-4.1": "4.1", "gpt-4.1-nano": "4.1-n",
         "gpt-5-mini": "5-m", "gpt-5.5": "5.5"}
TOPIC = {"speed_limits": "Speed limits", "fines_payment": "Fine payment", "alcohol": "Alcohol",
         "demerit_points": "Penalty points", "accidents": "Accidents", "licences": "Licensing",
         "micromobility": "Micromobility"}

SPECIALS = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
            "~": r"\textasciitilde{}", "^": r"\textasciicircum{}", "\\": r"\textbackslash{}"}


def esc(s: str) -> str:
    return "".join(SPECIALS.get(ch, ch) for ch in s)


def main() -> int:
    TABLES.mkdir(exist_ok=True)
    gold = json.loads((EVAL_DIR / "gold_queries.json").read_text(encoding="utf-8"))["queries"]
    aliases = json.loads((EVAL_DIR / "fact_aliases.json").read_text(encoding="utf-8"))
    aliases.pop("_meta", None)

    # --- compact table for Chapter 3 ---------------------------------------
    lines = ["\\begin{table}[H]", "\\centering",
             "\\caption{The 24-item gold-standard question set: identifier, language, topic, difficulty and question. "
             "Reference answers, key facts and expected citations are given in full in Appendix~B.}",
             "\\label{tab:gold-questions}", "{\\footnotesize", "\\def\\arraystretch{1.15}",
             "\\begin{tabularx}{\\textwidth}{@{}l l l l >{\\raggedright\\arraybackslash}X@{}}", "\\toprule",
             "\\textbf{ID} & \\textbf{Lang.} & \\textbf{Topic} & \\textbf{Difficulty} & \\textbf{Question} \\\\", "\\midrule"]
    for q in gold:
        lines.append(f"{q['id']} & {q['lang'].upper()} & {TOPIC.get(q['topic'], q['topic'])} & {q['difficulty']} & {esc(q['question'])} \\\\")
    lines += ["\\bottomrule", "\\end{tabularx}}", "\\end{table}"]
    (TABLES / "tab_gold_questions.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # --- Appendix B: full items ----------------------------------------------
    out = ["\\begin{spacing}{1.15}", "\\footnotesize"]
    for q in gold:
        al = aliases.get(q["id"], {})
        facts = "; ".join(esc(f) + (" (" + ", ".join(esc(a) for a in al.get(f, []) if a != f) + ")" if al.get(f) else "")
                          for f in q["key_facts"])
        out += [f"\\noindent\\textbf{{{q['id']}}} \\quad {q['lang'].upper()} \\quad {TOPIC.get(q['topic'], q['topic'])} "
                f"\\quad {q['difficulty']} \\\\",
                f"\\textit{{Question:}} {esc(q['question'])} \\\\",
                f"\\textit{{Reference answer:}} {esc(q['reference_answer'])} \\\\",
                f"\\textit{{Key facts (accepted aliases):}} {facts} \\\\",
                f"\\textit{{Expected citation:}} {esc(' or '.join(q['expected_citations']))}",
                "\\par\\vspace{5pt}"]
    out += ["\\end{spacing}"]
    (TABLES / "app_gold_full.tex").write_text("\n".join(out) + "\n", encoding="utf-8")

    # --- Appendix B (continued): OOS set --------------------------------------
    oos = json.loads((EVAL_DIR / "oos_queries.json").read_text(encoding="utf-8"))["queries"]
    lines = ["\\begin{table}[H]", "\\centering",
             "\\caption{The out-of-scope robustness set: ten road-related questions on topics not covered by the corpus.}",
             "\\label{tab:oos-questions}", "{\\footnotesize", "\\def\\arraystretch{1.15}",
             "\\begin{tabularx}{\\textwidth}{@{}l l >{\\raggedright\\arraybackslash}X@{}}", "\\toprule",
             "\\textbf{ID} & \\textbf{Lang.} & \\textbf{Question} \\\\", "\\midrule"]
    for q in oos:
        lines.append(f"{q['id']} & {q['lang'].upper()} & {esc(q['question'])} \\\\")
    lines += ["\\bottomrule", "\\end{tabularx}}", "\\end{table}"]
    (TABLES / "app_oos_questions.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # --- Appendix C: per-question results (run 1) -----------------------------
    res = EVAL_DIR / "results_multimodel.csv"
    if res.exists():
        with open(res, encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f) if r["run"] == "1"]
        models = [m for m in MODEL_ORDER if any(r["model"] == m for r in rows)]
        cell = {(r["model"], r["condition"], r["id"]): r for r in rows}
        hdr = ["\\textbf{ID}"] + [f"\\textbf{{{LABEL[m]}}}" for m in models]
        lines = ["\\begin{spacing}{1.15}", "\\footnotesize",
                 "Cell entries are fact precision (alias matcher) followed by the judge-1 hallucination flag "
                 "(0 = no incorrect claim, 1 = incorrect claim) for run 1: RAG value / baseline value. "
                 "Model abbreviations: 4o = GPT-4o, 4o-m = GPT-4o mini, 4.1 = GPT-4.1, 4.1-n = GPT-4.1 nano, "
                 "5-m = GPT-5 mini, 5.5 = GPT-5.5.", "\\par\\vspace{4pt}",
                 "\\begin{longtable}{@{}l" + "c" * len(models) + "@{}}",
                 "\\caption{Per-question results for every generator model (run 1).}\\label{tab:per-question} \\\\",
                 "\\toprule", " & ".join(hdr) + " \\\\", "\\midrule", "\\endfirsthead",
                 "\\toprule", " & ".join(hdr) + " \\\\", "\\midrule", "\\endhead", "\\bottomrule", "\\endfoot"]
        for q in gold:
            cells = []
            for m in models:
                a, b = cell.get((m, "RAG", q["id"])), cell.get((m, "Baseline", q["id"]))
                if not a or not b:
                    cells.append("--")
                    continue
                cells.append(f"{float(a['kp_alias']):.2f}/{float(b['kp_alias']):.2f} "
                             f"({a['halluc_j1'] or '-'}/{b['halluc_j1'] or '-'})")
            lines.append(f"{q['id']} & " + " & ".join(cells) + " \\\\")
        lines += ["\\end{longtable}", "\\end{spacing}"]
        (TABLES / "app_per_question.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("wrote app_per_question.tex")
    print("wrote tab_gold_questions.tex, app_gold_full.tex, app_oos_questions.tex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
