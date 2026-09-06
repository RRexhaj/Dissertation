"""check_numbers.py -- flag numbers quoted in the thesis prose that do not
appear in any generated statistics file or table fragment.

Usage (from MalteseBot/):  python eval/check_numbers.py [--all]

Every decimal, percentage and "points" figure in the Abstract, Chapter 4 and
Chapter 5 is looked up (with half-unit rounding tolerance, as a proportion or
as a percentage) in stats_summary.json, summary_*.json, judge_agreement.json,
the user-study analysis_summary_v2.json and every Tables/*.tex fragment.
Integers are checked only when larger than 30 (counts below that are usually
question or participant numbers). Unmatched numbers are printed with context
for hand review; the script does not edit anything.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LATEX = Path(r"C:\Users\rexha\Downloads\Dissertation_LaTeX")
REPO = Path(r"C:\Users\rexha\Downloads\Dissertation_repo")
PROSE = [
    LATEX / "Front" / "05_Abstract.tex",
    LATEX / "Sections" / "04_AnalysisResultsDiscussion.tex",
    LATEX / "Sections" / "05_ConclusionsRecommendations.tex",
]
if "--all" in sys.argv:
    PROSE += [LATEX / "Sections" / "01_Introduction.tex", LATEX / "Sections" / "03_Methodology.tex"]
JSONS = [
    HERE / "stats_summary.json",
    HERE / "summary_multimodel.json",
    HERE / "summary_oos.json",
    HERE / "summary_ablation.json",
    HERE / "summary.json",
    HERE / "judge_agreement.json",
    REPO / "analysis_summary_v2.json",
    REPO / "analysis_summary_PRIMARY.json",
]

NUM = r"(?<![\w.])[-+]?\d+(?:\.\d+)?(?![\w])"
BS = "\\"  # a single backslash, kept out of the regex literals below


def walk(o, out):
    if isinstance(o, bool):
        return
    if isinstance(o, (int, float)):
        out.append(float(o))
    elif isinstance(o, dict):
        for v in o.values():
            walk(v, out)
    elif isinstance(o, list):
        for v in o:
            walk(v, out)


refs: list[float] = []
for j in JSONS:
    if j.exists():
        walk(json.loads(j.read_text(encoding="utf-8")), refs)
if "--tables" in sys.argv:  # optional: also accept the rounded values printed in the table fragments
    for t in (LATEX / "Tables").glob("*.tex"):
        for m in re.finditer(NUM, t.read_text(encoding="utf-8")):
            refs.append(float(m.group(0)))
refs = sorted(set(abs(r) for r in refs))


def matches(x: float, dec: int, pct: bool) -> bool:
    """x matches a reference value r when x is r (or |r|) rounded to the same
    number of decimals; a figure written with % or pp matches 100 * r only,
    a figure in "points" may be percentage points or Likert-scale points and
    so matches either form, and a bare figure above 1 may match either form.

    Limitation: with ~1,500 reference values every two-decimal proportion in
    [0, 1] matches something, so the check is informative for percentages,
    point differences, Likert means and three-decimal values, but not for
    bare proportions; those are read against the generated tables by hand.
    It also cannot detect a correct number attached to the wrong claim."""
    for r in refs:
        if pct or x > 1:
            if round(100 * r, dec) == x:
                return True
        if (not pct or pct == "points") and round(r, dec) == x:
            return True
    return False


# Patterns removed before numbers are extracted (LaTeX commands, statute
# references, years, physical units).  Backslashes are built with BS so that
# the file survives shells that collapse doubled backslashes.
STRIP = [
    r"%.*",
    re.escape(BS) + r"(?:input|label|ref|cite|includegraphics|caption|url)\{[^}]*\}",
    re.escape(BS) + r"[a-zA-Z]+\*?",
    r"Cap\.~?65",
    r"S\.L\.[~\s]*65\.\d+",
    r"\b(?:article|articles|regulation|regulations|section|chapter|figure|table)s?~?\s*\d+(?:\.\d+)?",
    r"\b(?:19|20)\d\d\b",
    r"\d+\s*km/h",
    r"\d+\s*mg\b",
    r"\d+\s*g/l",
    r"\d+\s*ml\b",
]
FIND = r"(?<![\w.])(\d+(?:\.\d+)?)(\s*(?:%|points?|pp))?"

total_bad = 0
for f in PROSE:
    lines = f.read_text(encoding="utf-8").split("\n")
    bad = []
    for ln, raw in enumerate(lines, 1):
        s = raw
        for pat in STRIP:
            s = re.sub(pat, " ", s)
        for m in re.finditer(FIND, s):
            num = m.group(1)
            dec = len(num.split(".")[1]) if "." in num else 0
            x = float(num)
            if dec == 0 and x <= 30 and not m.group(2):
                continue
            suffix = (m.group(2) or "").strip()
            pct = "points" if suffix.startswith("point") else bool(suffix)
            if not matches(x, dec, pct):
                a, b = max(0, m.start() - 45), min(len(s), m.end() + 35)
                bad.append((ln, num, s[a:b]))
    print(f"\n### {f.name}: {len(bad)} unmatched")
    for ln, num, ctx in bad:
        print(f"  L{ln:<4} {num:>8}  ...{ctx.strip()}...")
    total_bad += len(bad)
print(f"\nTOTAL unmatched: {total_bad}  (reference values: {len(refs)})")
