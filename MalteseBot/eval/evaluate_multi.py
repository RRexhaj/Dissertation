"""evaluate_multi.py — Multi-model technical evaluation (resubmission, Sep 2026).

Design
------
2 conditions (RAG vs no-retrieval Baseline) x M generator models x 24 gold
questions x R runs, plus an out-of-scope robustness set. Retrieval is performed
ONCE per question and the identical context is given to every model and run, so
differences between models reflect generation only.

Two independent LLM judges (default gpt-4.1 and gpt-4o) score each answer on two
separate criteria: ``incorrect_claim`` (a wrong, fabricated or contradicting
legal statement — the strict definition of hallucination used in the
resubmission) and ``omission`` (a key fact left out). The June 2026 judge
collapsed both into one "hallucination" flag; the resubmission reports them
separately.

Fact precision is reported twice: ``kp_strict`` is the pre-registered June
matcher (evaluate.keyword_precision, unchanged) and ``kp_alias`` accepts the
equivalent surface forms listed in fact_aliases.json.

Reuses evaluate.py helpers unchanged. Never overwrites the June results.

Run (from MalteseBot, inside the venv):
    .venv\\Scripts\\python.exe eval\\evaluate_multi.py --smoke
    .venv\\Scripts\\python.exe eval\\evaluate_multi.py --runs 3 --workers 4
    .venv\\Scripts\\python.exe eval\\evaluate_multi.py --oos

Outputs (eval/):
    results_multimodel.csv   one row per (model, condition, run, question)
    summary_multimodel.json  aggregates per model x condition, by language/difficulty/run
    results_oos.csv, summary_oos.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from compliance import detect_language
from corpus import load_corpus
from prompts import PROMPTS
from retrieval import HybridRetriever, build_embedder

from evaluate import (BASELINE_SYSTEM, has_expected_citation, has_invalid_citation,
                      keyword_precision, rouge_l, _norm)
from llm import chat

EVAL_DIR = Path(__file__).resolve().parent
GOLD = EVAL_DIR / "gold_queries.json"
ALIASES = EVAL_DIR / "fact_aliases.json"
OOS = EVAL_DIR / "oos_queries.json"

DEFAULT_MODELS = ["gpt-4o-2024-11-20", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-nano",
                  "gpt-5-mini", "gpt-5.5"]
DEFAULT_JUDGES = ["gpt-4.1", "gpt-4o"]
CONDITIONS = ("RAG", "Baseline")

FIELDS = ["model", "snapshot", "condition", "run", "id", "lang", "topic", "difficulty",
          "kp_strict", "kp_alias", "cite_ok", "invalid_cite", "mirror",
          "halluc_j1", "omission_j1", "reason_j1", "halluc_j2", "omission_j2", "reason_j2",
          "rouge_l", "latency_s", "tokens_in", "tokens_out", "finish", "answer"]

OOS_FIELDS = ["model", "snapshot", "condition", "run", "id", "lang", "refusal", "mirror",
              "asserts_j1", "reason_j1", "asserts_j2", "reason_j2",
              "latency_s", "tokens_in", "tokens_out", "answer"]

_csv_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Scoring additions
# ---------------------------------------------------------------------------

def keyword_precision_alias(answer: str, key_facts: list[str],
                            aliases: dict[str, list[str]] | None) -> float:
    """Alias-tolerant fact precision: a fact counts if ANY of its surface forms
    (the strict string itself or one of its aliases) appears in the answer."""
    if not key_facts:
        return 0.0
    a = _norm(answer)
    hits = 0
    for fact in key_facts:
        forms = [fact] + list((aliases or {}).get(fact, []))
        if any(_norm(f) in a for f in forms):
            hits += 1
    return hits / len(key_facts)


REFUSAL_PATTERNS = [
    # English
    r"do(?:es)? not have (?:that |this |any )?information", r"don't have (?:that |this |any )?information",
    r"no information (?:on|about|regarding)", r"not (?:covered|addressed|included|contained|available) in",
    r"does not (?:cover|address|include|contain|specify|provide)", r"doesn't (?:cover|address|include|contain|specify)",
    r"cannot find", r"can't find", r"unable to (?:find|provide|answer)", r"not able to (?:find|provide|answer)",
    r"outside (?:the )?scope", r"beyond (?:the )?scope", r"not in cap\.? ?65",
    r"not (?:in|within) the (?:provided|supplied|available)",
    r"no (?:specific )?(?:provision|rule|information) (?:in|on|about)", r"i (?:do not|don't) know",
    # Maltese
    r"ma nsibx", r"m'?għandix informazzjoni", r"ma għandix informazzjoni", r"ma jkoprix", r"mhux kopert",
    r"ma fihx", r"ma nistax (?:insib|nagħti|inwieġeb)", r"m'?hemmx informazzjoni", r"ma jinsabx", r"ma ssemmix",
    r"mhux inkluż", r"ma hemmx dispożizzjoni", r"barra mill-ambitu",
]
_REFUSAL_RE = re.compile("|".join(REFUSAL_PATTERNS), re.IGNORECASE)


def is_refusal(answer: str) -> bool:
    return bool(_REFUSAL_RE.search(answer))


def mirrors_language(answer: str, lang: str) -> bool:
    return detect_language(answer) == lang


# ---------------------------------------------------------------------------
# Judges
# ---------------------------------------------------------------------------

JUDGE_SYSTEM = "You are a strict legal-accuracy evaluator for Maltese road-traffic law."

JUDGE_TEMPLATE = (
    "Question: {question}\n\n"
    "Reference correct answer: {reference}\n"
    "Key facts a correct answer must contain: {facts}\n\n"
    "Candidate answer to evaluate:\n{answer}\n\n"
    "Evaluate the candidate on TWO SEPARATE criteria.\n"
    "1. incorrect_claim: true if the candidate makes ANY statement about the law that is "
    "factually incorrect, fabricated, or contradicts the reference or key facts (a wrong "
    "number, a wrong rule, an invented provision, penalty or procedure). Correct extra detail "
    "does not count. Disclaimers do not count. Leaving something out is NOT an incorrect claim.\n"
    "2. omission: true if the candidate fails to state one or more of the key facts.\n"
    'Respond ONLY with JSON: {{"incorrect_claim": true|false, "omission": true|false, "reason": "<short>"}}'
)

OOS_JUDGE_TEMPLATE = (
    "Question: {question}\n\n"
    "Context: this question is about a topic that is NOT covered by the assistant's sources "
    "(Cap. 65 extracts and selected subsidiary legislation). A well-behaved answer says the "
    "information is not available in its sources and refers the user elsewhere.\n\n"
    "Candidate answer to evaluate:\n{answer}\n\n"
    "asserts_specifics: true if the candidate asserts concrete legal figures, rules, fees, "
    "limits, deadlines or procedures AS FACT (for example a specific fine amount or weight "
    "limit) rather than acknowledging that its sources do not cover the question. General "
    "advice to consult an authority does not count as asserting specifics.\n"
    'Respond ONLY with JSON: {{"asserts_specifics": true|false, "reason": "<short>"}}'
)


def judge_answer(client, judge_model: str, q: dict, answer: str) -> tuple[bool | None, bool | None, str]:
    user = JUDGE_TEMPLATE.format(question=q["question"], reference=q["reference_answer"],
                                 facts=", ".join(q["key_facts"]), answer=answer)
    res = chat(client, judge_model,
               [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}],
               temperature=0.0, json_mode=True)
    try:
        data = json.loads(res.text)
        return (bool(data.get("incorrect_claim", False)), bool(data.get("omission", False)),
                str(data.get("reason", "")))
    except Exception as e:  # noqa: BLE001
        return (None, None, f"(judge parse error: {e})")


def judge_oos(client, judge_model: str, q: dict, answer: str) -> tuple[bool | None, str]:
    user = OOS_JUDGE_TEMPLATE.format(question=q["question"], answer=answer)
    res = chat(client, judge_model,
               [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}],
               temperature=0.0, json_mode=True)
    try:
        data = json.loads(res.text)
        return (bool(data.get("asserts_specifics", False)), str(data.get("reason", "")))
    except Exception as e:  # noqa: BLE001
        return (None, f"(judge parse error: {e})")


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate(client, model: str, condition: str, question: str, context: str | None):
    if condition == "RAG":
        if not context:
            return None
        messages = PROMPTS.to_messages(context=context, question=question)
    else:
        messages = [{"role": "system", "content": BASELINE_SYSTEM},
                    {"role": "user", "content": question}]
    return chat(client, model, messages, temperature=0.1)


def build_contexts(gold: list[dict], *, k: int, rebuild: bool) -> dict[str, tuple[str, list[str]]]:
    """Retrieve once per question; the same context is reused for every model/run."""
    print("Building retriever (hybrid + rerank) ...", flush=True)
    embedder = build_embedder(local=False)
    retriever = HybridRetriever(embedder, rerank=True)
    if rebuild or retriever.is_empty():
        chunks = load_corpus()
        print(f"Indexing {len(chunks)} chunks ...", flush=True)
        retriever.index(chunks, rebuild=True)
    ctx: dict[str, tuple[str, list[str]]] = {}
    for q in gold:
        retrieved = retriever.search(q["question"], k=k, lang=q["lang"])
        if not retrieved:
            ctx[q["id"]] = ("", [])
            continue
        context = "\n\n".join(f"[{r.citation}]\n{r.text}" for r in retrieved)
        ctx[q["id"]] = (context, sorted({r.citation for r in retrieved}))
    return ctx


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------

def _append_row(path: Path, fields: list[str], row: dict) -> None:
    with _csv_lock:
        new = not path.exists() or path.stat().st_size == 0
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            if new:
                w.writeheader()
            w.writerow(row)


def _existing_keys(path: Path) -> set[tuple]:
    if not path.exists():
        return set()
    with open(path, encoding="utf-8") as f:
        return {(r["model"], r["condition"], r["run"], r["id"]) for r in csv.DictReader(f)}


def _b(v) -> str:
    return "" if v is None else str(int(v))


def run_one(client, judges: list[str], aliases: dict, ctx: dict, model: str, condition: str,
            run: int, q: dict, out: Path) -> dict:
    context, _ = ctx[q["id"]]
    res = generate(client, model, condition, q["question"], context)
    ans = res.text if res else "(no relevant context found)"
    row = {
        "model": model, "snapshot": res.snapshot if res else "", "condition": condition, "run": run,
        "id": q["id"], "lang": q["lang"], "topic": q["topic"], "difficulty": q["difficulty"],
        "kp_strict": round(keyword_precision(ans, q["key_facts"]), 3),
        "kp_alias": round(keyword_precision_alias(ans, q["key_facts"], aliases.get(q["id"])), 3),
        "cite_ok": int(has_expected_citation(ans, q["expected_citations"])),
        "invalid_cite": int(has_invalid_citation(ans)),
        "mirror": int(mirrors_language(ans, q["lang"])),
        "rouge_l": round(rouge_l(ans, q["reference_answer"]), 3),
        "latency_s": round(res.latency_s, 2) if res else 0.0,
        "tokens_in": res.tokens_in if res else 0, "tokens_out": res.tokens_out if res else 0,
        "finish": res.finish_reason if res else "",
        "answer": ans.replace("\n", " "),
    }
    for i, jm in enumerate(judges[:2], start=1):
        h, o, r = judge_answer(client, jm, q, ans)
        row[f"halluc_j{i}"], row[f"omission_j{i}"], row[f"reason_j{i}"] = _b(h), _b(o), r.replace("\n", " ")
    for i in range(len(judges), 2):
        row[f"halluc_j{i+1}"] = row[f"omission_j{i+1}"] = row[f"reason_j{i+1}"] = ""
    _append_row(out, FIELDS, row)
    return row


def run_one_oos(client, judges: list[str], ctx: dict, model: str, condition: str,
                run: int, q: dict, out: Path) -> dict:
    context, _ = ctx[q["id"]]
    res = generate(client, model, condition, q["question"], context)
    ans = res.text if res else "(no relevant context found)"
    row = {
        "model": model, "snapshot": res.snapshot if res else "", "condition": condition, "run": run,
        "id": q["id"], "lang": q["lang"],
        "refusal": int(is_refusal(ans)), "mirror": int(mirrors_language(ans, q["lang"])),
        "latency_s": round(res.latency_s, 2) if res else 0.0,
        "tokens_in": res.tokens_in if res else 0, "tokens_out": res.tokens_out if res else 0,
        "answer": ans.replace("\n", " "),
    }
    for i, jm in enumerate(judges[:2], start=1):
        a, r = judge_oos(client, jm, q, ans)
        row[f"asserts_j{i}"], row[f"reason_j{i}"] = _b(a), r.replace("\n", " ")
    for i in range(len(judges), 2):
        row[f"asserts_j{i+1}"] = row[f"reason_j{i+1}"] = ""
    _append_row(out, OOS_FIELDS, row)
    return row


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _mean(rows, key) -> float | None:
    vals = [float(r[key]) for r in rows if r.get(key) not in ("", None)]
    return round(sum(vals) / len(vals), 3) if vals else None


def agg(rows: list[dict]) -> dict:
    if not rows:
        return {}
    return {
        "n": len(rows),
        "kp_strict": _mean(rows, "kp_strict"), "kp_alias": _mean(rows, "kp_alias"),
        "citation_rate": _mean(rows, "cite_ok"), "invalid_citation_rate": _mean(rows, "invalid_cite"),
        "hallucination_j1": _mean(rows, "halluc_j1"), "omission_j1": _mean(rows, "omission_j1"),
        "hallucination_j2": _mean(rows, "halluc_j2"), "omission_j2": _mean(rows, "omission_j2"),
        "mirror_rate": _mean(rows, "mirror"), "rouge_l": _mean(rows, "rouge_l"),
        "latency_s": _mean(rows, "latency_s"),
        "tokens_in": _mean(rows, "tokens_in"), "tokens_out": _mean(rows, "tokens_out"),
    }


def _model_order(m: str) -> int:
    return DEFAULT_MODELS.index(m) if m in DEFAULT_MODELS else 99


def summarise(csv_path: Path, out_path: Path) -> dict:
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    models = sorted({r["model"] for r in rows}, key=_model_order)
    summary: dict = {"models": models, "overall": {}, "by_language": {}, "by_difficulty": {}, "by_run": {}}
    for m in models:
        mrows = [r for r in rows if r["model"] == m]
        summary["overall"][m] = {c: agg([r for r in mrows if r["condition"] == c]) for c in CONDITIONS}
        summary["by_language"][m] = {
            lang: {c: agg([r for r in mrows if r["condition"] == c and r["lang"] == lang]) for c in CONDITIONS}
            for lang in ("en", "mt")}
        summary["by_difficulty"][m] = {
            d: {c: agg([r for r in mrows if r["condition"] == c and r["difficulty"] == d]) for c in CONDITIONS}
            for d in ("easy", "medium", "hard")}
        runs = sorted({r["run"] for r in mrows}, key=int)
        summary["by_run"][m] = {
            run: {c: agg([r for r in mrows if r["condition"] == c and r["run"] == run]) for c in CONDITIONS}
            for run in runs}
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def print_summary(summary: dict) -> None:
    print("\n=== model x condition (all runs pooled) ===")
    print(f"{'model':<20}{'cond':<10}{'n':>4}{'kp_str':>8}{'kp_ali':>8}{'cite':>7}"
          f"{'hal_j1':>8}{'omi_j1':>8}{'hal_j2':>8}{'mirror':>8}{'lat_s':>7}")
    for m in summary["models"]:
        for c in CONDITIONS:
            a = summary["overall"][m].get(c) or {}
            if not a:
                continue

            def f(k, a=a):
                return "-" if a.get(k) is None else f"{a[k]:.3f}"
            print(f"{m:<20}{c:<10}{a['n']:>4}{f('kp_strict'):>8}{f('kp_alias'):>8}{f('citation_rate'):>7}"
                  f"{f('hallucination_j1'):>8}{f('omission_j1'):>8}{f('hallucination_j2'):>8}"
                  f"{f('mirror_rate'):>8}{f('latency_s'):>7}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--judges", default=",".join(DEFAULT_JUDGES))
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0, help="only the first N gold questions (debug)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=str(EVAL_DIR / "results_multimodel.csv"))
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--no-resume", action="store_true", help="ignore rows already in --out")
    ap.add_argument("--smoke", action="store_true", help="1 question x 1 run x all models -> eval/smoke.csv")
    ap.add_argument("--oos", action="store_true", help="run the out-of-scope set instead of the gold set")
    ap.add_argument("--summarise-only", action="store_true")
    args = ap.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY not set (checked .env).")

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    judges = [j.strip() for j in args.judges.split(",") if j.strip()]
    out = Path(args.out)
    if args.smoke:
        out, args.limit, args.runs = EVAL_DIR / "smoke.csv", 1, 1
        if out.exists():
            out.unlink()
    if args.oos and args.out == str(EVAL_DIR / "results_multimodel.csv"):
        out = EVAL_DIR / "results_oos.csv"

    if args.summarise_only:
        summ_path = out.with_name(out.stem.replace("results", "summary") + ".json")
        print_summary(summarise(out, summ_path))
        return 0

    if args.oos:
        gold = json.loads(OOS.read_text(encoding="utf-8"))["queries"]
        runs = 1
    else:
        gold = json.loads(GOLD.read_text(encoding="utf-8"))["queries"]
        runs = args.runs
    if args.limit:
        gold = gold[:args.limit]
    aliases = json.loads(ALIASES.read_text(encoding="utf-8"))
    aliases.pop("_meta", None)

    ctx = build_contexts(gold, k=args.k, rebuild=args.rebuild)

    from openai import OpenAI
    client = OpenAI()

    done = set() if args.no_resume else _existing_keys(out)
    tasks = [(m, c, r, q) for m in models for c in CONDITIONS for r in range(1, runs + 1) for q in gold
             if (m, c, str(r), q["id"]) not in done]
    print(f"{len(tasks)} generations to run ({len(done)} already present) with judges {judges}", flush=True)

    t0 = time.time()
    n_done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {}
        for m, c, r, q in tasks:
            if args.oos:
                fut = ex.submit(run_one_oos, client, judges, ctx, m, c, r, q, out)
            else:
                fut = ex.submit(run_one, client, judges, aliases, ctx, m, c, r, q, out)
            futs[fut] = (m, c, r, q["id"])
        for fut in as_completed(futs):
            m, c, r, qid = futs[fut]
            n_done += 1
            try:
                row = fut.result()
                tag = (f"ref={row['refusal']}" if args.oos
                       else f"kp={row['kp_strict']}/{row['kp_alias']} hal={row['halluc_j1']}/{row['halluc_j2']}")
                print(f"[{n_done}/{len(tasks)}] {m} {c} run{r} {qid}: {tag} ({row['latency_s']}s)", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[{n_done}/{len(tasks)}] {m} {c} run{r} {qid}: FAILED {type(e).__name__}: "
                      f"{str(e)[:120]}", flush=True)
    print(f"finished in {(time.time() - t0) / 60:.1f} min", flush=True)

    if args.oos:
        with open(out, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        summary = {}
        for m in models:
            summary[m] = {}
            for c in CONDITIONS:
                sub = [r for r in rows if r["model"] == m and r["condition"] == c]
                summary[m][c] = {"n": len(sub), "refusal_rate": _mean(sub, "refusal"),
                                 "asserts_j1": _mean(sub, "asserts_j1"), "asserts_j2": _mean(sub, "asserts_j2"),
                                 "mirror_rate": _mean(sub, "mirror")}
        (EVAL_DIR / "summary_oos.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
    else:
        summ_path = EVAL_DIR / ("smoke_summary.json" if args.smoke else "summary_multimodel.json")
        print_summary(summarise(out, summ_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
