"""evaluate.py — Technical accuracy evaluation for MalteseLegalBot (RQ1 / H1).

Compares the full RAG system against a no-retrieval GPT-4o baseline on a
24-question bilingual gold set. Metrics, all reported RAG vs Baseline and
broken down by language and difficulty:

    * keyword/fact precision  — fraction of gold key_facts present
    * citation validity       — expected statutory citation present (RAG)
    * hallucination rate      — GPT-4o-as-judge flags unsupported legal claims
    * ROUGE-L                  — longest-common-subsequence F1 vs reference

Run (from the MalteseBot folder, inside the venv):
    .venv\\Scripts\\python.exe eval\\evaluate.py --rebuild

Outputs:
    eval/results_per_query.csv
    eval/summary.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from corpus import load_corpus
from prompts import PROMPTS
from retrieval import HybridRetriever, build_embedder

EVAL_DIR = Path(__file__).resolve().parent
GOLD = EVAL_DIR / "gold_queries.json"
GEN_MODEL = "gpt-4o"
JUDGE_MODEL = "gpt-4o"

KNOWN_CITATIONS = [
    "Cap. 65", "Kap. 65", "S.L. 65.11", "S.L. 65.18",
    "S.L. 65.32", "S.L. 65.10", "S.L. 65.33", "S.L. 65.04",
    "Driver FAQ",
]

BASELINE_SYSTEM = (
    "You are a helpful assistant. Answer the user's question about Maltese "
    "road-traffic law clearly and concisely. Mirror the language of the question."
)


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s.lower())


def keyword_precision(answer: str, key_facts: list[str]) -> float:
    if not key_facts:
        return 0.0
    a = _norm(answer)
    hits = sum(1 for f in key_facts if _norm(f) in a)
    return hits / len(key_facts)


def cited_tags(answer: str) -> list[str]:
    return re.findall(r"\[([^\]]+)\]", answer)


def has_expected_citation(answer: str, expected: list[str]) -> bool:
    return any(e.lower() in answer.lower() for e in expected)


def has_invalid_citation(answer: str) -> bool:
    """True if the answer cites a bracketed tag that matches no known source."""
    for tag in cited_tags(answer):
        if not any(tag.startswith(k) or k in tag for k in KNOWN_CITATIONS):
            return True
    return False


def rouge_l(answer: str, reference: str) -> float:
    a = answer.lower().split()
    b = reference.lower().split()
    if not a or not b:
        return 0.0
    # LCS length via DP
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[len(a)][len(b)]
    prec = lcs / len(a)
    rec = lcs / len(b)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def rag_answer(client, retriever, question: str, lang: str) -> tuple[str, list[str]]:
    retrieved = retriever.search(question, k=5, lang=lang)
    if not retrieved:
        return ("(no relevant context found)", [])
    context = "\n\n".join(f"[{r.citation}]\n{r.text}" for r in retrieved)
    messages = PROMPTS.to_messages(context=context, question=question)
    resp = client.chat.completions.create(
        model=GEN_MODEL, messages=messages, temperature=0.1,
    )
    cites = sorted({r.citation for r in retrieved})
    return (resp.choices[0].message.content.strip(), cites)


def baseline_answer(client, question: str) -> str:
    resp = client.chat.completions.create(
        model=GEN_MODEL,
        messages=[
            {"role": "system", "content": BASELINE_SYSTEM},
            {"role": "user", "content": question},
        ],
        temperature=0.1,
    )
    return resp.choices[0].message.content.strip()


def judge_hallucination(client, question: str, answer: str,
                        reference: str, key_facts: list[str]) -> tuple[bool, str]:
    user = (
        f"Question: {question}\n\n"
        f"Reference correct answer: {reference}\n"
        f"Key facts that must be correct: {', '.join(key_facts)}\n\n"
        f"Candidate answer to evaluate:\n{answer}\n\n"
        "Does the candidate answer contain ANY statement about the law that is "
        "factually incorrect, fabricated, or contradicts the reference / key facts? "
        "Ignore extra correct detail and ignore disclaimers. "
        'Respond ONLY with JSON: {"hallucination": true|false, "reason": "<short>"}.'
    )
    resp = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": "You are a strict legal-accuracy evaluator for Maltese traffic law."},
            {"role": "user", "content": user},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    try:
        data = json.loads(resp.choices[0].message.content)
        return (bool(data.get("hallucination", False)), str(data.get("reason", "")))
    except Exception as e:
        return (False, f"(judge parse error: {e})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true", help="rebuild the ChromaDB index from the corpus")
    ap.add_argument("--limit", type=int, default=0, help="only run the first N queries (debug)")
    args = ap.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY not set (checked .env).")

    gold = json.loads(GOLD.read_text(encoding="utf-8"))["queries"]
    if args.limit:
        gold = gold[:args.limit]

    print(f"Building retriever (rerank=True) ...", flush=True)
    embedder = build_embedder(local=False)
    retriever = HybridRetriever(embedder, rerank=True)
    if args.rebuild or retriever.is_empty():
        chunks = load_corpus()
        print(f"Indexing {len(chunks)} chunks ...", flush=True)
        retriever.index(chunks, rebuild=True)
    else:
        print("Using existing index.", flush=True)

    from openai import OpenAI
    client = OpenAI()

    rows = []
    for i, q in enumerate(gold, 1):
        qid, lang = q["id"], q["lang"]
        print(f"[{i}/{len(gold)}] {qid} ({lang}, {q['difficulty']}) ...", flush=True)

        ans_rag, _ = rag_answer(client, retriever, q["question"], lang)
        ans_base = baseline_answer(client, q["question"])

        for cond, ans in [("RAG", ans_rag), ("Baseline", ans_base)]:
            kp = keyword_precision(ans, q["key_facts"])
            cite_ok = has_expected_citation(ans, q["expected_citations"])
            halluc, reason = judge_hallucination(client, q["question"], ans, q["reference_answer"], q["key_facts"])
            rl = rouge_l(ans, q["reference_answer"])
            rows.append({
                "id": qid, "lang": lang, "topic": q["topic"], "difficulty": q["difficulty"],
                "condition": cond,
                "keyword_precision": round(kp, 3),
                "has_expected_citation": int(cite_ok),
                "invalid_citation": int(has_invalid_citation(ans)),
                "hallucination": int(halluc),
                "rouge_l": round(rl, 3),
                "judge_reason": reason,
                "answer": ans.replace("\n", " "),
            })

    # Write per-query CSV
    out_csv = EVAL_DIR / "results_per_query.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Aggregate
    def agg(subset):
        n = len(subset)
        if n == 0:
            return {}
        return {
            "n": n,
            "keyword_precision": round(sum(r["keyword_precision"] for r in subset) / n, 3),
            "citation_rate": round(sum(r["has_expected_citation"] for r in subset) / n, 3),
            "hallucination_rate": round(sum(r["hallucination"] for r in subset) / n, 3),
            "rouge_l": round(sum(r["rouge_l"] for r in subset) / n, 3),
        }

    summary = {"overall": {}, "by_language": {}, "by_difficulty": {}}
    for cond in ("RAG", "Baseline"):
        cond_rows = [r for r in rows if r["condition"] == cond]
        summary["overall"][cond] = agg(cond_rows)
        for lang in ("en", "mt"):
            summary["by_language"].setdefault(lang, {})[cond] = agg([r for r in cond_rows if r["lang"] == lang])
        for diff in ("easy", "medium", "hard"):
            summary["by_difficulty"].setdefault(diff, {})[cond] = agg([r for r in cond_rows if r["difficulty"] == diff])

    (EVAL_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Pretty print
    print("\n================ OVERALL ================")
    print(f"{'metric':<22}{'RAG':>10}{'Baseline':>12}")
    for m in ("keyword_precision", "citation_rate", "hallucination_rate", "rouge_l"):
        print(f"{m:<22}{summary['overall']['RAG'][m]:>10}{summary['overall']['Baseline'][m]:>12}")
    print("\nBy language (keyword_precision / hallucination_rate):")
    for lang in ("en", "mt"):
        r = summary["by_language"][lang]
        print(f"  {lang}: RAG {r['RAG']['keyword_precision']}/{r['RAG']['hallucination_rate']}  "
              f"Baseline {r['Baseline']['keyword_precision']}/{r['Baseline']['hallucination_rate']}")
    print("\nBy difficulty (keyword_precision / hallucination_rate):")
    for diff in ("easy", "medium", "hard"):
        r = summary["by_difficulty"][diff]
        print(f"  {diff}: RAG {r['RAG']['keyword_precision']}/{r['RAG']['hallucination_rate']}  "
              f"Baseline {r['Baseline']['keyword_precision']}/{r['Baseline']['hallucination_rate']}")
    print(f"\nWrote {out_csv.name} and summary.json to {EVAL_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
