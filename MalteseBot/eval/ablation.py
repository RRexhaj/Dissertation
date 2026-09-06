"""ablation.py — Retrieval ablation for RQ1c (resubmission, Sep 2026).

Retrieval configurations (the knobs already exposed by HybridRetriever):
    bm25          alpha=0.0, rerank=False
    dense         alpha=1.0, rerank=False
    hybrid        alpha=0.6, rerank=False
    hybrid+rerank alpha=0.6, rerank=True   (the deployed configuration)
crossed with two embedders (text-embedding-3-large, the deployed one, and
text-embedding-3-small). Each embedder gets its own ChromaDB directory under
eval/chroma_ablation/ so the deployed index is never touched.

Retrieval metrics over the 24 gold queries: Hit@1, Hit@5 and reciprocal rank of
the first retrieved chunk whose citation matches an expected citation.

Downstream (optional, --downstream): for the four configurations with the large
embedder, generate a GPT-4o answer from each configuration's context (one run)
and score it with the same metrics and judge as evaluate_multi.py.

Run (from MalteseBot, inside the venv):
    .venv\\Scripts\\python.exe eval\\ablation.py
    .venv\\Scripts\\python.exe eval\\ablation.py --downstream

Outputs (eval/): results_ablation_retrieval.csv, results_ablation_downstream.csv,
                 summary_ablation.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
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

from corpus import load_corpus
from prompts import PROMPTS
from retrieval import HybridRetriever, _OpenAIEmbedder

from evaluate import has_expected_citation, has_invalid_citation, keyword_precision, rouge_l
from evaluate_multi import judge_answer, keyword_precision_alias, mirrors_language
from llm import chat

EVAL_DIR = Path(__file__).resolve().parent
GOLD = EVAL_DIR / "gold_queries.json"
ALIASES = EVAL_DIR / "fact_aliases.json"

CONFIGS = [  # name, alpha, rerank
    ("bm25", 0.0, False),
    ("dense", 1.0, False),
    ("hybrid", 0.6, False),
    ("hybrid+rerank", 0.6, True),
]
EMBEDDERS = ["text-embedding-3-large", "text-embedding-3-small"]
DOWNSTREAM_MODEL = "gpt-4o-2024-11-20"
DOWNSTREAM_JUDGE = "gpt-4.1"


def citation_matches(citation: str, expected: list[str]) -> bool:
    c = citation.lower()
    return any(c.startswith(e.lower()) for e in expected)


def first_hit_rank(retrieved, expected: list[str]) -> int | None:
    for rank, r in enumerate(retrieved, start=1):
        if citation_matches(r.citation, expected):
            return rank
    return None


def build(embedder_name: str, chunks) -> HybridRetriever:
    emb = _OpenAIEmbedder(model=embedder_name)
    chroma_dir = EVAL_DIR / "chroma_ablation" / embedder_name.replace("/", "_")
    r = HybridRetriever(emb, chroma_dir=chroma_dir, alpha=0.6, rerank=True)
    if r.is_empty():
        print(f"Indexing {len(chunks)} chunks with {embedder_name} ...", flush=True)
        r.index(chunks, rebuild=True)
    else:
        r._load_bm25_from_collection()
    return r


def configure(r: HybridRetriever, alpha: float, rerank: bool, reranker) -> None:
    r.alpha = alpha
    r._reranker = reranker if rerank else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--downstream", action="store_true")
    ap.add_argument("--embedders", default=",".join(EMBEDDERS))
    args = ap.parse_args()
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY not set (checked .env).")

    gold = json.loads(GOLD.read_text(encoding="utf-8"))["queries"]
    aliases = json.loads(ALIASES.read_text(encoding="utf-8"))
    aliases.pop("_meta", None)
    chunks = load_corpus()

    ret_rows: list[dict] = []
    contexts: dict[tuple[str, str], dict[str, str]] = {}   # (embedder, config) -> qid -> context
    for emb_name in [e.strip() for e in args.embedders.split(",") if e.strip()]:
        r = build(emb_name, chunks)
        reranker = r._reranker
        for name, alpha, rerank in CONFIGS:
            configure(r, alpha, rerank, reranker)
            contexts[(emb_name, name)] = {}
            for q in gold:
                t0 = time.time()
                retrieved = r.search(q["question"], k=args.k, lang=q["lang"])
                lat = time.time() - t0
                rank = first_hit_rank(retrieved, q["expected_citations"])
                ret_rows.append({
                    "embedder": emb_name, "config": name, "id": q["id"], "lang": q["lang"],
                    "topic": q["topic"], "difficulty": q["difficulty"],
                    "hit1": int(rank == 1), "hit5": int(rank is not None and rank <= args.k),
                    "rr": round(1.0 / rank, 3) if rank else 0.0,
                    "latency_s": round(lat, 3),
                    "top_citations": " | ".join(x.citation for x in retrieved),
                })
                contexts[(emb_name, name)][q["id"]] = "\n\n".join(f"[{x.citation}]\n{x.text}" for x in retrieved)
            hits = [row for row in ret_rows if row["embedder"] == emb_name and row["config"] == name]
            print(f"{emb_name:<26}{name:<15} hit@1={sum(x['hit1'] for x in hits)/len(hits):.3f} "
                  f"hit@{args.k}={sum(x['hit5'] for x in hits)/len(hits):.3f} "
                  f"MRR={sum(x['rr'] for x in hits)/len(hits):.3f}", flush=True)

    with open(EVAL_DIR / "results_ablation_retrieval.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(ret_rows[0].keys()))
        w.writeheader()
        w.writerows(ret_rows)

    summary: dict = {"retrieval": {}, "downstream": {}}
    for emb_name in {row["embedder"] for row in ret_rows}:
        summary["retrieval"][emb_name] = {}
        for name, _, _ in CONFIGS:
            sub = [row for row in ret_rows if row["embedder"] == emb_name and row["config"] == name]
            if not sub:
                continue
            summary["retrieval"][emb_name][name] = {
                "n": len(sub),
                "hit1": round(sum(x["hit1"] for x in sub) / len(sub), 3),
                "hit5": round(sum(x["hit5"] for x in sub) / len(sub), 3),
                "mrr": round(sum(x["rr"] for x in sub) / len(sub), 3),
                "latency_s": round(sum(x["latency_s"] for x in sub) / len(sub), 3),
                "by_lang": {lang: {
                    "hit5": round(sum(x["hit5"] for x in sub if x["lang"] == lang) / max(1, sum(1 for x in sub if x["lang"] == lang)), 3),
                    "mrr": round(sum(x["rr"] for x in sub if x["lang"] == lang) / max(1, sum(1 for x in sub if x["lang"] == lang)), 3),
                } for lang in ("en", "mt")},
            }

    if args.downstream:
        from openai import OpenAI
        client = OpenAI()
        emb_name = EMBEDDERS[0]
        ds_rows: list[dict] = []
        for name, _, _ in CONFIGS:
            for q in gold:
                context = contexts[(emb_name, name)][q["id"]]
                messages = PROMPTS.to_messages(context=context, question=q["question"])
                res = chat(client, DOWNSTREAM_MODEL, messages, temperature=0.1)
                ans = res.text
                h, o, reason = judge_answer(client, DOWNSTREAM_JUDGE, q, ans)
                ds_rows.append({
                    "config": name, "model": DOWNSTREAM_MODEL, "id": q["id"], "lang": q["lang"],
                    "difficulty": q["difficulty"],
                    "kp_strict": round(keyword_precision(ans, q["key_facts"]), 3),
                    "kp_alias": round(keyword_precision_alias(ans, q["key_facts"], aliases.get(q["id"])), 3),
                    "cite_ok": int(has_expected_citation(ans, q["expected_citations"])),
                    "invalid_cite": int(has_invalid_citation(ans)),
                    "mirror": int(mirrors_language(ans, q["lang"])),
                    "halluc": "" if h is None else int(h), "omission": "" if o is None else int(o),
                    "reason": reason.replace("\n", " "),
                    "rouge_l": round(rouge_l(ans, q["reference_answer"]), 3),
                    "latency_s": round(res.latency_s, 2), "answer": ans.replace("\n", " "),
                })
                print(f"downstream {name:<15}{q['id']} kp={ds_rows[-1]['kp_strict']}/{ds_rows[-1]['kp_alias']} "
                      f"hal={ds_rows[-1]['halluc']}", flush=True)
        with open(EVAL_DIR / "results_ablation_downstream.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(ds_rows[0].keys()))
            w.writeheader()
            w.writerows(ds_rows)
        for name, _, _ in CONFIGS:
            sub = [row for row in ds_rows if row["config"] == name]

            def mean(key, sub=sub):
                vals = [float(x[key]) for x in sub if x[key] != ""]
                return round(sum(vals) / len(vals), 3) if vals else None
            summary["downstream"][name] = {"n": len(sub), "kp_strict": mean("kp_strict"),
                                           "kp_alias": mean("kp_alias"), "citation_rate": mean("cite_ok"),
                                           "hallucination": mean("halluc"), "omission": mean("omission"),
                                           "mirror_rate": mean("mirror"), "rouge_l": mean("rouge_l")}

    (EVAL_DIR / "summary_ablation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
