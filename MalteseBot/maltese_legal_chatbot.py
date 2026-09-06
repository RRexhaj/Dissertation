"""maltese_legal_chatbot.py — MalteseLegalBot v2 (thesis-aligned).

Bilingual retrieval-augmented chatbot for minor Maltese traffic-law queries.
Architecture matches the design described in Chapters 2-3 of the dissertation:

    fetch.py        bilingual Cap. 65 corpus from legislation.mt
    corpus.py       hierarchical chunking (article -> paragraph)
    retrieval.py    BM25 + dense (text-embedding-3-large) + cross-encoder rerank
    prompts.py      constrained prompt with inline-citation few-shot
    compliance.py   EU AI Act banner, GDPR pseudonymised logging, LESA contact
    maltese_legal_chatbot.py  (this file) — CLI driver

The previous milestone-4 prototype lives untouched in ./_v1_archive/.

Setup
-----
    pip install -r requirements.txt
    cp .env.example .env  # fill in OPENAI_API_KEY
    python fetch.py                          # one-time corpus build
    python maltese_legal_chatbot.py --rebuild  # one-time index build
    python maltese_legal_chatbot.py            # chat
"""
from __future__ import annotations

import argparse
import os
import sys

from dotenv import load_dotenv

import compliance
from corpus import load_corpus
from prompts import PROMPTS
from retrieval import HybridRetriever, aggregate_confidence, build_embedder

load_dotenv()

DEFAULT_LLM = "gpt-4o"


def _ensure_openai_key(local_embed: bool) -> None:
    if local_embed:
        return
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit(
            "OPENAI_API_KEY not set. Either export it / put it in .env, or run "
            "with --local-embeddings (HuggingFace MiniLM, no API calls)."
        )


def _build_chat_client(model: str):
    from openai import OpenAI
    return OpenAI(), model


def _format_context(retrieved) -> str:
    blocks: list[str] = []
    for r in retrieved:
        cite = r.metadata.get("citation", r.metadata.get("source", "unknown"))
        blocks.append(f"[{cite}]\n{r.text}")
    return "\n\n".join(blocks)


def chat_once(question: str, retriever: HybridRetriever, openai_client, model: str,
              session_logger: compliance.SessionLogger, *, lang_override: str | None = None,
              k: int = 5) -> str:
    lang = lang_override or compliance.detect_language(question)
    retrieved = retriever.search(question, k=k, lang=lang)

    if not retrieved:
        # Empty-context refusal (per system prompt rule 1).
        msg = (
            "Ma nsibx informazzjoni dwar dan f'Kap. 65. Ċempel lil-LESA fuq "
            "+356 2122 2253 jew avukat liċenzjat."
            if lang == "mt"
            else "I don't have information on that in Cap. 65. Contact LESA on "
                 "+356 2122 2253 or a licensed advocate."
        )
        session_logger.log(
            query=question, answer=msg, confidence="low",
            citations=[], retrieval_lang=lang, rerank_scores=[],
        )
        return msg

    context = _format_context(retrieved)
    messages = PROMPTS.to_messages(context=context, question=question)

    resp = openai_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.1,
    )
    answer = resp.choices[0].message.content.strip()

    label, score = aggregate_confidence(retrieved)
    cites = sorted({r.metadata.get("citation", r.metadata.get("source", "?"))
                    for r in retrieved})
    k_align = sum(1 for r in retrieved if r.rerank_score > 0 or r.fused_score > 0.3)
    confidence_line = compliance.format_confidence(
        label, score, k_align, len(retrieved), lang,
    )

    session_logger.log(
        query=question, answer=answer, confidence=label,
        citations=cites, retrieval_lang=lang,
        rerank_scores=[r.rerank_score for r in retrieved],
    )

    sources_label = "Sorsi" if lang == "mt" else "Sources"
    sources_block = "\n".join(f"  - {c}" for c in cites)
    return (
        f"{answer}\n\n"
        f"{confidence_line}\n"
        f"{sources_label}:\n{sources_block}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="MalteseLegalBot v2 — bilingual RAG over Cap. 65.")
    ap.add_argument("--rebuild", action="store_true", help="re-ingest the corpus into ChromaDB")
    ap.add_argument("--local-embeddings", action="store_true",
                    help="use HuggingFace MiniLM instead of OpenAI embeddings (no API cost)")
    ap.add_argument("--no-rerank", action="store_true", help="disable the cross-encoder rerank")
    ap.add_argument("--model", default=DEFAULT_LLM, help=f"OpenAI chat model (default: {DEFAULT_LLM})")
    ap.add_argument("--lang", choices=["en", "mt", "auto"], default="auto",
                    help="force the retrieval / answer language")
    ap.add_argument("--k", type=int, default=5, help="number of chunks to inject (default: 5)")
    args = ap.parse_args()

    _ensure_openai_key(args.local_embeddings)

    embedder = build_embedder(local=args.local_embeddings)
    retriever = HybridRetriever(embedder, rerank=not args.no_rerank)

    if args.rebuild or retriever.is_empty():
        chunks = load_corpus()
        if not chunks:
            sys.exit(
                "No chunks loaded. Run `python fetch.py` first to build "
                "data/cleaned_txt/{en,mt}/."
            )
        print(f"indexing {len(chunks)} chunks ...", flush=True)
        retriever.index(chunks, rebuild=args.rebuild)
        print("indexing done.")

    client, model = _build_chat_client(args.model)
    session = compliance.SessionLogger()

    banner_lang = "mt" if args.lang == "mt" else "en"
    print(compliance.banner(banner_lang))

    try:
        while True:
            try:
                q = input("You: ").strip()
            except EOFError:
                break
            if not q:
                continue
            if q.lower() in {"exit", "quit", ":q", "ohroġ"}:
                break
            override = args.lang if args.lang in ("en", "mt") else None
            try:
                answer = chat_once(q, retriever, client, model, session,
                                   lang_override=override, k=args.k)
            except Exception as e:
                # Don't crash the REPL on a single bad query.
                answer = f"[error] {type(e).__name__}: {e}"
            print(f"Bot: {answer}\n")
    except KeyboardInterrupt:
        pass

    print(compliance.disclaimer(banner_lang))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
