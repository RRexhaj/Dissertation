"""app.py — Streamlit web UI for MalteseLegalBot.

Run:
    streamlit run app.py

The UI adds streaming responses, a settings sidebar, a confidence badge, a
sources expander, and the EU AI Act disclosure banner on top of the same
retrieval/prompt/compliance modules used by the CLI.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import compliance
from compliance import LESA_CONTACT, detect_language, format_confidence
from corpus import load_corpus
from prompts import PROMPTS
from retrieval import HybridRetriever, aggregate_confidence, build_embedder

# ── Page config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="MalteseLegalBot",
    page_icon="🇲🇹",
    layout="wide",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    /* Confidence badges */
    .badge-high   { background:#1a7a3e; color:#fff; padding:2px 10px;
                    border-radius:12px; font-size:.82rem; font-weight:600; }
    .badge-medium { background:#b07a00; color:#fff; padding:2px 10px;
                    border-radius:12px; font-size:.82rem; font-weight:600; }
    .badge-low    { background:#a02020; color:#fff; padding:2px 10px;
                    border-radius:12px; font-size:.82rem; font-weight:600; }
    /* Disclaimer line */
    .disclaimer { font-size:.78rem; color:#888; margin-top:6px; }
    /* AI Act banner */
    .ai-banner { background:#1e3a5f; color:#cce; padding:8px 14px;
                 border-radius:6px; font-size:.84rem; margin-bottom:10px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Cached resources (created once per server process) ──────────────────────

@st.cache_resource(show_spinner="Loading retrieval index …")
def _get_retriever(rerank: bool) -> HybridRetriever:
    embedder = build_embedder(local=bool(os.getenv("LOCAL_EMBEDS")))
    retriever = HybridRetriever(embedder, rerank=rerank)
    if retriever.is_empty():
        chunks = load_corpus()
        if not chunks:
            st.error(
                "No corpus found. Run `python fetch.py` first to download "
                "Cap. 65 into data/cleaned_txt/{en,mt}/ then restart."
            )
            st.stop()
        retriever.index(chunks)
    return retriever


@st.cache_resource
def _get_openai(model: str):
    from openai import OpenAI
    return OpenAI(), model


@st.cache_resource
def _get_logger() -> compliance.SessionLogger:
    return compliance.SessionLogger()


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/7/73/Flag_of_Malta.svg/120px-Flag_of_Malta.svg.png",
        width=50,
    )
    st.title("MalteseLegalBot")
    st.caption("AI guide to Maltese road-traffic law")
    st.divider()

    lang_choice = st.selectbox(
        "Language",
        options=["auto", "en", "mt"],
        format_func=lambda x: {"auto": "🔍 Auto-detect", "en": "🇬🇧 English",
                                "mt": "🇲🇹 Maltese"}[x],
    )
    model_choice = st.selectbox(
        "Model",
        options=["gpt-4o", "gpt-4o-mini"],
        help="gpt-4o: best quality | gpt-4o-mini: cheaper",
    )
    k_choice = st.slider("Context chunks (k)", min_value=2, max_value=10, value=5,
                         help="Number of Cap. 65 excerpts injected into each prompt")
    rerank_choice = st.toggle("Cross-encoder rerank", value=True,
                              help="ms-marco cross-encoder reranks the BM25+dense candidates")
    show_context = st.toggle("Show retrieved excerpts", value=False,
                             help="Display the raw Cap. 65 chunks fed to the model")

    st.divider()
    if st.button("🔄 Rebuild index", use_container_width=True):
        # Drop only the cached retriever (leaving the OpenAI client and logger
        # intact), then rebuild the index on the cached instance so the rest of
        # the app uses the freshly rebuilt retriever rather than a discarded copy.
        _get_retriever.clear()
        with st.spinner("Rebuilding index …"):
            retriever = _get_retriever(rerank_choice)
            chunks = load_corpus()
            retriever.index(chunks, rebuild=True)
        st.success(f"Index rebuilt — {len(chunks)} chunks")

    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption(
        "⚠️ **Not legal advice.** For your specific case contact "
        f"**{LESA_CONTACT}** or a licensed advocate."
    )

# ── EU AI Act disclosure banner ──────────────────────────────────────────────

st.markdown(
    '<div class="ai-banner">'
    "🤖 <b>EU AI Act art. 50 disclosure</b> — You are interacting with an AI system. "
    "Outputs are AI-generated content. Scope: Maltese road-traffic law (Cap. 65 + subsidiary). "
    f"<b>Not legal advice.</b> For your specific case contact {LESA_CONTACT}."
    "</div>",
    unsafe_allow_html=True,
)

# ── Chat history ──────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "⚖️"):
        st.markdown(msg["content"], unsafe_allow_html=True)
        if msg["role"] == "assistant" and msg.get("meta"):
            _meta = msg["meta"]
            _label = _meta.get("confidence_label", "")
            _score = _meta.get("confidence_score", 0.0)
            _k_align = _meta.get("k_align", 0)
            _k_total = _meta.get("k_total", 0)
            _lang = _meta.get("lang", "en")
            _badge_cls = f"badge-{_label}"
            badge_text = {"high": "✅ High", "medium": "⚠️ Medium", "low": "❌ Low"}.get(_label, _label)
            st.markdown(
                f'<span class="{_badge_cls}">{badge_text} confidence</span> '
                f'<span style="font-size:.78rem;color:#888;">'
                f"({_k_align}/{_k_total} sources align, score={_score:.2f})</span>",
                unsafe_allow_html=True,
            )
            if _meta.get("citations"):
                with st.expander("📄 Sources"):
                    for c in _meta["citations"]:
                        st.markdown(f"- `{c}`")
            if show_context and _meta.get("context_chunks"):
                with st.expander("🔎 Retrieved excerpts"):
                    for chunk in _meta["context_chunks"]:
                        st.markdown(f"**{chunk['citation']}** (`{chunk['lang']}`)")
                        st.text(chunk["text"][:400] + ("…" if len(chunk["text"]) > 400 else ""))
                        st.divider()
            st.markdown(
                f'<div class="disclaimer">AI-generated · not legal advice · {LESA_CONTACT}</div>',
                unsafe_allow_html=True,
            )

# ── Input + response ─────────────────────────────────────────────────────────

placeholder_en = "Ask about Maltese traffic law… e.g. 'What is the fine for speeding?'"
placeholder_mt = "Staqsi dwar il-liġi tat-traffiku Maltija… eż. 'X'inhi l-multa għal sewqan b'veloċità żejda?'"
placeholder = placeholder_mt if lang_choice == "mt" else placeholder_en

if question := st.chat_input(placeholder):
    # Show user message.
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(question)

    # Determine language.
    lang = lang_choice if lang_choice in ("en", "mt") else detect_language(question)

    # Retrieve.
    retriever = _get_retriever(rerank_choice)
    with st.spinner("Searching Cap. 65 …"):
        retrieved = retriever.search(question, k=k_choice, lang=lang if lang != "auto" else None)

    # Assemble prompt.
    if not retrieved:
        no_info = (
            "Ma nsibx informazzjoni dwar dan f'Kap. 65. "
            f"Ċempel lil-LESA fuq {LESA_CONTACT} jew avukat liċenzjat."
            if lang == "mt"
            else f"I don't have information on that in Cap. 65. "
                 f"Contact {LESA_CONTACT} or a licensed advocate."
        )
        with st.chat_message("assistant", avatar="⚖️"):
            st.markdown(no_info)
        st.session_state.messages.append({"role": "assistant", "content": no_info, "meta": None})
        st.stop()

    context = "\n\n".join(
        f"[{r.metadata.get('citation', r.metadata.get('source', '?'))}]\n{r.text}"
        for r in retrieved
    )
    messages = PROMPTS.to_messages(context=context, question=question)

    # Stream the response.
    openai_client, model = _get_openai(model_choice)
    with st.chat_message("assistant", avatar="⚖️"):
        response_container = st.empty()
        full_answer = ""
        stream = openai_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.1,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            full_answer += delta
            response_container.markdown(full_answer + "▌")
        response_container.markdown(full_answer)

        # Confidence + sources.
        label, score = aggregate_confidence(retrieved)
        cites = sorted({r.metadata.get("citation", r.metadata.get("source", "?"))
                        for r in retrieved})
        k_align = sum(1 for r in retrieved if r.rerank_score > 0 or r.fused_score > 0.3)
        badge_cls = f"badge-{label}"
        badge_text = {"high": "✅ High", "medium": "⚠️ Medium", "low": "❌ Low"}.get(label, label)
        st.markdown(
            f'<span class="{badge_cls}">{badge_text} confidence</span> '
            f'<span style="font-size:.78rem;color:#888;">'
            f"({k_align}/{len(retrieved)} sources align, score={score:.2f})</span>",
            unsafe_allow_html=True,
        )
        with st.expander("📄 Sources"):
            for c in cites:
                st.markdown(f"- `{c}`")
        if show_context:
            with st.expander("🔎 Retrieved excerpts"):
                for r in retrieved:
                    st.markdown(
                        f"**{r.metadata.get('citation', '?')}** "
                        f"(`{r.metadata.get('lang', '?')}` · "
                        f"fused={r.fused_score:.2f} rerank={r.rerank_score:.2f})"
                    )
                    st.text(r.text[:400] + ("…" if len(r.text) > 400 else ""))
                    st.divider()
        st.markdown(
            f'<div class="disclaimer">AI-generated · not legal advice · {LESA_CONTACT}</div>',
            unsafe_allow_html=True,
        )

    # Log (pseudonymised).
    _get_logger().log(
        query=question,
        answer=full_answer,
        confidence=label,
        citations=cites,
        retrieval_lang=lang,
        rerank_scores=[r.rerank_score for r in retrieved],
    )

    # Persist to session.
    meta = {
        "confidence_label": label,
        "confidence_score": score,
        "k_align": k_align,
        "k_total": len(retrieved),
        "lang": lang,
        "citations": cites,
        "context_chunks": [
            {
                "citation": r.metadata.get("citation", r.metadata.get("source", "?")),
                "lang": r.metadata.get("lang", "?"),
                "text": r.text,
            }
            for r in retrieved
        ],
    }
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_answer,
        "meta": meta,
    })
