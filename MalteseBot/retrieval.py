"""retrieval.py — Hybrid BM25 + dense retrieval with cross-encoder reranking.

This is the architecture prescribed in Chapter 2 of the dissertation:

    1. Dense retrieval over a ChromaDB collection of bilingual Cap. 65 chunks
       (text-embedding-3-large, d=3072) for semantic recall.
    2. BM25 over the same chunks for lexical coverage of statute-specific
       terms (e.g. "ammenda", "demerit points").
    3. Reciprocal-rank fusion with alpha=0.4 weighting on the dense side
       (matches the section 2.3.3 mitigation table).
    4. Cross-encoder rerank pass (sentence-transformers/ms-marco-MiniLM-L-6-v2)
       on the top-N fused candidates.
    5. Language filter so Maltese queries don't get drowned by English chunks.

ChromaDB persists at ./chroma_db relative to the project so the retrieval
index survives between runs.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from rank_bm25 import BM25Okapi

from corpus import Chunk

CHROMA_DIR = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "maltese_traffic_law"

logger = logging.getLogger(__name__)


# ---- Embeddings -----------------------------------------------------------

class _OpenAIEmbedder:
    """Thin wrapper around OpenAI v1 embeddings client."""

    def __init__(self, model: str = "text-embedding-3-large"):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model
        self.dim = 3072 if "large" in model else 1536

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        # OpenAI tolerates batches up to 2048 inputs; statute corpora are small.
        out: list[list[float]] = []
        BATCH = 64
        for i in range(0, len(texts), BATCH):
            resp = self.client.embeddings.create(
                model=self.model,
                input=list(texts[i:i + BATCH]),
            )
            out.extend(d.embedding for d in resp.data)
        return out


class _LocalEmbedder:
    """HuggingFace fallback for offline / zero-cost runs."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return self.model.encode(list(texts), show_progress_bar=False).tolist()


def build_embedder(local: bool):
    if local or os.getenv("LOCAL_EMBEDS"):
        return _LocalEmbedder()
    return _OpenAIEmbedder()


# ---- BM25 -----------------------------------------------------------------

_TOKEN_RE = __import__("re").compile(r"\w+", __import__("re").UNICODE)


def _tokenize(s: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(s)]


# ---- Reranker -------------------------------------------------------------

class _CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: Sequence[str]) -> list[float]:
        if not candidates:
            return []
        pairs = [[query, c] for c in candidates]
        return self.model.predict(pairs).tolist()


# ---- Retriever ------------------------------------------------------------

@dataclass
class RetrievedChunk:
    text: str
    metadata: dict
    dense_score: float
    bm25_score: float
    rerank_score: float
    fused_score: float

    @property
    def citation(self) -> str:
        return self.metadata.get("citation", self.metadata.get("source", "unknown"))

    @property
    def lang(self) -> str:
        return self.metadata.get("lang", "en")


class HybridRetriever:
    """Hybrid BM25 + dense retrieval over a persistent ChromaDB collection."""

    def __init__(
        self,
        embedder,
        *,
        chroma_dir: Path = CHROMA_DIR,
        alpha: float = 0.6,
        rerank: bool = True,
    ):
        import chromadb
        chroma_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(chroma_dir))
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self.embedder = embedder
        self.alpha = alpha  # weight on dense; (1-alpha) on bm25
        self._bm25: BM25Okapi | None = None
        self._bm25_docs: list[str] = []
        self._bm25_metas: list[dict] = []
        self._reranker = _CrossEncoderReranker() if rerank else None

    # -- index ----------------------------------------------------------

    def is_empty(self) -> bool:
        try:
            return self._collection.count() == 0
        except Exception:
            return True

    def index(self, chunks: list[Chunk], *, rebuild: bool = False) -> None:
        if rebuild and self._collection.count() > 0:
            self._client.delete_collection(COLLECTION_NAME)
            self._collection = self._client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )

        if self._collection.count() > 0 and not rebuild:
            logger.info("collection already populated (%d) — skipping ingest",
                        self._collection.count())
            self._load_bm25_from_collection()
            return

        ids = [f"c{i:05d}" for i in range(len(chunks))]
        docs = [c.text for c in chunks]
        metas = [c.to_metadata() for c in chunks]

        logger.info("embedding %d chunks", len(docs))
        embeddings = self.embedder.embed(docs)

        # Chroma writes in batches of <= ~5k.
        BATCH = 500
        for i in range(0, len(ids), BATCH):
            self._collection.add(
                ids=ids[i:i + BATCH],
                documents=docs[i:i + BATCH],
                metadatas=metas[i:i + BATCH],
                embeddings=embeddings[i:i + BATCH],
            )
        self._bm25_docs = docs
        self._bm25_metas = metas
        self._bm25 = BM25Okapi([_tokenize(d) for d in docs])

    def _load_bm25_from_collection(self) -> None:
        # Re-pull every doc from Chroma to (re)build BM25 in memory.
        got = self._collection.get(include=["documents", "metadatas"])
        self._bm25_docs = list(got["documents"])
        self._bm25_metas = list(got["metadatas"])
        if self._bm25_docs:
            self._bm25 = BM25Okapi([_tokenize(d) for d in self._bm25_docs])

    # -- query ----------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        k: int = 5,
        candidates: int = 25,
        lang: str | None = None,
    ) -> list[RetrievedChunk]:
        if self._bm25 is None:
            self._load_bm25_from_collection()
        if not self._bm25_docs:
            return []

        # 1. dense
        q_vec = self.embedder.embed([query])[0]
        where = {"lang": lang} if lang else None
        dense = self._collection.query(
            query_embeddings=[q_vec],
            n_results=candidates,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        dense_docs = dense["documents"][0]
        dense_metas = dense["metadatas"][0]
        dense_dists = dense["distances"][0]
        # cosine distance -> similarity in [0,1]
        dense_scores = {
            (m.get("source"), m.get("section_id"), m.get("chunk_idx")): 1 - d
            for m, d in zip(dense_metas, dense_dists)
        }

        # 2. bm25
        bm25_raw = self._bm25.get_scores(_tokenize(query))
        if bm25_raw.max() > 0:
            bm25_norm = bm25_raw / bm25_raw.max()
        else:
            bm25_norm = bm25_raw
        bm25_pairs: list[tuple[float, str, dict]] = []
        for i, m in enumerate(self._bm25_metas):
            if lang and m.get("lang") != lang:
                continue
            bm25_pairs.append((float(bm25_norm[i]), self._bm25_docs[i], m))
        bm25_pairs.sort(key=lambda t: t[0], reverse=True)
        bm25_pairs = bm25_pairs[:candidates]
        bm25_scores = {
            (m.get("source"), m.get("section_id"), m.get("chunk_idx")): s
            for s, _, m in bm25_pairs
        }

        # 3. fuse
        all_keys = set(dense_scores) | set(bm25_scores)
        fused: list[RetrievedChunk] = []
        # Build a quick lookup for text by key.
        all_docs = {(m.get("source"), m.get("section_id"), m.get("chunk_idx")): (d, m)
                    for d, m in zip(dense_docs, dense_metas)}
        for s, d, m in bm25_pairs:
            all_docs.setdefault((m.get("source"), m.get("section_id"), m.get("chunk_idx")), (d, m))

        for key in all_keys:
            d_score = dense_scores.get(key, 0.0)
            b_score = bm25_scores.get(key, 0.0)
            fused_score = self.alpha * d_score + (1 - self.alpha) * b_score
            text, meta = all_docs[key]
            fused.append(RetrievedChunk(
                text=text,
                metadata=meta,
                dense_score=d_score,
                bm25_score=b_score,
                rerank_score=0.0,
                fused_score=fused_score,
            ))
        fused.sort(key=lambda r: r.fused_score, reverse=True)
        fused = fused[:k * 3]

        # 4. rerank (if enabled and we have the model)
        if self._reranker and fused:
            scores = self._reranker.rerank(query, [r.text for r in fused])
            for r, s in zip(fused, scores):
                r.rerank_score = float(s)
            fused.sort(key=lambda r: r.rerank_score, reverse=True)

        return fused[:k]


def aggregate_confidence(retrieved: Iterable[RetrievedChunk]) -> tuple[str, float]:
    """Map top-k rerank/fused scores into a discrete trust label.

    The thresholds align with the "High (3/3 sources align)" UX cue described
    in section 2.5.1 of the dissertation. They are deliberately conservative —
    bilingual rerank scores are noisier than monolingual ones.
    """
    retrieved = list(retrieved)
    if not retrieved:
        return ("low", 0.0)
    # If reranker scores are populated, prefer those (they're sigmoid-ish).
    if any(r.rerank_score for r in retrieved):
        score = sum(r.rerank_score for r in retrieved[:3]) / min(3, len(retrieved))
        # ms-marco cross-encoder logits typically span [-10, +10].
        if score >= 4.0:
            return ("high", score)
        if score >= 0.5:
            return ("medium", score)
        return ("low", score)
    score = sum(r.fused_score for r in retrieved[:3]) / min(3, len(retrieved))
    if score >= 0.55:
        return ("high", score)
    if score >= 0.30:
        return ("medium", score)
    return ("low", score)
