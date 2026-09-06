"""Offline tests for the hybrid retriever (BM25 path, language filter, confidence mapping).

Uses a deterministic hashing embedder so no API key, network or model download is
needed; the dense signal is switched off (alpha=0) where lexical behaviour is tested.
"""
import hashlib
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from corpus import Chunk  # noqa: E402
from retrieval import HybridRetriever, RetrievedChunk, _tokenize, aggregate_confidence  # noqa: E402


class HashEmbedder:
    """Bag-of-words hashed into 16 dimensions; deterministic and offline."""
    dim = 16

    def embed(self, texts):
        out = []
        for t in texts:
            v = [0.0] * self.dim
            for tok in _tokenize(t):
                v[int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16) % self.dim] += 1.0
            norm = sum(x * x for x in v) ** 0.5 or 1.0
            out.append([x / norm for x in v])
        return out


def _chunks():
    return [
        Chunk(text="Speed limit: 50 km/h in towns and villages, 80 km/h on the open road.",
              source="speed_limits_regs.txt", lang="en", section_id="reg. 127",
              parent_section="", chunk_idx=0, citation="S.L. 65.11 reg. 127"),
        Chunk(text="Alcohol limit: 50 milligrammes of alcohol per 100 millilitres of blood; breath 22 microgrammes.",
              source="traffic_act.txt", lang="en", section_id="art. 15",
              parent_section="", chunk_idx=0, citation="Cap. 65 art. 15"),
        Chunk(text="Limitu tal-veloċità: 50 km/s fil-bliet u l-irħula, 80 km/s barra.",
              source="speed_limits_regs.txt", lang="mt", section_id="reg. 127",
              parent_section="", chunk_idx=0, citation="S.L. 65.11 reg. 127"),
    ]


@pytest.fixture(scope="module")
def retriever(tmp_path_factory):
    r = HybridRetriever(HashEmbedder(), chroma_dir=tmp_path_factory.mktemp("chroma"),
                        alpha=0.0, rerank=False)
    r.index(_chunks(), rebuild=True)
    return r


def test_bm25_only_ranks_lexical_match_first(retriever):
    res = retriever.search("alcohol limit blood", k=2, lang="en")
    assert res, "no results"
    assert res[0].citation == "Cap. 65 art. 15"
    assert res[0].bm25_score >= res[-1].bm25_score


def test_language_filter_restricts_to_requested_language(retriever):
    res = retriever.search("speed limit", k=3, lang="mt")
    assert res and all(r.lang == "mt" for r in res)
    res_en = retriever.search("speed limit", k=3, lang="en")
    assert res_en and all(r.lang == "en" for r in res_en)


def test_retrieved_chunk_carries_metadata(retriever):
    res = retriever.search("towns villages open road", k=1, lang="en")
    assert res[0].metadata["section_id"] == "reg. 127"
    assert res[0].metadata["source"] == "speed_limits_regs.txt"


def _rc(score: float) -> RetrievedChunk:
    return RetrievedChunk(text="", metadata={}, dense_score=0.0, bm25_score=0.0,
                          rerank_score=score, fused_score=0.0)


def test_aggregate_confidence_thresholds():
    assert aggregate_confidence([_rc(6.0), _rc(5.0), _rc(4.0)])[0] == "high"
    assert aggregate_confidence([_rc(1.0), _rc(0.5), _rc(0.0)])[0] == "medium"
    assert aggregate_confidence([_rc(-2.0), _rc(-3.0), _rc(-4.0)])[0] == "low"
    assert aggregate_confidence([])[0] == "low"
