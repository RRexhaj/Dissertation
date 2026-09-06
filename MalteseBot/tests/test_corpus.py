"""Unit + regression tests for the hierarchical chunker and the validated corpus."""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from corpus import SUB_CHUNK_CHARS, _chunks_from_text, _split_section_body, load_corpus  # noqa: E402


def test_section_detection_citation_and_parent():
    text = ("PART II\n"
            "15. General limit.\n(1) The limit is 50 milligrammes per 100 millilitres.\n"
            "15A. Contravention notice.\nA fine doubles after fifteen days and triples after thirty.\n")
    chunks = _chunks_from_text(text, source="traffic_act.txt", lang="en",
                               citation_prefix="Cap. 65", section_label="art.")
    assert [c.section_id for c in chunks] == ["art. 15", "art. 15A"]
    assert chunks[0].citation == "Cap. 65 art. 15"
    assert chunks[1].citation == "Cap. 65 art. 15A"
    assert chunks[0].parent_section.startswith("PART II")
    assert all(c.lang == "en" and c.source == "traffic_act.txt" for c in chunks)
    meta = chunks[0].to_metadata()
    assert set(meta) >= {"source", "lang", "section_id", "parent_section", "chunk_idx", "citation"}


def test_maltese_section_opener_recognised():
    text = "Artikolu 15\nIl-limitu huwa 50 mg.\nArtikolu 16\nRegola oħra.\n"
    chunks = _chunks_from_text(text, source="traffic_act.txt", lang="mt",
                               citation_prefix="Cap. 65", section_label="art.")
    assert [c.section_id for c in chunks] == ["art. 15", "art. 16"]


def test_unstructured_document_becomes_single_block():
    chunks = _chunks_from_text("How do I pay? Online at contraventions.gov.mt within 15 days.",
                               source="traffic_faq.txt", lang="en",
                               citation_prefix="Driver FAQ", section_label="")
    assert len(chunks) == 1
    assert chunks[0].section_id == "document"
    assert chunks[0].citation == "Driver FAQ (traffic_faq.txt)"


def test_long_section_is_split_within_size_limit_and_covers_text():
    paragraph = "The registered owner shall be liable for the contravention. " * 6
    body = "\n\n".join(paragraph for _ in range(6))  # ~2.2k characters
    pieces = _split_section_body(body)
    assert len(pieces) > 1
    assert all(len(p) <= SUB_CHUNK_CHARS for p in pieces)
    assert body.strip().endswith(pieces[-1][-40:])


def test_real_corpus_counts_match_dissertation():
    # Regression guard: Chapter 3 reports 136 chunks, 78 English and 58 Maltese.
    chunks = load_corpus()
    by_lang = {}
    for c in chunks:
        by_lang[c.lang] = by_lang.get(c.lang, 0) + 1
    assert len(chunks) == 136
    assert by_lang == {"en": 78, "mt": 58}
    assert all(c.citation for c in chunks)
    assert all(c.text.strip() for c in chunks)
