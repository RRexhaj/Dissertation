"""corpus.py — Load + hierarchically chunk Cap. 65 and friends.

Implements the article -> paragraph -> sentence chunking strategy described in
Chapter 2 of the dissertation (RAG section: hierarchical chunking, bilingual
metadata indexing). Each emitted chunk carries:

    source         — file name
    lang           — "en" | "mt"
    section_id     — e.g. "art. 45" / "reg. 9" / "FAQ"
    parent_section — coarser unit (e.g. Part / Schedule heading) when known
    chunk_idx      — index of this sub-chunk within its section
    citation       — display string used in the answer ("Cap. 65 art. 45")

Section detection is regex-based and tolerant: legislation.mt PDFs lose layout
during text extraction, so we look for line-starting patterns like "45." or
"SECTION 6" or "(1)" rather than depending on whitespace.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

DATA_ROOT = Path(__file__).parent / "data" / "cleaned_txt"

# Sub-chunk size targets (characters). Small enough for retrieval precision,
# large enough to keep an article subsection intact.
SUB_CHUNK_CHARS = 800
SUB_CHUNK_OVERLAP = 120

# Section openers we recognise in either language.
#   "45." / "45 ." / "SECTION 6" / "Artikolu 45" / "regulation 9"
SECTION_PATTERNS = [
    re.compile(r"(?im)^\s*(?P<id>\d{1,3}[A-Z]?)\s*\.\s+(?=\S)"),
    re.compile(r"(?im)^\s*SECTION\s+(?P<id>\d{1,3}[A-Z]?)\b"),
    re.compile(r"(?im)^\s*Artikolu\s+(?P<id>\d{1,3}[A-Z]?)\b"),
    re.compile(r"(?im)^\s*Regulation\s+(?P<id>\d{1,3}[A-Z]?)\b"),
    re.compile(r"(?im)^\s*Regolament\s+(?P<id>\d{1,3}[A-Z]?)\b"),
]

# Coarser headings used as parent_section labels.
PARENT_PATTERNS = [
    re.compile(r"(?im)^\s*PART\s+[IVXLCDM]+\b.*$"),
    re.compile(r"(?im)^\s*PARTI\s+[IVXLCDM]+\b.*$"),
    re.compile(r"(?im)^\s*FIRST\s+SCHEDULE\b.*$"),
    re.compile(r"(?im)^\s*SECOND\s+SCHEDULE\b.*$"),
    re.compile(r"(?im)^\s*L-EWWEL\s+SKEDA\b.*$"),
    re.compile(r"(?im)^\s*IT-TIENI\s+SKEDA\b.*$"),
]


@dataclass
class Chunk:
    text: str
    source: str
    lang: str
    section_id: str
    parent_section: str
    chunk_idx: int
    citation: str
    metadata_extra: dict = field(default_factory=dict)

    def to_metadata(self) -> dict:
        meta = {
            "source": self.source,
            "lang": self.lang,
            "section_id": self.section_id,
            "parent_section": self.parent_section,
            "chunk_idx": self.chunk_idx,
            "citation": self.citation,
        }
        meta.update(self.metadata_extra)
        return meta


def _normalise(text: str) -> str:
    # Collapse runaway whitespace from PDF extraction without destroying
    # paragraph boundaries.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _find_section_starts(text: str) -> list[tuple[int, str]]:
    """Return [(start_offset, section_id), ...] sorted by offset."""
    hits: list[tuple[int, str]] = []
    for pat in SECTION_PATTERNS:
        for m in pat.finditer(text):
            hits.append((m.start(), m.group("id")))
    hits.sort(key=lambda t: t[0])
    # Deduplicate near-identical hits at the same offset.
    deduped: list[tuple[int, str]] = []
    for off, sid in hits:
        if deduped and abs(deduped[-1][0] - off) < 4:
            continue
        deduped.append((off, sid))
    return deduped


def _nearest_parent(text: str, position: int) -> str:
    """Return the most recent PART/SCHEDULE heading before `position`."""
    best = ""
    for pat in PARENT_PATTERNS:
        for m in pat.finditer(text, 0, position + 1):
            best = m.group(0).strip()
    return best


def _split_section_body(body: str) -> list[str]:
    """Soft sub-chunking that prefers paragraph and subsection boundaries."""
    body = body.strip()
    if len(body) <= SUB_CHUNK_CHARS:
        return [body] if body else []

    # Prefer breaks at numbered subsections "(1)", "(a)", etc., else paragraphs.
    pieces: list[str] = []
    cursor = 0
    sub_re = re.compile(r"(?m)(?<=\n)\s*\((?:\d+|[a-z]+)\)\s")
    while cursor < len(body):
        end = min(cursor + SUB_CHUNK_CHARS, len(body))
        if end < len(body):
            window = body[cursor:end]
            m = None
            for cand in sub_re.finditer(window):
                m = cand
            if m and m.start() > SUB_CHUNK_CHARS // 2:
                end = cursor + m.start()
            else:
                nl = body.rfind("\n\n", cursor + SUB_CHUNK_CHARS // 2, end)
                if nl != -1:
                    end = nl
        piece = body[cursor:end].strip()
        if piece:
            pieces.append(piece)
        if end >= len(body):
            break
        cursor = max(end - SUB_CHUNK_OVERLAP, cursor + 1)
    return pieces


def _chunks_from_text(
    text: str,
    *,
    source: str,
    lang: str,
    citation_prefix: str,
    section_label: str,
) -> list[Chunk]:
    text = _normalise(text)
    if not text:
        return []

    starts = _find_section_starts(text)
    out: list[Chunk] = []

    if not starts:
        # No statutory structure detected (e.g. FAQ files): emit as one block.
        for idx, piece in enumerate(_split_section_body(text)):
            out.append(Chunk(
                text=piece,
                source=source,
                lang=lang,
                section_id="document",
                parent_section="",
                chunk_idx=idx,
                citation=f"{citation_prefix} ({source})",
            ))
        return out

    # Bracket each section by the next section's offset.
    boundaries = [(s, sid) for s, sid in starts] + [(len(text), "")]
    for i in range(len(boundaries) - 1):
        start, sid = boundaries[i]
        end, _ = boundaries[i + 1]
        body = text[start:end]
        parent = _nearest_parent(text, start)
        section_id = f"{section_label} {sid}".strip()
        citation = f"{citation_prefix} {section_label.replace('art.', 'art.').replace('reg.', 'reg.')} {sid}".strip()
        for j, piece in enumerate(_split_section_body(body)):
            out.append(Chunk(
                text=piece,
                source=source,
                lang=lang,
                section_id=section_id,
                parent_section=parent,
                chunk_idx=j,
                citation=citation,
            ))
    return out


# Citation prefixes per filename (file-name -> (citation_prefix, section_label))
CITATION_MAP = {
    "traffic_act.txt":           ("Cap. 65",         "art."),
    "micromobility_regs.txt":    ("S.L. 65.32",      "reg."),
    "traffic_signs_regs.txt":    ("S.L. 65.10",      "reg."),
    "speed_limits_regs.txt":     ("S.L. 65.11",      "reg."),
    "driving_licences.txt":      ("S.L. 65.18",      "reg."),
    "alcohol_regs.txt":          ("Cap. 65",         "art."),
    "enforcement_camera.txt":    ("S.L. 65.33",      "reg."),
    "motor_vehicle_insurance.txt": ("S.L. 65.04",   "reg."),
    "penalty_points.txt":        ("S.L. 65.18",      "art."),
    "payment_options.txt":       ("Driver FAQ",      ""),
    "traffic_faq.txt":           ("Driver FAQ",      ""),
}


def load_corpus(root: Path | None = None) -> list[Chunk]:
    """Walk data/cleaned_txt/<lang>/*.txt and emit hierarchically-tagged chunks."""
    root = root or DATA_ROOT
    chunks: list[Chunk] = []
    for lang_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        lang = lang_dir.name
        for fp in sorted(lang_dir.glob("*.txt")):
            citation_prefix, section_label = CITATION_MAP.get(
                fp.name, (fp.stem.replace("_", " ").title(), "")
            )
            text = fp.read_text(encoding="utf-8", errors="ignore")
            chunks.extend(_chunks_from_text(
                text,
                source=fp.name,
                lang=lang,
                citation_prefix=citation_prefix,
                section_label=section_label,
            ))
    return chunks


if __name__ == "__main__":
    chunks = load_corpus()
    print(f"loaded {len(chunks)} chunks")
    by_lang: dict[str, int] = {}
    for c in chunks:
        by_lang[c.lang] = by_lang.get(c.lang, 0) + 1
    for lang, n in by_lang.items():
        print(f"  {lang}: {n}")
    if chunks:
        sample = chunks[0]
        print("\nfirst chunk:")
        print(f"  citation: {sample.citation}")
        print(f"  source:   {sample.source}")
        print(f"  lang:     {sample.lang}")
        print(f"  parent:   {sample.parent_section!r}")
        print(f"  text[:200]: {sample.text[:200]!r}")
