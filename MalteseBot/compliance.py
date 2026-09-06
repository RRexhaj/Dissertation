"""compliance.py — EU AI Act + GDPR scaffolding for the prototype.

Three concerns from Chapter 2.5 of the dissertation are operationalised here:

  * Transparency (EU AI Act limited-risk obligation, applicable Feb 2026):
    a startup banner discloses AI status, scope, and limitations; every
    response carries an "AI-generated" tag.

  * Pseudonymised logging (GDPR / IDPC, sec. 2.5.4): we log query + answer
    hashes plus a sha256-derived pseudonym of a per-process session id.
    No IP addresses, no raw user identifiers.

  * Trust calibration (sec. 2.5.1): the confidence label from retrieval is
    surfaced to the user as "High / Medium / Low" plus the underlying
    "k of n sources align" cue.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

LESA_CONTACT = "LESA +356 2122 2253"

EU_AI_ACT_BANNER_EN = """\
============================================================
 MalteseLegalBot  ·  AI-powered informational tool
 Scope: Maltese road-traffic law (Cap. 65 + subsidiary)
 NOT legal advice.  For your specific case contact {lesa}.
 EU AI Act art. 50 disclosure: you are interacting with an
 AI system. Outputs are AI-generated content.
 GDPR: queries are pseudonymised before logging; no IP is
 stored. Type 'exit' to quit.
============================================================
""".format(lesa=LESA_CONTACT)

EU_AI_ACT_BANNER_MT = """\
============================================================
 MalteseLegalBot  ·  Għodda informattiva mħaddma mill-IA
 Kamp: Liġi tat-traffiku Maltija (Kap. 65 + sussidjarji)
 MHIX parir legali. Għall-każ speċifiku tiegħek ċempel
 lil {lesa}.
 Att tal-IA tal-UE art. 50: qed tinteraġixxi ma' sistema IA.
 L-output huwa kontenut iġġenerat mill-IA.
 GDPR: il-mistoqsijiet jiġu pseudonymised qabel ma jiġu
 lloggjati; l-ebda IP ma jinħażen. Ikteb 'exit' biex tieqaf.
============================================================
""".format(lesa=LESA_CONTACT)


def banner(lang: str = "en") -> str:
    return EU_AI_ACT_BANNER_MT if lang == "mt" else EU_AI_ACT_BANNER_EN


@dataclass
class SessionLogger:
    """Pseudonymised JSON-Lines logger satisfying GDPR Art. 25."""

    log_path: Path = field(default_factory=lambda: Path(__file__).parent / "logs" / "queries.jsonl")
    salt: str = field(default_factory=lambda: secrets.token_hex(8))
    session_id: str = field(default_factory=lambda: secrets.token_hex(8))

    def __post_init__(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def _pseudonym(self) -> str:
        return hashlib.sha256(f"{self.session_id}:{self.salt}".encode()).hexdigest()[:16]

    def log(
        self,
        *,
        query: str,
        answer: str,
        confidence: str,
        citations: list[str],
        retrieval_lang: str,
        rerank_scores: list[float],
    ) -> None:
        record = {
            "ts": int(time.time()),
            "session": self._pseudonym(),
            # We hash the raw query to keep the log usable for reproducibility
            # studies without retaining the natural-language text itself.
            "q_hash": hashlib.sha256(query.encode("utf-8")).hexdigest()[:16],
            "q_lang": retrieval_lang,
            "answer_len": len(answer),
            "confidence": confidence,
            "citations": citations,
            "rerank_scores": [round(s, 4) for s in rerank_scores],
        }
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def format_confidence(label: str, score: float, k_align: int, k_total: int, lang: str) -> str:
    """Human-readable trust-calibration cue."""
    if lang == "mt":
        labels = {"high": "Għolja", "medium": "Medja", "low": "Baxxa"}
        return f"Fiduċja: {labels.get(label, label)} ({k_align}/{k_total} sorsi jaqblu, score={score:.2f})"
    labels = {"high": "High", "medium": "Medium", "low": "Low"}
    return f"Confidence: {labels.get(label, label)} ({k_align}/{k_total} sources align, score={score:.2f})"


def disclaimer(lang: str = "en") -> str:
    if lang == "mt":
        return f"Iġġenerat mill-IA, mhux parir legali. Għal pariri speċifiċi ċempel lil {LESA_CONTACT}."
    return f"AI-generated, not legal advice. For your specific case contact {LESA_CONTACT}."


_MT_WORDS = {
    "il", "l", "tal", "tat", "tad", "tar", "tas", "lil", "fil", "mill", "bil", "mal", "mat", "ghal",
    "ghall", "ta", "ma", "u", "li", "biex", "jekk", "jien", "int", "huwa", "hija", "hi", "ahna",
    "intom", "huma", "imma", "izda", "qieghed", "qed", "kif", "fejn", "min", "x", "dan", "din",
    "dawn", "hemm", "hawn", "mhux", "le", "iva", "kull", "wara", "qabel", "fuq", "taht", "sa",
    "jew", "ghax", "meta", "nista", "tista", "jista", "trid", "irid", "ghandek", "ghandi", "ghandu",
    "inhu", "inhi", "kemm", "liema", "ghaliex", "ukoll", "biss", "xi", "hu", "jiena",
}
_EN_WORDS = {
    "the", "is", "are", "of", "and", "to", "in", "a", "an", "you", "your", "for", "if", "not",
    "on", "with", "it", "this", "that", "be", "can", "or", "at", "by", "from", "as", "was",
    "will", "must", "may", "have", "has", "do", "does", "what", "when", "how", "which", "who",
    "there", "no", "any", "should", "would", "could", "am", "i", "my", "me", "we",
}
_WORD_RE = re.compile(r"[a-z]+")


def detect_language(text: str) -> str:
    """Light-weight Maltese-vs-English heuristic.

    Maltese-specific characters (ċ ġ ħ ż and capitals) decide immediately.
    Otherwise the text is tokenised into whole words and the number of
    high-frequency Maltese function words is compared with the number of
    English ones. Whole-word matching replaces the substring matching of the
    June 2026 version, which classified long English answers as Maltese
    because "li" occurs inside "limit" and "hi" inside "this".
    """
    mt_chars = set("ċġħżĊĠĦŻ")
    if any(ch in mt_chars for ch in text):
        return "mt"
    tokens = _WORD_RE.findall(text.lower())
    mt_hits = sum(1 for t in tokens if t in _MT_WORDS)
    en_hits = sum(1 for t in tokens if t in _EN_WORDS)
    if mt_hits >= 2 and mt_hits > en_hits:
        return "mt"
    return "en"
