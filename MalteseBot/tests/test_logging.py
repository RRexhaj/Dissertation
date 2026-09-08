"""Pseudonymised logging: no raw query text is written, feedback is linked by hash."""
import json

from compliance import SessionLogger


def test_query_log_is_pseudonymised(tmp_path):
    logger = SessionLogger(log_path=tmp_path / "q.jsonl")
    logger.log(query="What is the speed limit in Valletta?", answer="50 km/h", confidence="high",
               citations=["S.L. 65.11 reg. 127"], retrieval_lang="en", rerank_scores=[0.91])
    lines = (tmp_path / "q.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert "Valletta" not in lines[0]
    assert rec["q_lang"] == "en" and rec["confidence"] == "high"
    assert len(rec["q_hash"]) == 16 and len(rec["session"]) == 16


def test_feedback_is_linked_by_hash_only(tmp_path):
    logger = SessionLogger(log_path=tmp_path / "q.jsonl")
    q, a = "Can I refuse a breath test?", "No, refusal is an offence."
    logger.log(query=q, answer=a, confidence="medium", citations=[], retrieval_lang="en", rerank_scores=[])
    logger.log_feedback(query=q, answer=a, rating="down", reason="  too short  ", citations=["Cap. 65 art. 15G"])
    recs = [json.loads(l) for l in (tmp_path / "q.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(recs) == 2
    fb = recs[1]
    assert fb["kind"] == "feedback" and fb["rating"] == "down" and fb["reason"] == "too short"
    assert fb["q_hash"] == recs[0]["q_hash"]
    assert "refuse" not in json.dumps(fb)
