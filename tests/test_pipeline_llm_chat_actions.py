import json

import pytest

from services import actions_service, auth, chat_service, db, llm_service, mom_service, pipeline, retrieval
from services.preprocessing import parse_transcript
from services.schemas import LLMCandidate, QAAnswer

TEXT = """[00:00:05] Rahul: Mark, please fix the refund cases by Friday.
[00:00:20] Mark: I'll fix the refund cases by Friday.
[00:00:40] Rahul: Someone needs to contact the vendor tomorrow.
[00:01:00] Rahul: We have decided to launch the beta on 1 November.
[00:01:20] Ana: Who is handling the contract?
"""


def make():
    mid = db.create_meeting("Sync", "2026-10-12", "transcript")
    db.replace_segments(mid, parse_transcript(TEXT))
    return mid


def test_classical_only_analysis_persists_and_never_invents_owners():
    mid = make()
    stats = pipeline.analyze_meeting(mid, use_llm=False)
    acts = db.rows("SELECT * FROM action_items WHERE meeting_id=?", (mid,))
    names = {"Mark", "Rahul", "Ana"}
    assert any(a["owner"] == "Mark" and a["deadline_date"] == "2026-10-16" for a in acts)
    assert all(a["owner"] in names or a["owner"] is None for a in acts)
    vendor = [a for a in acts if "vendor" in a["task"]]
    assert vendor and vendor[0]["owner"] is None and vendor[0]["status"] == "needs_review"
    assert db.rows("SELECT * FROM decisions WHERE meeting_id=?", (mid,))
    assert stats["llm_calls"] == 0 and db.rows("SELECT * FROM chunks WHERE meeting_id=?", (mid,))


def test_llm_answer_with_unsupported_owner_is_dropped_and_flagged(monkeypatch):
    mid = make()
    monkeypatch.setattr(llm_service, "is_configured", lambda: True)

    def fake(system, user, schema, max_tokens=600):
        return LLMCandidate(is_action_item=True, task="contact the vendor", owner="Zed Hallucinated",
                            deadline_text="next Thursday", evidence_quote="Someone needs to contact the vendor tomorrow",
                            uncertain_fields=[])
    monkeypatch.setattr(llm_service, "chat_json", fake)
    monkeypatch.setattr(llm_service, "chat_text", lambda *a, **k: "- summary")
    pipeline.analyze_meeting(mid, use_llm=True)
    row = [a for a in db.rows("SELECT * FROM action_items WHERE meeting_id=?", (mid,)) if "vendor" in a["task"]][0]
    assert row["owner"] is None and row["deadline_text"] is None  # unsupported -> removed, not trusted
    assert row["status"] == "needs_review" and row["source"] == "llm"


def test_llm_failure_degrades_to_review_not_crash(monkeypatch):
    mid = make()
    monkeypatch.setattr(llm_service, "is_configured", lambda: True)

    def boom(*a, **k):
        raise llm_service.LLMUnavailable("rate limited")
    monkeypatch.setattr(llm_service, "chat_json", boom)
    monkeypatch.setattr(llm_service, "chat_text", boom)
    stats = pipeline.analyze_meeting(mid, use_llm=True)
    assert stats["llm_failures"] >= 1 and stats["summary_method"] == "extractive"


def test_chat_refuses_when_citation_is_not_retrieved(monkeypatch):
    mid = make()
    retrieval.index_meeting(mid)
    monkeypatch.setattr(llm_service, "is_configured", lambda: True)
    monkeypatch.setattr(llm_service, "chat_json", lambda *a, **k: QAAnswer(answer="Launch is 5 May.", supported=True, citations=[99999]))
    res = chat_service.answer("What did we decide about the beta launch?", [mid])
    assert "5 May" not in res["answer"]


def test_chat_returns_supported_answer_with_valid_citation(monkeypatch):
    mid = make()
    retrieval.index_meeting(mid)
    hit = retrieval.keyword_search("beta launch", [mid])[0]
    monkeypatch.setattr(llm_service, "is_configured", lambda: True)
    monkeypatch.setattr(llm_service, "chat_json", lambda *a, **k: QAAnswer(answer="Launch the beta on 1 November.", supported=True, citations=[hit["chunk_id"]]))
    res = chat_service.answer("What did we decide about the beta launch?", [mid])
    assert res["supported"] and res["citations"][0]["chunk_id"] == hit["chunk_id"]


def test_chat_no_hits_says_not_supported():
    res = chat_service.answer("Anything?", [])
    assert res["supported"] is False


def test_ics_is_idempotent_and_confirmation_is_audited():
    mid = make()
    pipeline.analyze_meeting(mid, use_llm=False)
    a = [r for r in db.rows("SELECT * FROM action_items WHERE meeting_id=?", (mid,)) if r["deadline_date"]][0]
    p1 = actions_service.propose_event(mid, a["id"])
    p2 = actions_service.propose_event(mid, a["id"])
    assert p1 == p2
    ics = actions_service.build_ics(p1)
    assert "BEGIN:VEVENT" in ics and "DTSTART;VALUE=DATE:20261016" in ics
    actions_service.confirm_event(p1)
    assert db.one("SELECT 1 FROM audit_logs WHERE event='calendar_confirmed'")


def test_no_proposal_without_resolved_deadline():
    mid = make()
    pipeline.analyze_meeting(mid, use_llm=False)
    a = [r for r in db.rows("SELECT * FROM action_items WHERE meeting_id=?", (mid,)) if not r["deadline_date"]]
    if a:
        assert actions_service.propose_event(mid, a[0]["id"]) is None


def test_email_only_prepares_draft_and_validates_recipients():
    good, bad = actions_service.parse_recipients("a@b.com, nope, c@d.org")
    assert good == ["a@b.com", "c@d.org"] and bad == ["nope"]
    mid = make()
    d = actions_service.prepare_email(mid, good, "Subj", "# Title\n**bold**")
    assert d["mailto"].startswith("mailto:a@b.com") and "To: a@b.com" in d["eml"]
    assert db.one("SELECT status FROM email_deliveries")["status"] == "draft_prepared"


def test_mom_markdown_and_docx():
    mid = make()
    pipeline.analyze_meeting(mid, use_llm=False)
    md = mom_service.build_markdown(mid)
    assert "Minutes of Meeting" in md and "## Action Items" in md
    assert mom_service.to_docx_bytes(md)[:2] == b"PK"


def test_password_check(monkeypatch):
    monkeypatch.setenv("APP_PASSWORD", "test-pass")
    assert auth.check_password("test-pass") and not auth.check_password("wrong")


def test_json_extraction_helper():
    assert llm_service._extract_json('noise {"a": 1} tail') == {"a": 1}
