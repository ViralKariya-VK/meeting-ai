from datetime import date

from services import classical_nlp, extraction
from services.confidence_router import route

MEETING = date(2026, 10, 12)


def cand(text, speaker=None, context=None, kind="action", conf=0.95, margin=0.8):
    sent = {"seg_idx": 0, "text": text, "speaker": speaker}
    pred = {"label": "ACTION_ITEM" if kind == "action" else "DECISION", "conf": conf, "margin": margin, "probs": {}}
    return extraction.build_candidate(sent, pred, kind, context or text, MEETING)


def test_named_owner_task_deadline():
    c = cand("Rahul will send the project report to Ananya by Monday.")
    assert c.owner == "Rahul" and c.owner_flag == "ok"
    assert c.task.startswith("send the project report")
    assert c.deadline_text.lower().endswith("monday")


def test_first_person_uses_speaker_name():
    c = cand("I'll fix the refund cases by Friday.", speaker="Mark")
    assert c.owner == "Mark"
    assert route(c) is None


def test_first_person_without_named_speaker_is_not_guessed():
    c = cand("I'll fix the refund cases by Friday.", speaker="SPEAKER_01")
    assert c.owner is None
    assert route(c) == "OWNER_UNCERTAIN"


def test_unowned_task_is_uncertain():
    c = cand("Someone should contact the vendor tomorrow.")
    assert c.owner is None
    assert route(c) == "OWNER_UNCERTAIN"


def test_vague_deadline_and_hedge_routed():
    assert route(cand("Rahul will finish the charts soon.")) == "DEADLINE_AMBIGUOUS"
    assert route(cand("Rahul will finish the charts, unless the API slips.")) == "CONDITIONAL_COMMITMENT"


def test_low_confidence_and_margin_routed():
    assert route(cand("Rahul will send it by Friday.", conf=0.6)) == "LOW_CLASSIFICATION_CONFIDENCE"
    assert route(cand("Rahul will send it by Friday.", margin=0.05)) == "LOW_CLASS_MARGIN"


def test_cue_check_blocks_confident_but_cueless_sentences():
    assert route(cand("Spending on cloud hosting went up by twelve percent this quarter.")) == "NO_ACTION_CUE"
    assert route(cand("The beta feedback has been mostly positive.", kind="decision")) == "NO_DECISION_CUE"


def test_clean_decision_is_reliable():
    assert route(cand("We have decided to launch the beta on 1 November.", kind="decision")) is None


def test_conflicting_deadlines_flagged():
    a = cand("Mark will fix the refund cases by Friday.")
    b = cand("Mark will fix the refund cases by Wednesday.")
    b.seg_idx = 5
    extraction.detect_conflicts([a, b])
    assert a.conflict and b.conflict
    assert route(a) == "CONFLICTING_EXTRACTIONS"


def test_request_and_acknowledgement_are_merged():
    a = cand("Can you fix those refund cases by Friday, Mark?")
    b = cand("I'll fix the refund cases by Friday.", speaker="Mark")
    b.seg_idx, b.evidence_idx = 1, [1]
    merged = extraction.dedupe([a, b])
    assert len(merged) == 1 and merged[0].evidence_idx == [0, 1]


def test_rules_make_questions_and_backchannel_non_actions():
    out = classical_nlp.predict(["Who is handling the vendor contract?", "Okay.", "Rahul will send the report by Monday."])
    assert out[0]["label"] == "QUESTION" and out[1]["label"] == "DISCUSSION"
    assert out[2]["label"] == "ACTION_ITEM"


def test_negation_is_preserved_by_preprocessing():
    from services.preprocessing import split_sentences
    assert split_sentences("I will not send the report.") == ["I will not send the report."]
