"""Confidence gateway: decides whether a classical candidate is reliable or must go to the LLM / human review."""
from __future__ import annotations

import re
from typing import Optional

from pydantic import ValidationError

from . import config
from .extraction import Candidate
from .schemas import ActionItemModel, DecisionModel

REASONS = [
    "LOW_CLASSIFICATION_CONFIDENCE", "LOW_CLASS_MARGIN", "NO_ACTION_CUE", "NO_DECISION_CUE", "OWNER_UNCERTAIN",
    "DEADLINE_AMBIGUOUS", "CONDITIONAL_COMMITMENT", "MISSING_EVIDENCE", "SCHEMA_VALIDATION_FAILED",
    "CONFLICTING_EXTRACTIONS",
]

# A confident classifier is not enough: the sentence must also contain a lexical cue of the claimed type.
_ACTION_CUE = re.compile(r"\b(will|won't|shall|'ll|going to|gonna|has to|have to|needs? to|need\b|must|should|"
                         r"please|can you|could you|would you|responsible|action|take care|agreed to|promised|"
                         r"to (?:do|send|fix|update|review|prepare|share|contact|write|draft|test|book|finish|label))", re.I)
_DECISION_CUE = re.compile(r"\b(decid\w*|agreed?|approv\w*|settled|go(?:ing)? with|go(?:ing)? ahead|final\w*|"
                           r"decision|voted|freeze|cancel\w*|postpone\w*|drop|adopt\w*|let's (?:go|do|keep|use)|"
                           r"we(?:'ll| will) (?:use|keep|stay|launch|move)|it is|it's)\b", re.I)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", re.sub(r"\s+", " ", (s or "").lower())).strip()


def supported_by(text: Optional[str], context: str) -> bool:
    return not text or _norm(text) in _norm(context)


def validate_schema(c: Candidate) -> Optional[str]:
    try:
        if c.kind == "action":
            ActionItemModel(task=c.task or "", owner=c.owner, deadline_text=c.deadline_text,
                            deadline_date=c.deadline_date, evidence_idx=c.evidence_idx)
        else:
            DecisionModel(text=c.task or "", evidence_idx=c.evidence_idx)
    except ValidationError as e:
        return str(e.errors()[0]["loc"])
    return None


def route(c: Candidate, conf_min: Optional[float] = None, margin_min: Optional[float] = None) -> Optional[str]:
    """Return a fallback reason, or None when the candidate is reliable enough to keep as-is."""
    conf_min = config.CONF_MIN if conf_min is None else conf_min
    margin_min = config.MARGIN_MIN if margin_min is None else margin_min
    if c.conf < conf_min:
        return "LOW_CLASSIFICATION_CONFIDENCE"
    if c.margin < margin_min:
        return "LOW_CLASS_MARGIN"
    if c.label not in ("ACTION_ITEM", "DECISION"):
        return "LOW_CLASS_MARGIN"  # promoted from a near-miss: needs a second opinion
    if c.kind == "action" and not _ACTION_CUE.search(c.sentence):
        return "NO_ACTION_CUE"
    if c.kind == "decision" and not _DECISION_CUE.search(c.sentence):
        return "NO_DECISION_CUE"
    if c.kind == "action":
        if c.owner_flag in ("missing", "ambiguous", "multi"):
            return "OWNER_UNCERTAIN"
        if c.deadline_text and c.deadline_flag in ("vague", "hedged"):
            return "DEADLINE_AMBIGUOUS"
        if c.hedged:
            return "CONDITIONAL_COMMITMENT"
        owner_ok = (not c.owner) or supported_by(c.owner, c.context) or supported_by(c.owner, c.speaker or "")
        if not owner_ok or not supported_by(c.deadline_text, c.sentence):
            return "MISSING_EVIDENCE"
    elif c.hedged:
        return "CONDITIONAL_COMMITMENT"
    if validate_schema(c):
        return "SCHEMA_VALIDATION_FAILED"
    if c.conflict:
        return "CONFLICTING_EXTRACTIONS"
    return None
