"""Rule + NER extraction of owner / task / deadline candidates from classified sentences.
Output is always a *candidate*: ambiguity is flagged rather than guessed."""
from __future__ import annotations

import calendar
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

from . import config

_nlp = None
_nlp_failed = False

WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
          "november", "december"]
_NOT_NAMES = {w.capitalize() for w in WEEKDAYS + MONTHS} | {
    "I", "We", "You", "They", "He", "She", "It", "The", "This", "That", "Okay", "Ok", "So", "Alright", "Right", "Well",
    "Yeah", "Please", "Action", "Next", "Let", "Let's", "Can", "Could", "Someone", "Everyone", "Team", "Thanks",
    "Item", "Budget", "Option", "Plan", "Final", "Approved", "Great", "And", "But", "Also", "Then", "Maybe", "Today",
    "Tomorrow", "Anyone", "Nobody", "Okay,", "Wait", "Honestly"}
_HEDGE = re.compile(r"\b(unless|if|maybe|probably|might|perhaps|hopefully|try to|aim(?:ing)? for|should probably|"
                    r"depending|assuming|provided that)\b", re.I)
_VAGUE = re.compile(r"\b(soon|asap|as soon as possible|later|sometime|eventually|when (?:we|i|you) can|"
                    r"before the next (?:meeting|call|sprint)|over the weekend|this week|next week|this month|"
                    r"next month|this quarter)\b", re.I)

_DL_PATTERNS = [
    re.compile(r"\b(?:by|before|on|until|due|for)?\s*(?:the )?(?:next |this |coming )?"
               r"(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)(?: (?:morning|afternoon|evening))?\b", re.I),
    re.compile(r"\b(?:by )?tomorrow(?: (?:morning|afternoon|evening))?\b", re.I),
    re.compile(r"\b(?:by )?(?:end of (?:the )?(?:day|week|month|quarter)|eod|today)\b", re.I),
    re.compile(r"\bby the \d{1,2}(?:st|nd|rd|th)\b", re.I),
    re.compile(r"\b(?:by |on |before )?\d{1,2}(?:st|nd|rd|th)? (?:of )?(?:%s)\b" % "|".join(MONTHS), re.I),
    re.compile(r"\b(?:by |on |before )?(?:%s) \d{1,2}(?:st|nd|rd|th)?\b" % "|".join(MONTHS), re.I),
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
    _VAGUE,
]
_MODAL = re.compile(
    r"(?:\bwill\b|\bis going to\b|\bare going to\b|\b(?:i|we|she|he|they)'ll\b|\bhas to\b|\bneeds? to\b|\bagreed to\b|"
    r"\bpromised to\b|\bis responsible for\b|\bto\b|\bplease\b|\bcan you\b|\bcould you(?: please)?\b|"
    r"\bwould you\b|\bgoing to\b|\bhave\b)\s+", re.I)
_FIRST_PERSON = re.compile(r"\b(i'll|i will|i can|i'm going to|i am going to|i'd take|i'll take)\b", re.I)
_UNOWNED = re.compile(r"\b(someone|somebody|anyone|we should|we need to|we have to|we must|let's|everyone|the team)\b", re.I)
_FILLER_START = re.compile(r"^(?:okay|ok|so|alright|right|well|yeah)[,.]?\s+", re.I)


@dataclass
class Candidate:
    kind: str                       # action | decision
    sentence: str
    seg_idx: int
    speaker: Optional[str]
    label: str
    conf: float
    margin: float
    probs: dict
    context: str = ""
    evidence_idx: list = field(default_factory=list)
    owner: Optional[str] = None
    owner_flag: str = "n/a"         # ok | missing | ambiguous | multi | n/a
    task: Optional[str] = None
    deadline_text: Optional[str] = None
    deadline_date: Optional[date] = None
    deadline_flag: str = "none"     # none | ok | vague | hedged
    hedged: bool = False
    conflict: bool = False
    fallback_reason: Optional[str] = None
    source: str = "classical"
    status: str = "suggested"
    evidence_quote: str = ""
    confidence: float = 0.0
    notes: list = field(default_factory=list)


# ---- NER ------------------------------------------------------------------

def get_nlp():
    global _nlp, _nlp_failed
    if _nlp is None and not _nlp_failed:
        try:
            import spacy

            _nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer", "attribute_ruler", "tagger", "senter"])
        except Exception:
            _nlp_failed = True
    return _nlp


def find_persons(text: str) -> list[str]:
    nlp = get_nlp()
    names: list[str] = []
    if nlp is not None:
        names = [e.text.strip() for e in nlp(text).ents if e.label_ == "PERSON"]
    # heuristic backup: capitalised tokens that are not sentence starters/stopwords
    for m in re.finditer(r"(?<![.!?]\s)(?<!^)\b([A-Z][a-z]{2,})\b", text):
        w = m.group(1)
        if w not in _NOT_NAMES and w not in names:
            names.append(w)
    # names at the very start followed by modal/comma ("Rahul will ...", "Priya, can you ...")
    m = re.match(r"^(?:%s)?([A-Z][a-z]{2,})(?:,| will| is| has| agreed| said| promised| and)" % _FILLER_START.pattern, text)
    if m and m.group(1) not in _NOT_NAMES and m.group(1) not in names:
        names.insert(0, m.group(1))
    # keep only tokens that are single clean names or "A B"; drop weekday/month words
    clean = []
    for n in names:
        if n.lower() in WEEKDAYS or n.lower() in MONTHS or n in _NOT_NAMES:
            continue
        if n not in clean:
            clean.append(n)
    return clean


# ---- deadlines ------------------------------------------------------------

def find_deadline(sentence: str) -> Optional[str]:
    for pat in _DL_PATTERNS:
        m = pat.search(sentence)
        if m:
            return m.group(0).strip(" ,.")
    nlp = get_nlp()
    if nlp is not None:
        for e in nlp(sentence).ents:
            if e.label_ == "DATE":
                return e.text.strip()
    return None


def _next_weekday(ref: date, weekday: int, force_next_week: bool) -> date:
    if force_next_week:
        monday_next = ref + timedelta(days=7 - ref.weekday())
        return monday_next + timedelta(days=weekday)
    delta = (weekday - ref.weekday()) % 7
    return ref + timedelta(days=delta or 7)


def normalize_deadline(text: Optional[str], meeting_date: date) -> tuple[Optional[date], str]:
    """Return (date or None, flag) where flag is none | ok | vague. Never guesses beyond the rules."""
    if not text:
        return None, "none"
    t = text.lower().strip()
    iso = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", t)
    if iso:
        try:
            return date(int(iso.group(1)), int(iso.group(2)), int(iso.group(3))), "ok"
        except ValueError:
            return None, "vague"
    if re.search(r"\btomorrow\b", t):
        return meeting_date + timedelta(days=1), "ok"
    if re.search(r"\b(today|eod|end of (the )?day)\b", t):
        return meeting_date, "ok"
    if re.search(r"end of (the )?week", t):
        return meeting_date + timedelta(days=(4 - meeting_date.weekday()) % 7), "ok"
    if re.search(r"end of (the )?month", t):
        return date(meeting_date.year, meeting_date.month, calendar.monthrange(meeting_date.year, meeting_date.month)[1]), "ok"
    wd = next((i for i, w in enumerate(WEEKDAYS) if re.search(rf"\b{w}\b", t)), None)
    if wd is not None and not _VAGUE.search(t):
        return _next_weekday(meeting_date, wd, force_next_week=bool(re.search(r"\bnext\b", t))), "ok"
    m = re.search(r"\bthe (\d{1,2})(?:st|nd|rd|th)\b", t)
    if m:
        day = int(m.group(1))
        y, mo = meeting_date.year, meeting_date.month
        for _ in range(2):
            try:
                d = date(y, mo, day)
                if d >= meeting_date:
                    return d, "ok"
            except ValueError:
                pass
            mo, y = (1, y + 1) if mo == 12 else (mo + 1, y)
        return None, "vague"
    m = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)? (?:of )?(%s)\b" % "|".join(MONTHS), t) or \
        re.search(r"\b(%s) (\d{1,2})(?:st|nd|rd|th)?\b" % "|".join(MONTHS), t)
    if m:
        a, b = m.groups()
        day, month = (int(a), MONTHS.index(b) + 1) if a.isdigit() else (int(b), MONTHS.index(a) + 1)
        for y in (meeting_date.year, meeting_date.year + 1):
            try:
                d = date(y, month, day)
            except ValueError:
                return None, "vague"
            if d >= meeting_date:
                return d, "ok"
    return None, "vague"


# ---- extraction -----------------------------------------------------------

def _strip_filler(s: str) -> str:
    return _FILLER_START.sub("", s.strip())


def _speaker_is_name(speaker: Optional[str]) -> bool:
    return bool(speaker) and not re.match(r"(?i)^(speaker|unknown|spk|host|guest)[ _-]?\d*$", speaker or "")


def extract_task(sentence: str, deadline_text: Optional[str]) -> str:
    s = _strip_filler(sentence)
    if deadline_text:
        s = re.sub(re.escape(deadline_text), "", s, flags=re.I)
    s = re.sub(r"^action item:?\s*|^action for \w+:\s*|^next step:\s*", "", s, flags=re.I)
    m = _MODAL.search(s)
    if m:
        rest = re.sub(r"^(?:take this one and|follow up and)\s+", "", s[m.end():], flags=re.I)
        s = rest or s
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s+,", ",", s)
    s = re.sub(r"[\s,;:]+$", "", s).rstrip(".?!").strip()
    s = re.sub(r",\s*(?:[A-Z][a-z]+)$", "", s)  # trailing vocative ("..., Rahul")
    return (s[:160] or sentence[:160]).strip()


def extract_owner(sentence: str, speaker: Optional[str]) -> tuple[Optional[str], str]:
    s = _strip_filler(sentence)
    persons = find_persons(s)
    if _FIRST_PERSON.search(s) and not re.search(r"\bI'd like you\b", s, re.I):
        return (speaker, "ok") if _speaker_is_name(speaker) else (None, "ambiguous")
    m = re.match(r"^action (?:item )?(?:for|:)\s*([A-Z][a-z]+)", s, re.I)
    if m:
        return m.group(1).capitalize(), "ok"
    m = re.match(r"^([A-Z][a-z]{2,}),\s*(?:please|can|could|would|i'd)", s)
    if m and m.group(1) not in _NOT_NAMES:
        return m.group(1), "ok"
    m = re.search(r"(?:can|could|would) you\b.*?,\s*([A-Z][a-z]{2,})\s*\??$", s)
    if m:
        return m.group(1), "ok"
    m = re.match(r"^([A-Z][a-z]{2,}) and ([A-Z][a-z]{2,}) (?:will|are|have|need)", s)
    if m:
        return f"{m.group(1)} and {m.group(2)}", "multi"
    m = re.search(r"\b(?:let's have|we need|i'd like|asked|get)\s+([A-Z][a-z]{2,})\b", s)
    if m and m.group(1) not in _NOT_NAMES:
        return m.group(1), "ok"
    m = re.match(r"^([A-Z][a-z]{2,})\s+(?:will|is going to|has to|needs to|agreed to|promised to|is responsible|to |said)", s)
    if m and m.group(1) not in _NOT_NAMES:
        return m.group(1), "ok" if len(persons) <= 1 else "ok"
    if len(persons) == 1 and not _UNOWNED.search(s):
        return persons[0], "ok"
    if len(persons) > 1:
        return None, "ambiguous"
    return None, "missing"


def build_candidate(sent: dict, pred: dict, kind: str, context: str, meeting_date: date) -> Candidate:
    text = sent["text"]
    c = Candidate(kind=kind, sentence=text, seg_idx=sent["seg_idx"], speaker=sent.get("speaker"),
                  label=pred["label"], conf=pred["conf"], margin=pred["margin"], probs=pred["probs"],
                  context=context, evidence_idx=[sent["seg_idx"]], evidence_quote=text,
                  confidence=round(pred["conf"], 4))
    c.hedged = bool(_HEDGE.search(text))
    if kind == "decision":
        c.task = _strip_filler(text).rstrip()
        return c
    c.deadline_text = find_deadline(text)
    c.deadline_date, c.deadline_flag = normalize_deadline(c.deadline_text, meeting_date)
    if c.deadline_flag == "ok" and c.hedged:
        c.deadline_flag = "hedged"
    c.owner, c.owner_flag = extract_owner(text, sent.get("speaker"))
    c.task = extract_task(text, c.deadline_text)
    return c


def _tokens(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]{3,}", (s or "").lower())}


_STOP = {"them", "those", "these", "this", "that", "the", "with", "and", "everyone", "also", "from", "for", "our", "all"}


def _content_tokens(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]{3,}", (s or "").lower()) if w not in _STOP}


def dedupe(cands: list[Candidate]) -> list[Candidate]:
    """Merge a request and its acknowledgement ("Can you fix X by Friday, Mark?" / "I'll fix X by Friday")."""
    out: list[Candidate] = []
    for c in cands:
        merged = False
        if c.kind == "action" and c.owner:
            for prev in reversed(out[-6:]):
                if prev.kind != "action" or (prev.owner or "").lower() != c.owner.lower():
                    continue
                if abs(c.seg_idx - prev.seg_idx) > 4:
                    continue
                a, b = _content_tokens(prev.task), _content_tokens(c.task)
                if not a or not b or len(a & b) / min(len(a), len(b)) < 0.6:
                    continue
                if prev.deadline_text and c.deadline_text and \
                        (prev.deadline_date, prev.deadline_text.lower()) != (c.deadline_date, c.deadline_text.lower()):
                    continue  # different deadlines: leave both so detect_conflicts can flag them
                prev.evidence_idx = sorted(set(prev.evidence_idx) | set(c.evidence_idx))
                if not prev.deadline_text and c.deadline_text:
                    prev.deadline_text, prev.deadline_date, prev.deadline_flag = c.deadline_text, c.deadline_date, c.deadline_flag
                if len(c.task or "") > len(prev.task or ""):
                    prev.task = c.task
                prev.evidence_quote = f"{prev.evidence_quote} / {c.evidence_quote}"
                prev.context = prev.context if c.sentence in prev.context else prev.context + "\n" + c.context
                merged = True
                break
        if not merged:
            out.append(c)
    return out


def detect_conflicts(cands: list[Candidate]) -> None:
    """Two action candidates about (nearly) the same task with different deadlines are flagged."""
    acts = [c for c in cands if c.kind == "action" and c.deadline_text]
    for i, a in enumerate(acts):
        for b in acts[i + 1:]:
            ta, tb = _tokens(a.task), _tokens(b.task)
            if not ta or not tb:
                continue
            jac = len(ta & tb) / len(ta | tb)
            same_owner = (a.owner or "").lower() == (b.owner or "").lower()
            diff_dl = (a.deadline_date, a.deadline_text.lower()) != (b.deadline_date, b.deadline_text.lower())
            if jac >= 0.5 and same_owner and diff_dl and a.seg_idx != b.seg_idx:
                a.conflict = b.conflict = True
