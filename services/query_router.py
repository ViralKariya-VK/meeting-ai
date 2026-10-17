"""Routes a user question to SQL filters, FTS5, semantic or hybrid retrieval."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

from . import db, retrieval
from .extraction import MONTHS


@dataclass
class Route:
    mode: str                              # keyword | semantic | hybrid
    reason: str
    meeting_ids: Optional[list[int]] = None
    date_filter: Optional[str] = None      # human readable
    cleaned_query: str = ""
    notes: list = field(default_factory=list)


def _date_range(q: str, today: date) -> Optional[tuple[date, date, str]]:
    ql = q.lower()
    if "yesterday" in ql:
        d = today - timedelta(days=1)
        return d, d, "yesterday"
    if "today" in ql:
        return today, today, "today"
    if "last week" in ql:
        start = today - timedelta(days=today.weekday() + 7)
        return start, start + timedelta(days=6), "last week"
    if "last month" in ql:
        first = today.replace(day=1)
        end = first - timedelta(days=1)
        return end.replace(day=1), end, "last month"
    if "this month" in ql:
        return today.replace(day=1), today, "this month"
    m = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?(%s)\b" % "|".join(MONTHS), ql) or \
        re.search(r"\b(%s)\s+(\d{1,2})(?:st|nd|rd|th)?\b" % "|".join(MONTHS), ql)
    if m:
        a, b = m.groups()
        day, month = (int(a), MONTHS.index(b) + 1) if a.isdigit() else (int(b), MONTHS.index(a) + 1)
        for y in (today.year, today.year - 1):
            try:
                d = date(y, month, day)
            except ValueError:
                return None
            if d <= today + timedelta(days=1):
                return d, d, d.isoformat()
    return None


def plan(query: str, scope_ids: Optional[list[int]] = None, today: Optional[date] = None) -> Route:
    today = today or date.today()
    q = query.strip()
    ids = scope_ids
    notes: list[str] = []
    date_label = None
    dr = _date_range(q, today)
    if dr:
        lo, hi, date_label = dr
        pool = db.rows("SELECT id FROM meetings WHERE meeting_date BETWEEN ? AND ?", (lo.isoformat(), hi.isoformat()))
        in_range = [r["id"] for r in pool]
        ids = [i for i in in_range if (not scope_ids or i in scope_ids)]
        notes.append(f"date filter '{date_label}' matched {len(ids)} meeting(s)")
    has_quote = bool(re.search(r'"[^"]+"', q))
    short_exact = len(q.split()) <= 3 and not q.endswith("?")
    if has_quote or short_exact:
        mode, why = "keyword", "exact phrase / short keyword query"
    elif re.match(r"(?i)^(what|who|why|how|when|which|where|did|does|do|is|are|was|were|can|could)\b", q) or dr:
        mode, why = "hybrid", "question or date-filtered: combine keyword and semantic evidence"
    else:
        mode, why = "semantic", "free-text concept query"
    cleaned = q
    if date_label:
        cleaned = re.sub(r"(?i)\b(yesterday|today|last week|last month|this month|(\d{1,2}(st|nd|rd|th)?\s+(of\s+)?(%s))|((%s)\s+\d{1,2}(st|nd|rd|th)?))\b" % ("|".join(MONTHS), "|".join(MONTHS)), " ", q).strip() or q
    return Route(mode=mode, reason=why, meeting_ids=ids, date_filter=date_label, cleaned_query=cleaned, notes=notes)


def run(query: str, scope_ids: Optional[list[int]] = None, k: int = 6, mode: Optional[str] = None,
        today: Optional[date] = None) -> tuple[Route, list[dict]]:
    r = plan(query, scope_ids, today)
    if mode and mode != "auto":
        r.mode, r.reason = mode, f"user selected {mode}"
    if r.date_filter and r.meeting_ids == []:
        return r, []
    fn = {"keyword": retrieval.keyword_search, "semantic": retrieval.semantic_search,
          "hybrid": retrieval.hybrid_search}[r.mode]
    return r, fn(r.cleaned_query or query, r.meeting_ids, k=k)
