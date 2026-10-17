"""Small shared Streamlit helpers."""
from __future__ import annotations

import html
import re
from typing import Optional

import streamlit as st

from . import db
from .app_init import BADGE
from .preprocessing import fmt_ts


def pick_meeting(key: str, label: str = "Meeting", allow_none: bool = False) -> Optional[int]:
    meetings = db.list_meetings()
    if not meetings:
        st.info("No meetings yet. Add one on the Upload Meeting page.")
        return None
    ids = [m["id"] for m in meetings]
    names = {m["id"]: f"{m['title']} · {m['meeting_date']}" for m in meetings}
    options = ([None] if allow_none else []) + ids
    return st.selectbox(label, options, key=key, format_func=lambda i: "All meetings" if i is None else names[i])


def badges(*keys: str) -> str:
    return "".join(BADGE.get(k, "") for k in keys)


def esc(s: Optional[str]) -> str:
    return html.escape(s or "")


def highlight(text: str, query: str) -> str:
    out = esc(text)
    for term in {t for t in re.findall(r"[A-Za-z0-9']{3,}", query)}:
        out = re.sub(f"(?i)({re.escape(esc(term))})", r"<mark>\1</mark>", out)
    return out


def ts(seconds: Optional[float]) -> str:
    return fmt_ts(seconds) if seconds is not None else "--:--:--"


def explain_error(e: Exception) -> str:
    return f"{type(e).__name__}: {e}"
