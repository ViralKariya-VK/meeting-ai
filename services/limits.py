"""Per-session usage caps so the shared free Groq quota is not exhausted."""
from __future__ import annotations

import streamlit as st

from . import config


class LimitReached(Exception):
    pass


def _bump(key: str, cap: int, label: str, n: int = 1) -> None:
    used = st.session_state.get(key, 0)
    if used + n > cap:
        raise LimitReached(f"Demo limit reached for this session ({label}: {cap}). Reload the page to reset.")
    st.session_state[key] = used + n


def use_transcription() -> None:
    _bump("n_transcriptions", config.SESSION_MAX_TRANSCRIPTIONS, "transcriptions")


def use_llm_call(n: int = 1) -> None:
    _bump("n_llm_calls", config.SESSION_MAX_LLM_CALLS, "LLM calls", n)
