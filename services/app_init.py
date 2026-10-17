"""One-time per-process/session bootstrap used by every Streamlit page."""
from __future__ import annotations

import streamlit as st

from . import auth, config, db, limits, llm_service, seed

CSS = """
<style>
.block-container {padding-top: 2rem; max-width: 1200px;}
.mai-badge {display:inline-block; padding:2px 10px; border-radius:999px; font-size:0.75rem; font-weight:600; margin-right:6px;}
.mai-ok {background:#e3f6e8; color:#14632b;} .mai-review {background:#fff1d6; color:#8a5a00;}
.mai-llm {background:#e7edff; color:#27408b;} .mai-cls {background:#eef0f3; color:#3b4350;}
.mai-rej {background:#fde7e7; color:#8f1d1d;}
.mai-card {border:1px solid rgba(128,128,128,.25); border-radius:12px; padding:14px 16px; margin-bottom:10px;}
.mai-ev {font-size:0.85rem; opacity:.75; border-left:3px solid rgba(128,128,128,.4); padding-left:10px; margin-top:6px;}
.mai-hit mark {background:#ffe58f;}
</style>
"""

BADGE = {
    "confirmed": '<span class="mai-badge mai-ok">confirmed</span>',
    "suggested": '<span class="mai-badge mai-cls">suggested</span>',
    "needs_review": '<span class="mai-badge mai-review">needs review</span>',
    "rejected": '<span class="mai-badge mai-rej">rejected</span>',
    "llm": '<span class="mai-badge mai-llm">LLM-assisted</span>',
    "classical": '<span class="mai-badge mai-cls">classical NLP</span>',
}


@st.cache_resource(show_spinner="Preparing demo data (first run only)...")
def _bootstrap() -> bool:
    config.ensure_dirs()
    db.init_db()
    seed.seed_if_empty(use_llm=False)
    return True


def init(page_title: str = "Meeting AI", icon: str = ":material/groups:", wide: bool = True) -> None:
    st.set_page_config(page_title=f"{page_title} · Meeting AI", page_icon=icon, layout="wide" if wide else "centered")
    st.markdown(CSS, unsafe_allow_html=True)
    auth.require_login()
    _bootstrap()
    llm_service.set_budget_hook(limits.use_llm_call)
    with st.sidebar:
        st.caption("DEMO ONLY · storage is temporary · do not upload sensitive recordings")
        if not llm_service.is_configured():
            st.warning("GROQ_API_KEY not set: transcription and LLM features are off (classical NLP still works).")
        else:
            st.success(f"Groq connected · {config.LLM_MODEL()}")
