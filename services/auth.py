"""Shared-password gate for the demo (protects the shared Groq key)."""
from __future__ import annotations

import hmac
import time

import streamlit as st

from . import config


def password_configured() -> bool:
    return bool(config.get_secret("APP_PASSWORD"))


def check_password(candidate: str) -> bool:
    expected = config.get_secret("APP_PASSWORD") or ""
    return bool(expected) and hmac.compare_digest(candidate.encode(), expected.encode())


def require_login() -> None:
    """Call at the top of every page. Fails closed if APP_PASSWORD is missing
    (set ALLOW_NO_AUTH=1 for local development only)."""
    if st.session_state.get("authed"):
        return
    if not password_configured():
        if config.get_secret("ALLOW_NO_AUTH") == "1":
            st.session_state["authed"] = True
            return
        st.error("APP_PASSWORD is not configured. Set it in the host's secrets (or .env for local use).")
        st.stop()

    st.title("Meeting AI")
    st.caption("Demo access. Enter the shared password you were given.")
    with st.form("login"):
        pw = st.text_input("Password", type="password")
        ok = st.form_submit_button("Enter")
    if ok:
        fails = st.session_state.get("login_fails", 0)
        if fails:
            time.sleep(min(2 ** fails, 8))  # simple brute-force slow-down
        if check_password(pw):
            st.session_state["authed"] = True
            st.session_state["login_fails"] = 0
            st.rerun()
        st.session_state["login_fails"] = fails + 1
        st.error("Incorrect password.")
    st.stop()
