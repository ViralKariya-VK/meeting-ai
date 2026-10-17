"""Headless smoke tests: every page renders without exceptions and key flows work (no network, no LLM)."""
import os
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent

PAGES = ["app.py", "pages/1_Upload_Meeting.py", "pages/2_Meeting_Insights.py", "pages/3_Meeting_Search.py",
         "pages/4_Meeting_Chat.py", "pages/5_Minutes_of_Meeting.py", "pages/6_Settings.py"]


@pytest.fixture(autouse=True)
def fresh_bootstrap(monkeypatch):
    monkeypatch.setenv("ALLOW_NO_AUTH", "1")
    monkeypatch.setenv("APP_PASSWORD", "")
    st.cache_resource.clear()  # force re-seeding after the per-test DB reset
    yield


def run(page, **kw):
    at = AppTest.from_file(str(ROOT / page), default_timeout=60)
    at.run()
    assert not at.exception, f"{page}: {[e.value for e in at.exception]}"
    return at


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page):
    run(page)


def test_login_gate_blocks_without_password(monkeypatch):
    monkeypatch.setenv("ALLOW_NO_AUTH", "0")
    monkeypatch.setenv("APP_PASSWORD", "secret-pass")
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    at.run()
    assert not at.exception
    assert any("Password" in (t.label or "") for t in at.text_input)
    assert not at.metric  # dashboard content not rendered
    at.text_input[0].set_value("wrong")
    at.button[0].click().run()
    assert at.error and not at.metric


def test_search_page_returns_hits():
    at = run("pages/3_Meeting_Search.py")
    at.text_input[0].set_value("vendor contract renewal").run()
    assert not at.exception
    assert any("mai-card" in m.value for m in at.markdown)


def test_chat_without_llm_shows_passages():
    at = run("pages/4_Meeting_Chat.py")
    at.chat_input[0].set_value("Who is handling the vendor contract?").run()
    assert not at.exception
    assert len(at.chat_message) >= 2


def test_mom_generate_and_edit():
    at = run("pages/5_Minutes_of_Meeting.py")
    at.button[0].click().run()
    assert not at.exception
    assert "Minutes of Meeting" in at.text_area[0].value


def test_upload_transcript_flow():
    at = run("pages/1_Upload_Meeting.py")
    at.text_input[0].set_value("UI smoke meeting")
    ta = [t for t in at.text_area][0]
    ta.set_value("[00:00:05] Sam: I'll send the budget sheet by Friday.\n[00:00:20] Lee: We have decided to keep the current pricing.")
    at.run()  # widget edits trigger a rerun in the browser, which enables the button
    [b for b in at.button if b.label == "Analyse transcript"][0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Processed" in s.value for s in at.success)
