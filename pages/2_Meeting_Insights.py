import json

import pandas as pd
import plotly.express as px
import streamlit as st

from services import app_init, db, llm_service, pipeline
from services.ui import badges, esc, pick_meeting, ts

app_init.init("Meeting Insights")
st.title("Meeting Insights")

mid = pick_meeting("insights_meeting")
if mid is None:
    st.stop()
m = db.get_meeting(mid)
segments = db.get_segments(mid)

h1, h2 = st.columns([4, 1])
h1.caption(f"{m['meeting_date']} · {m['source_type']} · status: {m['status']}")
if h2.button("Re-run analysis", help="Replaces the extracted items for this meeting (your edits are lost)."):
    with st.status("Re-analysing", expanded=True) as box:
        pipeline.analyze_meeting(mid, use_llm=llm_service.is_configured(), progress=lambda x: box.write(x))
    st.rerun()

tab_sum, tab_act, tab_dec, tab_tr, tab_pipe = st.tabs(["Summary", "Action items", "Decisions", "Transcript", "Pipeline"])

with tab_sum:
    s = db.one("SELECT text, method FROM summaries WHERE meeting_id=? ORDER BY id DESC", (mid,))
    if s:
        st.markdown(s["text"])
        st.caption("LLM-written summary" if s["method"] == "llm" else
                   "Extractive summary (most central sentences). Enable the LLM on upload for an abstractive one.")
    else:
        st.info("No summary yet.")


def _evidence(idx_json: str, quote: str) -> str:
    idx = db.json_loads(idx_json, [])
    when = ""
    if idx:
        seg = next((x for x in segments if x["idx"] == idx[0]), None)
        if seg and seg.get("start_s") is not None:
            when = f"[{ts(seg['start_s'])}] "
    return f"<div class='mai-ev'>{when}“{esc(quote)}”</div>"


def _item_editor(kind: str, row) -> None:
    table = "action_items" if kind == "action" else "decisions"
    title = row["task"] if kind == "action" else row["text"]
    meta = ""
    if kind == "action":
        due = row["deadline_date"] or (f"{row['deadline_text']} (unresolved)" if row["deadline_text"] else "no deadline")
        meta = f"<br><small>Owner: <b>{esc(row['owner']) or 'unassigned'}</b> · Due: <b>{esc(due)}</b></small>"
    reason = f"<small> · routed because: {row['fallback_reason'].replace('_', ' ').lower()}</small>" if row["fallback_reason"] else ""
    st.markdown(
        f"<div class='mai-card'><b>{esc(title)}</b>{meta}<br>{badges(row['status'], row['source'])}"
        f"<small>confidence {row['confidence']:.2f}</small>{reason}{_evidence(row['evidence_idx'], row['evidence_quote'])}</div>",
        unsafe_allow_html=True)
    with st.expander("Review / edit", expanded=False):
        with st.form(f"{table}_{row['id']}"):
            new_title = st.text_input("Task" if kind == "action" else "Decision", value=title)
            if kind == "action":
                a, b = st.columns(2)
                owner = a.text_input("Owner", value=row["owner"] or "")
                deadline = b.text_input("Deadline date (YYYY-MM-DD)", value=row["deadline_date"] or "")
            status = st.radio("Status", ["suggested", "needs_review", "confirmed", "rejected"],
                              index=["suggested", "needs_review", "confirmed", "rejected"].index(row["status"]),
                              horizontal=True)
            if st.form_submit_button("Save"):
                if kind == "action":
                    from datetime import date
                    try:
                        dl = date.fromisoformat(deadline).isoformat() if deadline.strip() else None
                    except ValueError:
                        st.error("Deadline must look like 2026-10-31.")
                        st.stop()
                    db.execute("UPDATE action_items SET task=?, owner=?, deadline_date=?, status=? WHERE id=?",
                               (new_title.strip(), owner.strip() or None, dl, status, row["id"]))
                else:
                    db.execute("UPDATE decisions SET text=?, status=? WHERE id=?", (new_title.strip(), status, row["id"]))
                db.audit(f"{kind}_reviewed", mid, f"id={row['id']} -> {status}")
                st.rerun()


with tab_act:
    items = db.rows("SELECT * FROM action_items WHERE meeting_id=? ORDER BY CASE status WHEN 'needs_review' THEN 0 "
                    "WHEN 'suggested' THEN 1 WHEN 'confirmed' THEN 2 ELSE 3 END, id", (mid,))
    if not items:
        st.info("No action items found.")
    st.caption("Everything extracted is a *candidate* until you confirm it. Owners and deadlines are left empty when the transcript does not support them.")
    for r in items:
        _item_editor("action", r)

with tab_dec:
    items = db.rows("SELECT * FROM decisions WHERE meeting_id=? ORDER BY CASE status WHEN 'needs_review' THEN 0 "
                    "WHEN 'suggested' THEN 1 WHEN 'confirmed' THEN 2 ELSE 3 END, id", (mid,))
    if not items:
        st.info("No decisions found.")
    for r in items:
        _item_editor("decision", r)

with tab_tr:
    evidence = set()
    for t in ("action_items", "decisions"):
        for r in db.rows(f"SELECT evidence_idx FROM {t} WHERE meeting_id=?", (mid,)):
            evidence.update(db.json_loads(r["evidence_idx"], []))
    st.caption("Highlighted lines are evidence for extracted items.")
    out = []
    for s in segments:
        sp = f"<b>{esc(s['speaker'])}:</b> " if s.get("speaker") else ""
        line = f"<span style='opacity:.6'>{ts(s['start_s'])}</span> {sp}{esc(s['text'])}"
        out.append(f"<div style='padding:3px 6px;{'background:rgba(255,229,143,.35);border-radius:6px;' if s['idx'] in evidence else ''}'>{line}</div>")
    st.markdown("".join(out), unsafe_allow_html=True)

with tab_pipe:
    run = db.one("SELECT * FROM analysis_runs WHERE meeting_id=? ORDER BY id DESC", (mid,))
    if not run:
        st.info("No analysis run recorded.")
    else:
        stats = json.loads(run["stats_json"])
        cfg = json.loads(run["config_json"])
        c = st.columns(4)
        c[0].metric("Sentences", stats["sentences"])
        c[1].metric("Candidates", stats["candidates"])
        c[2].metric("Resolved by classical", stats["resolved_by_classical"])
        c[3].metric("LLM calls", stats["llm_calls"])
        rate = (stats["candidates"] - stats["resolved_by_classical"]) / stats["candidates"] if stats["candidates"] else 0
        st.write(f"Fallback rate: **{rate:.0%}** · LLM failures: {stats['llm_failures']} · "
                 f"rejected by LLM: {stats['llm_rejected']} · summary: {stats['summary_method']} · {stats['elapsed_s']}s")
        st.caption(f"Thresholds: confidence ≥ {cfg['conf_min']}, margin ≥ {cfg['margin_min']} · "
                   f"LLM: {cfg['llm_model'] or 'not used'}")
        for err in stats.get("llm_errors", []):
            st.error(f"LLM error: {err}")
        fr = stats.get("fallback_reasons", {})
        if fr:
            df = pd.DataFrame({"reason": list(fr), "count": list(fr.values())})
            st.plotly_chart(px.bar(df, x="reason", y="count", title="Why candidates were routed away from classical NLP"),
                            width="stretch")
