import pandas as pd
import streamlit as st

from services import app_init, db
from services.ui import badges

app_init.init("Dashboard")

st.title("Meeting AI")
st.caption("Hybrid NLP + agentic meeting intelligence · classical NLP first, LLM only when the evidence is uncertain.")
st.info(
    "**Demo only.** Storage is temporary and may reset when the app restarts. Uploaded audio goes to Groq for "
    "transcription: please use public or self-recorded, non-sensitive recordings.", icon=":material/info:")

meetings = db.list_meetings()
n_actions = db.one("SELECT COUNT(*) n FROM action_items")["n"]
n_decisions = db.one("SELECT COUNT(*) n FROM decisions")["n"]
n_review = db.one("SELECT (SELECT COUNT(*) FROM action_items WHERE status='needs_review') + "
                  "(SELECT COUNT(*) FROM decisions WHERE status='needs_review') n")["n"]
n_confirmed = db.one("SELECT (SELECT COUNT(*) FROM action_items WHERE status='confirmed') + "
                     "(SELECT COUNT(*) FROM decisions WHERE status='confirmed') n")["n"]

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Meetings", len(meetings))
c2.metric("Action items", n_actions)
c3.metric("Decisions", n_decisions)
c4.metric("Needs review", n_review)
c5.metric("Confirmed", n_confirmed)

left, right = st.columns([3, 2])
with left:
    st.subheader("Recent meetings")
    if meetings:
        df = pd.DataFrame([{
            "Title": m["title"], "Date": m["meeting_date"], "Source": m["source_type"], "Status": m["status"],
            "Duration (min)": round(m["duration_s"] / 60, 1) if m["duration_s"] else None} for m in meetings])
        st.dataframe(df, hide_index=True, width="stretch")
    else:
        st.write("No meetings yet.")
    st.page_link("pages/1_Upload_Meeting.py", label="Upload a meeting", icon=":material/upload:")

with right:
    st.subheader("Needs your review")
    rows = db.rows(
        "SELECT a.id, a.task, a.owner, a.fallback_reason, m.title FROM action_items a "
        "JOIN meetings m ON m.id=a.meeting_id WHERE a.status='needs_review' ORDER BY a.id DESC LIMIT 8")
    if not rows:
        st.success("Nothing waiting for review.")
    for r in rows:
        st.markdown(
            f"<div class='mai-card'><b>{r['task'][:90]}</b><br><small>{r['title']} · owner: {r['owner'] or 'unassigned'}"
            f"</small><br>{badges('needs_review')}<small>{(r['fallback_reason'] or '').replace('_', ' ').lower()}</small></div>",
            unsafe_allow_html=True)
    st.page_link("pages/2_Meeting_Insights.py", label="Open Meeting Insights", icon=":material/insights:")
