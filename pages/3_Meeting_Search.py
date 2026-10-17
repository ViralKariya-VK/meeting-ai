import streamlit as st

from services import app_init, db, query_router
from services.ui import highlight, ts

app_init.init("Meeting Search")
st.title("Meeting Search")
st.caption("Exact phrases use SQLite FTS5, concept queries use semantic embeddings, and questions combine both (reciprocal rank fusion).")

meetings = db.list_meetings()
names = {m["id"]: f"{m['title']} · {m['meeting_date']}" for m in meetings}
q = st.text_input("Search", placeholder='"release approval"  or  what did the team say about launch delays?')
c1, c2, c3 = st.columns([3, 2, 1])
scope = c1.multiselect("Limit to meetings", list(names), format_func=lambda i: names[i], placeholder="All meetings")
mode = c2.selectbox("Mode", ["auto", "keyword", "semantic", "hybrid"])
k = c3.number_input("Results", 1, 20, 6)

if q.strip():
    try:
        route, hits = query_router.run(q, scope or None, k=int(k), mode=mode)
    except RuntimeError as e:
        st.error(str(e))
        st.stop()
    st.caption(f"Route: **{route.mode}** — {route.reason}" + (f" · {'; '.join(route.notes)}" if route.notes else ""))
    if not hits:
        st.warning("No matching passages.")
    for h in hits:
        st.markdown(
            f"<div class='mai-card'><b>{h['title']}</b> · {h['meeting_date']} · {ts(h['start_s'])} "
            f"<span class='mai-badge mai-cls'>{h['via']}</span><small>score {h['score']:.3f}</small>"
            f"<div class='mai-hit' style='margin-top:6px'>{highlight(h['text'][:700], q)}"
            f"{'…' if len(h['text']) > 700 else ''}</div></div>", unsafe_allow_html=True)
