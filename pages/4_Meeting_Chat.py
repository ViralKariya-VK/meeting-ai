import streamlit as st

from services import app_init, chat_service, db, limits, llm_service

app_init.init("Meeting Chat")
st.title("Meeting Chat")
st.caption("Answers come only from retrieved transcript passages and show their sources. If the transcript does not support an answer, the assistant says so.")

meetings = db.list_meetings()
names = {m["id"]: f"{m['title']} · {m['meeting_date']}" for m in meetings}
scope = st.multiselect("Ask about", list(names), format_func=lambda i: names[i], placeholder="All meetings")
use_llm = st.toggle("Use Groq LLM for answers", value=llm_service.is_configured(), disabled=not llm_service.is_configured())

if "chat_id" not in st.session_state:
    st.session_state.chat_id = chat_service.new_session(scope or None)
    st.session_state.chat = []

if st.button("New conversation"):
    st.session_state.chat_id = chat_service.new_session(scope or None)
    st.session_state.chat = []
    st.rerun()


def render(msg: dict) -> None:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("route"):
            st.caption(msg["route"])
        for c in msg.get("citations", []):
            with st.expander(f"Source: {c['title']} · {c['meeting_date']} · {c['time']}"):
                st.write(c["text"])
        if msg["role"] == "assistant" and msg.get("supported") is False:
            st.caption("No supporting evidence found: nothing was guessed.")


for msg in st.session_state.chat:
    render(msg)

if question := st.chat_input("e.g. What did we decide about the launch date?"):
    st.session_state.chat.append({"role": "user", "content": question})
    render(st.session_state.chat[-1])
    with st.spinner("Searching transcripts…"):
        try:
            res = chat_service.answer(question, scope or None, st.session_state.chat_id, use_llm=use_llm)
            msg = {"role": "assistant", "content": res["answer"], "citations": res["citations"],
                   "supported": res["supported"],
                   "route": f"Retrieval: {res['route'].mode} ({res['route'].reason})"}
        except limits.LimitReached as e:
            msg = {"role": "assistant", "content": str(e), "supported": None}
        except RuntimeError as e:
            msg = {"role": "assistant", "content": f"Search index problem: {e}", "supported": None}
    st.session_state.chat.append(msg)
    render(msg)
