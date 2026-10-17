import pandas as pd
import plotly.express as px
import streamlit as st

from services import app_init, classical_nlp, config, db, embedding_service, llm_service, retrieval
from services.ui import pick_meeting

app_init.init("Settings")
st.title("Settings & Evaluation")

t_cfg, t_eval, t_data, t_priv = st.tabs(["Configuration", "Classifier evaluation", "Data", "Privacy"])

with t_cfg:
    st.caption("Values come from environment variables / host secrets. Secrets are never displayed.")
    rows = [
        ("Groq key configured", "yes" if llm_service.is_configured() else "no"),
        ("ASR model", config.ASR_MODEL()), ("LLM model", config.LLM_MODEL()),
        ("Embedding backend", embedding_service.backend()),
        ("Confidence threshold", config.CONF_MIN), ("Class margin threshold", config.MARGIN_MIN),
        ("Max media length (min)", config.MAX_AUDIO_MINUTES), ("Max upload (MB)", config.MAX_UPLOAD_MB),
        ("Chunk size (s) / overlap (s)", f"{config.CHUNK_SECONDS} / {config.CHUNK_OVERLAP_SECONDS}"),
        ("Per-session transcriptions / LLM calls", f"{config.SESSION_MAX_TRANSCRIPTIONS} / {config.SESSION_MAX_LLM_CALLS}"),
        ("Retrieval chunk words / overlap", f"{config.CHUNK_WORDS} / {config.CHUNK_OVERLAP_WORDS}"),
    ]
    st.dataframe(pd.DataFrame(rows, columns=["Setting", "Value"]), hide_index=True, width="stretch")
    st.caption(f"Session usage: {st.session_state.get('n_transcriptions', 0)} transcription(s), "
               f"{st.session_state.get('n_llm_calls', 0)} LLM call(s).")

with t_eval:
    metrics = classical_nlp.load_metrics()
    if not metrics:
        st.info("No metrics file yet. It is created when the classifier is trained.")
    else:
        st.warning("The classifier is trained on **synthetic, template-generated** sentences, so the hold-out score is "
                   "optimistic. The hand-written test set (48 sentences, never trained on) is more informative but was "
                   "written by the same author as the templates, so real meetings will likely score lower. This is exactly "
                   "why the confidence gateway and LLM fallback exist.")
        for name, label in (("hand_written_test", "Hand-written test set"), ("synthetic_holdout", "Synthetic hold-out")):
            d = metrics.get(name)
            if not d:
                continue
            st.subheader(f"{label} (n={d['n']})")
            c = st.columns(4)
            c[0].metric("Accuracy", f"{d['accuracy']:.1%}")
            c[1].metric("Macro F1", f"{d['macro_f1']:.3f}")
            c[2].metric("Brier score", f"{d['brier']:.3f}", help="Lower is better; measures probability calibration.")
            c[3].metric("Would fall back", f"{d['fallback_rate_at_thresholds']:.0%}")
            cm = d["confusion_matrix"]
            fig = px.imshow(cm["matrix"], x=cm["labels"], y=cm["labels"], text_auto=True, aspect="auto",
                            labels=dict(x="Predicted", y="Actual"))
            st.plotly_chart(fig, width="stretch", key=f"cm_{name}")
            if d.get("accuracy_when_confident") is not None:
                st.caption(f"Accuracy on confidently-classified sentences: {d['accuracy_when_confident']:.1%} · "
                           f"on sentences routed to fallback: "
                           f"{(d['accuracy_when_routed'] if d['accuracy_when_routed'] is not None else float('nan')):.1%}")

with t_data:
    st.subheader("Search index")
    st.write("The vector index is derived from the stored transcripts and can be rebuilt at any time.")
    if st.button("Rebuild search index"):
        n = retrieval.rebuild_all()
        st.success(f"Indexed {n} chunks.")
    st.subheader("Delete a meeting")
    st.caption("Removes the transcript, extracted items, chat-independent drafts, search vectors and any stored media.")
    mid = pick_meeting("delete_meeting", "Meeting to delete")
    if mid is not None:
        sure = st.checkbox("Yes, permanently delete this meeting and its derived data")
        if st.button("Delete meeting", disabled=not sure):
            db.delete_meeting(mid)
            st.success("Deleted.")
            st.rerun()
    st.subheader("Audit log (latest 30)")
    log = db.rows("SELECT ts, event, meeting_id, detail FROM audit_logs ORDER BY id DESC LIMIT 30")
    st.dataframe(pd.DataFrame([dict(r) for r in log]), hide_index=True, width="stretch")

with t_priv:
    st.markdown(
        """
**Demo notice**

- Audio you upload is converted on this server and sent to **Groq** for transcription; the file is deleted after processing.
- Transcript snippets may be sent to Groq's LLM when the classifier is unsure, and for chat, summaries and fallback.
- Storage on this host is **temporary**. It may reset at any time. Do not upload confidential or personal recordings.
- Only get consent from participants for recordings you upload.
- Extracted tasks, owners and deadlines are **suggestions** until you confirm them.
- The app never sends email or edits a calendar for you.
        """)
