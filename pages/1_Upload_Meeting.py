import uuid
from datetime import date
from pathlib import Path

import streamlit as st

from services import app_init, config, db, limits, llm_service, media, pipeline, transcription
from services.preprocessing import parse_transcript
from services.ui import explain_error

app_init.init("Upload Meeting")
st.title("Upload Meeting")
st.caption(f"Demo limits: media up to {config.MAX_AUDIO_MINUTES} min / {config.MAX_UPLOAD_MB} MB. "
           "Audio is transcribed by Groq Whisper and the uploaded file is deleted right after processing.")

c1, c2, c3 = st.columns([3, 2, 2])
title = c1.text_input("Title", placeholder="e.g. Sprint planning")
mdate = c2.date_input("Meeting date", value=date.today(), help="Used to resolve words like 'Friday' or 'tomorrow'.")
tz = c3.text_input("Timezone", value="UTC")
use_llm = st.checkbox("Use LLM fallback for uncertain cases", value=llm_service.is_configured(),
                      disabled=not llm_service.is_configured(),
                      help="Uncertain candidates are sent (with a few neighbouring sentences) to Groq.")

tab_audio, tab_text = st.tabs(["Audio / video", "Transcript (text)"])


def _finish(mid: int, status_box) -> None:
    status_box.update(label="Analysing transcript", state="running")
    stats = pipeline.analyze_meeting(mid, use_llm=use_llm, progress=lambda m: status_box.write(m))
    status_box.update(label="Done", state="complete")
    st.session_state["last_meeting_id"] = mid
    st.success(f"Processed. {stats['candidates']} candidates · {stats['resolved_by_classical']} resolved by classical NLP · "
               f"{stats['llm_calls']} LLM call(s).")
    st.page_link("pages/2_Meeting_Insights.py", label="Review the insights", icon=":material/insights:")


with tab_audio:
    if not llm_service.is_configured():
        st.warning("Transcription needs GROQ_API_KEY. Use the transcript tab, or configure the key.")
    if not media.ffmpeg_available():
        st.warning("ffmpeg is not available on this host; audio upload is disabled.")
    up = st.file_uploader("Recording", type=sorted(config.AUDIO_EXTENSIONS))
    lang = st.text_input("Language code (optional, e.g. en)", value="en", max_chars=5)
    if st.button("Transcribe and analyse", type="primary", disabled=up is None or not title.strip()
                 or not llm_service.is_configured() or not media.ffmpeg_available()):
        path = None
        mid = None
        try:
            if up.size > config.MAX_UPLOAD_MB * 1024 * 1024:
                raise media.MediaError(f"File is larger than the {config.MAX_UPLOAD_MB} MB demo limit.")
            ext = media.validate_extension(up.name, config.AUDIO_EXTENSIONS)
            limits.use_transcription()
            config.ensure_dirs()
            path = config.UPLOAD_DIR / f"{uuid.uuid4().hex}.{ext}"
            path.write_bytes(up.getbuffer())
            mid = db.create_meeting(title.strip(), mdate.isoformat(), "audio", tz.strip() or "UTC", media_path=str(path))
            with st.status("Transcribing with Groq Whisper", expanded=True) as box:
                segs, dur = transcription.transcribe_media(
                    path, mid, language=lang.strip() or None,
                    progress=lambda i, n: box.write(f"Transcribed chunk {i}/{n}"))
                if not segs:
                    raise ValueError("No speech was detected in the recording.")
                db.replace_segments(mid, segs)
                db.set_status(mid, "transcribed", duration_s=dur)
                _finish(mid, box)
        except (media.MediaError, limits.LimitReached, llm_service.LLMUnavailable, ValueError) as e:
            st.error(explain_error(e))
            if mid is not None:
                db.delete_meeting(mid)
        except Exception as e:  # unexpected: do not leave a half-built meeting around
            st.error(f"Unexpected error: {explain_error(e)}")
            if mid is not None:
                db.delete_meeting(mid)
        finally:
            if path:
                Path(path).unlink(missing_ok=True)  # never retain the uploaded recording

with tab_text:
    f = st.file_uploader("Transcript file", type=sorted(config.TEXT_EXTENSIONS), key="tfile")
    pasted = st.text_area("…or paste a transcript", height=220,
                          placeholder="[00:00:05] Rahul: Let's start.\n[00:00:12] Ananya: I'll send the report by Friday.")
    st.caption("Supported: plain text (optionally `[hh:mm:ss] Speaker: text`), WebVTT, SRT.")
    if st.button("Analyse transcript", type="primary", disabled=not title.strip() or not (f or pasted.strip())):
        try:
            raw = f.getvalue().decode("utf-8", errors="replace") if f else pasted
            if len(raw) > 400_000:
                raise ValueError("Transcript is too large for the demo (400k characters max).")
            segs = parse_transcript(raw, f.name if f else "")
            if not segs:
                raise ValueError("No text could be parsed from the transcript.")
            mid = db.create_meeting(title.strip(), mdate.isoformat(), "transcript", tz.strip() or "UTC")
            db.replace_segments(mid, segs)
            db.set_status(mid, "transcribed")
            with st.status("Analysing", expanded=True) as box:
                _finish(mid, box)
        except (ValueError, limits.LimitReached) as e:
            st.error(explain_error(e))
        except Exception as e:
            st.error(f"Unexpected error: {explain_error(e)}")
