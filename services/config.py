"""Central configuration. Secrets come from env vars first, then st.secrets."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def get_secret(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value:
        return value
    try:  # st.secrets raises if no secrets file exists
        import streamlit as st

        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return default


def _int(name: str, default: int) -> int:
    try:
        return int(get_secret(name, str(default)))
    except (TypeError, ValueError):
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(get_secret(name, str(default)))
    except (TypeError, ValueError):
        return default


STORAGE_DIR = Path(get_secret("STORAGE_DIR", str(ROOT / "storage")))
UPLOAD_DIR = STORAGE_DIR / "uploads"
AUDIO_DIR = STORAGE_DIR / "audio"
EXPORT_DIR = STORAGE_DIR / "exports"
DB_PATH = STORAGE_DIR / "meeting_ai.sqlite3"
SEED_DIR = ROOT / "seed"
CLASSIFIER_PATH = ROOT / "models" / "classical" / "sentence_clf.joblib"
CLASSIFIER_METRICS_PATH = ROOT / "models" / "classical" / "metrics.json"
SEED_LABELLED_CSV = ROOT / "models" / "data" / "seed_labelled.csv"

# Groq (model IDs change over time: keep them configurable)
GROQ_API_KEY = lambda: get_secret("GROQ_API_KEY")  # noqa: E731  (read lazily)
ASR_MODEL = lambda: get_secret("GROQ_ASR_MODEL", "whisper-large-v3-turbo")  # noqa: E731
LLM_MODEL = lambda: get_secret("GROQ_LLM_MODEL", "llama-3.3-70b-versatile")  # noqa: E731

# Demo guard-rails
MAX_AUDIO_MINUTES = _int("MAX_AUDIO_MINUTES", 15)
MAX_UPLOAD_MB = _int("MAX_UPLOAD_MB", 50)
CHUNK_SECONDS = _int("CHUNK_SECONDS", 600)
CHUNK_OVERLAP_SECONDS = _int("CHUNK_OVERLAP_SECONDS", 8)
SESSION_MAX_TRANSCRIPTIONS = _int("SESSION_MAX_TRANSCRIPTIONS", 5)
SESSION_MAX_LLM_CALLS = _int("SESSION_MAX_LLM_CALLS", 60)

# Confidence gateway (starting values; tune on held-out labelled data)
CONF_MIN = _float("CONF_MIN", 0.80)
MARGIN_MIN = _float("MARGIN_MIN", 0.15)

# Retrieval
CHUNK_WORDS = _int("CHUNK_WORDS", 250)
CHUNK_OVERLAP_WORDS = _int("CHUNK_OVERLAP_WORDS", 40)
EMBED_BACKEND = lambda: get_secret("EMBED_BACKEND", "fastembed")  # noqa: E731  fastembed | hash
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

AUDIO_EXTENSIONS = {"flac", "mp3", "mp4", "mpeg", "mpga", "m4a", "ogg", "wav", "webm", "mov", "mkv"}
TEXT_EXTENSIONS = {"txt", "vtt", "srt"}

LABELS = ["ACTION_ITEM", "DECISION", "DISCUSSION", "QUESTION"]


def ensure_dirs() -> None:
    for d in (STORAGE_DIR, UPLOAD_DIR, AUDIO_DIR, EXPORT_DIR):
        d.mkdir(parents=True, exist_ok=True)
