"""Audio -> transcript segments via Groq Whisper, chunk-aware."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from . import config, llm_service, media


def merge_chunk_segments(per_chunk: list[tuple[media.AudioChunk, list[dict]]]) -> list[dict]:
    """Shift segment times by the chunk offset and keep each segment only in the chunk that owns its midpoint,
    which removes the duplicated text from overlapping audio."""
    merged: list[dict] = []
    for chunk, segs in per_chunk:
        for s in segs:
            start = float(s["start"]) + chunk.start_s
            end = float(s["end"]) + chunk.start_s
            mid = (start + end) / 2
            if chunk.own_from <= mid < chunk.own_to:
                text = (s.get("text") or "").strip()
                prev = merged[-1]["text"].lower().strip(" .,!?") if merged else None
                if text and text.lower().strip(" .,!?") != prev:  # boundary-straddling segment seen by both chunks
                    merged.append({"start": start, "end": end, "speaker": None, "text": text})
    merged.sort(key=lambda x: x["start"])
    return merged


def _segments_of(resp) -> list[dict]:
    segs = getattr(resp, "segments", None)
    if segs is None and isinstance(resp, dict):
        segs = resp.get("segments")
    out = []
    for s in segs or []:
        get = (lambda k: s.get(k)) if isinstance(s, dict) else (lambda k: getattr(s, k, None))
        out.append({"start": get("start"), "end": get("end"), "text": get("text")})
    return out


def transcribe_chunk(path: Path, language: Optional[str] = None) -> list[dict]:
    client = llm_service.get_client()
    data = path.read_bytes()

    def call():
        kwargs = dict(
            file=(path.name, data), model=config.ASR_MODEL(), response_format="verbose_json",
            timestamp_granularities=["segment"], temperature=0.0,
        )
        if language:
            kwargs["language"] = language
        return client.audio.transcriptions.create(**kwargs)

    return _segments_of(llm_service._with_retry(call))


def transcribe_media(src: Path, meeting_id: int, language: Optional[str] = None,
                     progress: Optional[Callable[[int, int], None]] = None) -> tuple[list[dict], float]:
    chunks, duration = media.prepare_chunks(src, meeting_id)
    per_chunk = []
    for i, ch in enumerate(chunks):
        per_chunk.append((ch, transcribe_chunk(ch.path, language)))
        if progress:
            progress(i + 1, len(chunks))
    segments = merge_chunk_segments(per_chunk)
    for ch in chunks:  # converted audio is only needed during transcription
        ch.path.unlink(missing_ok=True)
    return segments, duration
