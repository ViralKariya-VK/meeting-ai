"""Transcript parsing and sentence splitting. Preprocessing is deliberately light:
negations, names, numbers and dates must survive, and the original text stays canonical."""
from __future__ import annotations

import re
from typing import Optional

_TS = r"(?:(\d{1,2}):)?(\d{1,2}):(\d{2})(?:[.,](\d{1,3}))?"
_CUE = re.compile(rf"^\s*{_TS}\s*-->\s*{_TS}")
_LINE_TS = re.compile(rf"^\s*[\[(]?{_TS}[\])]?\s*[-–]?\s*")
_SPEAKER = re.compile(r"^\s*(?:<v\s+([^>]+)>|([A-Z][\w.'-]*(?:\s[A-Z][\w.'-]*){0,2})\s*:\s+)")
_ABBREV = {"mr", "mrs", "ms", "dr", "vs", "e.g", "i.e", "etc", "prof", "sr", "jr", "st"}


def _secs(h, m, s, ms) -> float:
    return int(h or 0) * 3600 + int(m) * 60 + int(s) + (int(ms.ljust(3, "0")) / 1000 if ms else 0)


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", text)).strip()


def parse_transcript(text: str, filename: str = "") -> list[dict]:
    """Return segments: {start, end, speaker, text}. Handles .vtt/.srt cues, '[00:12] Name: text' lines, or prose."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").lstrip("﻿")
    lines = text.split("\n")
    segments: list[dict] = []

    if any(_CUE.match(l) for l in lines):
        i = 0
        while i < len(lines):
            m = _CUE.match(lines[i])
            if not m:
                i += 1
                continue
            g = m.groups()
            start, end = _secs(*g[0:4]), _secs(*g[4:8])
            i += 1
            buf = []
            while i < len(lines) and lines[i].strip() and not _CUE.match(lines[i]):
                buf.append(lines[i])
                i += 1
            raw = " ".join(buf)
            sp = _SPEAKER.match(raw)
            speaker = (sp.group(1) or sp.group(2)).strip() if sp else None
            body = _clean(raw[sp.end():] if sp else raw)
            if body:
                segments.append({"start": start, "end": end, "speaker": speaker, "text": body})
        return segments

    for line in lines:
        if not line.strip() or line.strip().isdigit():
            continue
        start = None
        ts = _LINE_TS.match(line)
        if ts and ts.group(2) is not None:
            start = _secs(*ts.groups()[0:4])
            line = line[ts.end():]
        sp = _SPEAKER.match(line)
        speaker = (sp.group(1) or sp.group(2)).strip() if sp else None
        body = _clean(line[sp.end():] if sp else line)
        if body:
            segments.append({"start": start, "end": None, "speaker": speaker, "text": body})

    # fill missing ends from the next start
    for a, b in zip(segments, segments[1:]):
        if a["end"] is None:
            a["end"] = b["start"]
    return segments


def split_sentences(text: str) -> list[str]:
    parts: list[str] = []
    buf = ""
    for tok in re.split(r"(\s+)", text):
        buf += tok
        if re.search(r"[.!?][\"')\]]*$", tok.strip()) and tok.strip():
            word = re.sub(r"[^\w.]", "", tok.strip().lower()).rstrip(".")
            if word in _ABBREV or re.fullmatch(r"[a-z]", word or "xx"):
                continue
            parts.append(buf.strip())
            buf = ""
    if buf.strip():
        parts.append(buf.strip())
    return [p for p in parts if p]


def sentences_from_segments(segments: list[dict]) -> list[dict]:
    """Flatten segments into sentence units that remember their segment index for evidence links."""
    out = []
    for i, seg in enumerate(segments):
        for s in split_sentences(seg["text"]):
            out.append({"seg_idx": i, "text": s, "speaker": seg.get("speaker"), "start_s": seg.get("start")})
    return out


def fmt_ts(seconds: Optional[float]) -> str:
    if seconds is None:
        return ""
    s = int(seconds)
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def segments_to_text(segments: list[dict]) -> str:
    lines = []
    for s in segments:
        prefix = f"[{fmt_ts(s.get('start'))}] " if s.get("start") is not None else ""
        spk = f"{s['speaker']}: " if s.get("speaker") else ""
        lines.append(f"{prefix}{spk}{s['text']}")
    return "\n".join(lines)
