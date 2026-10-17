"""ffmpeg helpers: validate, probe, normalise to 16 kHz mono FLAC, split into overlapping chunks."""
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from . import config


class MediaError(Exception):
    pass


@dataclass
class AudioChunk:
    path: Path
    start_s: float      # offset of this chunk in the original timeline
    own_from: float     # this chunk "owns" segments whose midpoint is in [own_from, own_to)
    own_to: float


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def validate_extension(filename: str, allowed: set[str]) -> str:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in allowed:
        raise MediaError(f"Unsupported file type '.{ext}'. Allowed: {', '.join(sorted(allowed))}")
    return ext


def probe_duration(path: Path) -> float:
    if not ffmpeg_available():
        raise MediaError("ffmpeg/ffprobe is not installed on this host.")
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
        capture_output=True, text=True, timeout=60,
    )
    if out.returncode != 0:
        raise MediaError("Could not read the media file (corrupt or unsupported).")
    try:
        return float(json.loads(out.stdout)["format"]["duration"])
    except (KeyError, ValueError, json.JSONDecodeError):
        raise MediaError("Could not determine the media duration.")


def plan_chunks(duration_s: float, chunk_s: int, overlap_s: int) -> list[tuple[float, float, float, float]]:
    """Return (start, length, own_from, own_to) per chunk. Ownership boundaries are the nominal
    multiples of chunk_s, so overlap text is de-duplicated by segment midpoint."""
    plans = []
    n = max(1, int(-(-duration_s // chunk_s)))
    for i in range(n):
        own_from = i * chunk_s
        own_to = duration_s + 1 if i == n - 1 else (i + 1) * chunk_s
        start = max(0.0, own_from - (overlap_s if i > 0 else 0))
        end = min(duration_s, own_to + overlap_s)
        plans.append((start, end - start, float(own_from), float(own_to)))
    return plans


def prepare_chunks(src: Path, meeting_id: int, max_minutes: int | None = None) -> tuple[list[AudioChunk], float]:
    """Convert to 16 kHz mono FLAC chunks. Rejects media longer than the demo cap."""
    max_minutes = max_minutes or config.MAX_AUDIO_MINUTES
    duration = probe_duration(src)
    if duration > max_minutes * 60:
        raise MediaError(f"Demo limit: media must be at most {max_minutes} minutes (this one is {duration/60:.1f}).")
    config.ensure_dirs()
    chunks: list[AudioChunk] = []
    for i, (start, length, own_from, own_to) in enumerate(
        plan_chunks(duration, config.CHUNK_SECONDS, config.CHUNK_OVERLAP_SECONDS)
    ):
        out = config.AUDIO_DIR / f"m{meeting_id}_{i:03d}.flac"
        cmd = ["ffmpeg", "-y", "-v", "error", "-ss", f"{start:.2f}", "-t", f"{length:.2f}", "-i", str(src),
               "-vn", "-ac", "1", "-ar", "16000", "-c:a", "flac", str(out)]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if r.returncode != 0 or not out.exists():
            raise MediaError("ffmpeg failed to convert the media: " + r.stderr[-200:])
        if out.stat().st_size > 24 * 1024 * 1024:
            raise MediaError("A converted chunk exceeds Groq's 25 MB limit; lower CHUNK_SECONDS.")
        chunks.append(AudioChunk(out, start, own_from, own_to))
    return chunks, duration
