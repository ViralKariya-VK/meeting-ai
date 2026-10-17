"""SQLite access: canonical store. Chunk FTS5 index lives alongside (rowid == chunks.id)."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

from . import config

_SCHEMA = Path(__file__).resolve().parent.parent / "database" / "schema.sql"


def connect() -> sqlite3.Connection:
    config.ensure_dirs()
    conn = sqlite3.connect(config.DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        conn.execute("PRAGMA journal_mode = WAL")
    except sqlite3.DatabaseError:
        pass
    return conn


@contextmanager
def session() -> Iterator[sqlite3.Connection]:
    conn = connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    with session() as c:
        c.executescript(_SCHEMA.read_text())
        try:
            c.execute("CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(text, tokenize='porter unicode61')")
        except sqlite3.OperationalError:
            pass  # FTS5 unavailable: keyword search falls back to LIKE


def has_fts(conn: sqlite3.Connection) -> bool:
    row = conn.execute("SELECT name FROM sqlite_master WHERE name='chunks_fts'").fetchone()
    return row is not None


def rows(sql: str, params: Iterable[Any] = ()) -> list[sqlite3.Row]:
    with session() as c:
        return c.execute(sql, tuple(params)).fetchall()


def one(sql: str, params: Iterable[Any] = ()) -> Optional[sqlite3.Row]:
    with session() as c:
        return c.execute(sql, tuple(params)).fetchone()


def execute(sql: str, params: Iterable[Any] = ()) -> int:
    with session() as c:
        return c.execute(sql, tuple(params)).lastrowid


def audit(event: str, meeting_id: Optional[int] = None, detail: str = "") -> None:
    execute("INSERT INTO audit_logs(event, meeting_id, detail) VALUES (?,?,?)", (event, meeting_id, detail))


# ---- meetings -------------------------------------------------------------

def create_meeting(title: str, meeting_date: str, source_type: str, timezone: str = "UTC",
                   media_path: Optional[str] = None, is_seed: bool = False) -> int:
    mid = execute(
        "INSERT INTO meetings(title, meeting_date, timezone, source_type, media_path, is_seed) VALUES (?,?,?,?,?,?)",
        (title, meeting_date, timezone, source_type, media_path, int(is_seed)),
    )
    audit("meeting_created", mid, f"{source_type}:{title}")
    return mid


def set_status(meeting_id: int, status: str, duration_s: Optional[float] = None) -> None:
    if duration_s is None:
        execute("UPDATE meetings SET status=? WHERE id=?", (status, meeting_id))
    else:
        execute("UPDATE meetings SET status=?, duration_s=? WHERE id=?", (status, duration_s, meeting_id))


def list_meetings() -> list[sqlite3.Row]:
    return rows("SELECT * FROM meetings ORDER BY meeting_date DESC, id DESC")


def get_meeting(meeting_id: int) -> Optional[sqlite3.Row]:
    return one("SELECT * FROM meetings WHERE id=?", (meeting_id,))


def replace_segments(meeting_id: int, segments: list[dict]) -> None:
    with session() as c:
        c.execute("DELETE FROM transcript_segments WHERE meeting_id=?", (meeting_id,))
        c.executemany(
            "INSERT INTO transcript_segments(meeting_id, idx, start_s, end_s, speaker, text) VALUES (?,?,?,?,?,?)",
            [(meeting_id, i, s.get("start"), s.get("end"), s.get("speaker"), s["text"]) for i, s in enumerate(segments)],
        )
        c.execute("DELETE FROM speakers WHERE meeting_id=?", (meeting_id,))
        labels = sorted({s.get("speaker") for s in segments if s.get("speaker")})
        c.executemany("INSERT INTO speakers(meeting_id, label) VALUES (?,?)", [(meeting_id, label) for label in labels])


def get_segments(meeting_id: int) -> list[dict]:
    return [dict(r) for r in rows(
        "SELECT idx, start_s, end_s, speaker, text FROM transcript_segments WHERE meeting_id=? ORDER BY idx", (meeting_id,))]


def delete_meeting(meeting_id: int) -> None:
    """Remove the meeting, its derived rows, vectors/FTS entries, and media files."""
    m = get_meeting(meeting_id)
    if m is None:
        return
    with session() as c:
        if has_fts(c):
            c.execute("DELETE FROM chunks_fts WHERE rowid IN (SELECT id FROM chunks WHERE meeting_id=?)", (meeting_id,))
        c.execute("DELETE FROM meetings WHERE id=?", (meeting_id,))  # cascades to chunks (vectors), items, drafts...
    for p in (m["media_path"],):
        if p:
            Path(p).unlink(missing_ok=True)
    for f in config.AUDIO_DIR.glob(f"m{meeting_id}_*"):
        f.unlink(missing_ok=True)
    audit("meeting_deleted", None, f"id={meeting_id}")


def json_loads(s: Optional[str], default: Any = None) -> Any:
    try:
        return json.loads(s) if s else default
    except (TypeError, ValueError):
        return default
