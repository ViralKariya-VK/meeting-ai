"""Chunking + hybrid retrieval. Chunks are the unit for both keyword (FTS5) and semantic (cosine) search.
Vectors live in SQLite and are searched with numpy (fine at demo scale); they are a derived, rebuildable index."""
from __future__ import annotations

import re
from typing import Optional

import numpy as np

from . import config, db, embedding_service


def chunk_segments(segments: list[dict], target_words: Optional[int] = None,
                   overlap_words: Optional[int] = None) -> list[dict]:
    """Group whole segments into ~target_words chunks, repeating trailing segments as overlap."""
    target = target_words or config.CHUNK_WORDS
    overlap = config.CHUNK_OVERLAP_WORDS if overlap_words is None else overlap_words
    chunks: list[dict] = []
    i, n = 0, len(segments)
    while i < n:
        j, words = i, 0
        while j < n and (words == 0 or words + len(segments[j]["text"].split()) <= target):
            words += len(segments[j]["text"].split())
            j += 1
        group = segments[i:j]
        starts = [s.get("start_s", s.get("start")) for s in group if s.get("start_s", s.get("start")) is not None]
        ends = [s.get("end_s", s.get("end")) for s in group if s.get("end_s", s.get("end")) is not None]
        speakers = sorted({s["speaker"] for s in group if s.get("speaker")})
        chunks.append({
            "text": " ".join((f"{s['speaker']}: " if s.get("speaker") else "") + s["text"] for s in group),
            "seg_start": i, "seg_end": j - 1,
            "start_s": min(starts) if starts else None, "end_s": max(ends) if ends else None,
            "speakers": ", ".join(speakers),
        })
        if j >= n:
            break
        # step back to create overlap, but always make progress
        back, acc = j, 0
        while back > i + 1 and acc < overlap:
            back -= 1
            acc += len(segments[back]["text"].split())
        i = max(back, i + 1)
    return chunks


def index_meeting(meeting_id: int) -> int:
    segments = db.get_segments(meeting_id)
    chunks = chunk_segments(segments)
    vecs = embedding_service.embed_texts([c["text"] for c in chunks]) if chunks else np.zeros((0, 1), np.float32)
    with db.session() as c:
        if db.has_fts(c):
            c.execute("DELETE FROM chunks_fts WHERE rowid IN (SELECT id FROM chunks WHERE meeting_id=?)", (meeting_id,))
        c.execute("DELETE FROM chunks WHERE meeting_id=?", (meeting_id,))
        for idx, (ch, v) in enumerate(zip(chunks, vecs)):
            cid = c.execute(
                "INSERT INTO chunks(meeting_id, idx, text, seg_start, seg_end, start_s, end_s, speakers, embedding)"
                " VALUES (?,?,?,?,?,?,?,?,?)",
                (meeting_id, idx, ch["text"], ch["seg_start"], ch["seg_end"], ch["start_s"], ch["end_s"],
                 ch["speakers"], v.tobytes())).lastrowid
            if db.has_fts(c):
                c.execute("INSERT INTO chunks_fts(rowid, text) VALUES (?,?)", (cid, ch["text"]))
        c.execute("INSERT INTO kv(key, value) VALUES ('embedding_backend', ?) "
                  "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (embedding_service.backend(),))
    return len(chunks)


def rebuild_all() -> int:
    total = 0
    for m in db.list_meetings():
        if db.get_segments(m["id"]):
            total += index_meeting(m["id"])
    return total


def _meeting_filter(meeting_ids: Optional[list[int]], alias: str = "c") -> tuple[str, list]:
    if not meeting_ids:
        return "", []
    return f" AND {alias}.meeting_id IN ({','.join('?' * len(meeting_ids))})", list(meeting_ids)


def _row_to_hit(r, score: float, how: str) -> dict:
    return {"chunk_id": r["id"], "meeting_id": r["meeting_id"], "title": r["title"], "meeting_date": r["meeting_date"],
            "text": r["text"], "start_s": r["start_s"], "end_s": r["end_s"], "seg_start": r["seg_start"],
            "seg_end": r["seg_end"], "speakers": r["speakers"], "score": float(score), "via": how}


def _fts_query(q: str) -> str:
    phrases = re.findall(r'"([^"]+)"', q)
    rest = re.sub(r'"[^"]*"', " ", q)
    terms = [f'"{p.strip()}"' for p in phrases if p.strip()]
    terms += [f'"{t}"' for t in re.findall(r"[A-Za-z0-9']{2,}", rest)]
    return " OR ".join(terms)


def keyword_search(query: str, meeting_ids: Optional[list[int]] = None, k: int = 8) -> list[dict]:
    flt, params = _meeting_filter(meeting_ids)
    with db.session() as conn:
        if db.has_fts(conn):
            fq = _fts_query(query)
            if not fq:
                return []
            sql = (f"SELECT c.*, m.title, m.meeting_date, bm25(chunks_fts) AS s FROM chunks_fts "
                   f"JOIN chunks c ON c.id = chunks_fts.rowid JOIN meetings m ON m.id = c.meeting_id "
                   f"WHERE chunks_fts MATCH ?{flt} ORDER BY s LIMIT ?")
            res = conn.execute(sql, [fq, *params, k]).fetchall()
            return [_row_to_hit(r, -r["s"], "keyword") for r in res]
        like = [f"%{t}%" for t in re.findall(r"[A-Za-z0-9']{3,}", query)]
        if not like:
            return []
        cond = " OR ".join("c.text LIKE ?" for _ in like)
        sql = (f"SELECT c.*, m.title, m.meeting_date FROM chunks c JOIN meetings m ON m.id=c.meeting_id "
               f"WHERE ({cond}){flt} LIMIT ?")
        return [_row_to_hit(r, 1.0, "keyword") for r in conn.execute(sql, [*like, *params, k]).fetchall()]


def semantic_search(query: str, meeting_ids: Optional[list[int]] = None, k: int = 8) -> list[dict]:
    flt, params = _meeting_filter(meeting_ids)
    with db.session() as conn:
        rows = conn.execute(
            f"SELECT c.*, m.title, m.meeting_date FROM chunks c JOIN meetings m ON m.id=c.meeting_id "
            f"WHERE c.embedding IS NOT NULL{flt}", params).fetchall()
        stored = conn.execute("SELECT value FROM kv WHERE key='embedding_backend'").fetchone()
    if not rows:
        return []
    if stored and stored["value"] != embedding_service.backend():
        raise RuntimeError("Index was built with a different embedding backend; rebuild it in Settings.")
    mat = np.vstack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
    q = embedding_service.embed_texts([query], is_query=True)[0]
    if mat.shape[1] != q.shape[0]:
        raise RuntimeError("Embedding dimensions differ from the index; rebuild it in Settings.")
    sims = mat @ q
    top = np.argsort(sims)[::-1][:k]
    return [_row_to_hit(rows[i], sims[i], "semantic") for i in top]


def hybrid_search(query: str, meeting_ids: Optional[list[int]] = None, k: int = 6, rrf_k: int = 60) -> list[dict]:
    """Reciprocal-rank fusion of keyword and semantic rankings."""
    kw = keyword_search(query, meeting_ids, k=k * 2)
    sem = semantic_search(query, meeting_ids, k=k * 2)
    fused: dict[int, dict] = {}
    for ranking in (kw, sem):
        for rank, hit in enumerate(ranking):
            entry = fused.setdefault(hit["chunk_id"], {**hit, "score": 0.0, "via": set()})
            entry["score"] += 1.0 / (rrf_k + rank + 1)
            entry["via"].add(hit["via"])
    out = sorted(fused.values(), key=lambda h: h["score"], reverse=True)[:k]
    for h in out:
        h["via"] = "+".join(sorted(h["via"]))
    return out
