from services import db, retrieval, query_router
from services.preprocessing import parse_transcript

TEXT = """[00:00:05] Ana: We discussed the launch delays caused by the vendor audit.
[00:01:00] Raj: The release approval from legal is still pending.
[00:02:00] Ana: Budget for hosting went up twelve percent this quarter.
[00:03:00] Raj: We should cancel two unused staging environments to save money.
"""


def make(title="M1", d="2026-10-12"):
    mid = db.create_meeting(title, d, "transcript")
    db.replace_segments(mid, parse_transcript(TEXT))
    retrieval.index_meeting(mid)
    return mid


def test_chunking_overlaps_and_progresses():
    segs = [{"text": "word " * 100, "speaker": None, "start": i, "end": i + 1} for i in range(6)]
    chunks = retrieval.chunk_segments(segs, target_words=250, overlap_words=100)
    assert len(chunks) >= 3 and chunks[0]["seg_start"] == 0 and chunks[-1]["seg_end"] == 5
    assert all(c2["seg_start"] > c1["seg_start"] for c1, c2 in zip(chunks, chunks[1:]))


def test_keyword_phrase_search():
    make()
    hits = retrieval.keyword_search('"release approval"')
    assert hits and "release approval" in hits[0]["text"]


def test_semantic_and_hybrid_find_related_chunk():
    make()
    sem = retrieval.semantic_search("vendor audit delays", k=2)
    assert sem and "vendor audit" in sem[0]["text"]
    hyb = retrieval.hybrid_search("hosting budget increase", k=2)
    assert hyb and "hosting" in hyb[0]["text"]


def test_scope_filter():
    a, b = make("A"), make("B")
    hits = retrieval.keyword_search("budget", [b])
    assert hits and all(h["meeting_id"] == b for h in hits)


def test_router_modes_and_date_filter():
    from datetime import date
    make("A", "2026-10-12")
    make("B", "2026-09-01")
    assert query_router.plan('"release approval"').mode == "keyword"
    assert query_router.plan("What did we decide about hosting?").mode == "hybrid"
    r = query_router.plan("What was said on 12 October?", today=date(2026, 10, 17))
    assert r.date_filter and len(r.meeting_ids) == 1


def test_delete_removes_everything_including_vectors_and_fts():
    mid = make()
    assert db.rows("SELECT id FROM chunks WHERE meeting_id=?", (mid,))
    db.delete_meeting(mid)
    assert not db.rows("SELECT id FROM chunks")
    assert not db.rows("SELECT rowid FROM chunks_fts")
    assert db.get_meeting(mid) is None
