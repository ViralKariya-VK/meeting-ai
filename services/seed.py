"""Seed the demo with sample meetings on first run (storage is ephemeral on free hosting)."""
from __future__ import annotations

from . import config, db, pipeline
from .preprocessing import parse_transcript

SEED_MEETINGS = [
    ("product_launch_sync.txt", "Product Launch Sync", "2026-10-12"),
    ("budget_review.txt", "Quarterly Budget Review", "2026-10-13"),
    ("research_standup.txt", "NLP Research Standup", "2026-10-14"),
]


def seed_if_empty(use_llm: bool = False) -> int:
    if db.one("SELECT id FROM meetings LIMIT 1") is not None:
        return 0
    n = 0
    for fname, title, mdate in SEED_MEETINGS:
        path = config.SEED_DIR / fname
        if not path.exists():
            continue
        segs = parse_transcript(path.read_text(encoding="utf-8"), fname)
        mid = db.create_meeting(title, mdate, "seed", is_seed=True)
        db.replace_segments(mid, segs)
        db.set_status(mid, "transcribed", duration_s=segs[-1]["start"] if segs else None)
        pipeline.analyze_meeting(mid, use_llm=use_llm)  # classical-only: no API quota spent at startup
        n += 1
    return n
