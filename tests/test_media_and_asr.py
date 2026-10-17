from pathlib import Path

from services import media, transcription


def test_chunk_plan_covers_duration_with_ownership():
    plans = media.plan_chunks(1500, 600, 8)
    assert len(plans) == 3
    assert plans[0][0] == 0 and plans[1][0] == 592  # overlap before chunk 2
    owns = [(p[2], p[3]) for p in plans]
    assert owns[0][1] == owns[1][0] and owns[1][1] == owns[2][0]  # no gaps / overlaps in ownership
    assert owns[2][1] > 1500


def test_short_audio_single_chunk():
    assert len(media.plan_chunks(30, 600, 8)) == 1


def test_merge_removes_overlap_duplicates_and_offsets_time():
    c0 = media.AudioChunk(Path("a"), 0.0, 0.0, 600.0)
    c1 = media.AudioChunk(Path("b"), 592.0, 600.0, 1200.0)
    seg0 = [{"start": 590.0, "end": 599.0, "text": "tail words"}, {"start": 598.0, "end": 601.0, "text": "bridge"}]
    seg1 = [{"start": 0.0, "end": 6.0, "text": "tail words"}, {"start": 6.0, "end": 10.0, "text": "bridge"},
            {"start": 12.0, "end": 15.0, "text": "new speech"}]
    out = transcription.merge_chunk_segments([(c0, seg0), (c1, seg1)])
    texts = [s["text"] for s in out]
    assert texts == ["tail words", "bridge", "new speech"]
    assert out[-1]["start"] == 604.0


def test_validate_extension_rejects_unknown():
    import pytest
    with pytest.raises(media.MediaError):
        media.validate_extension("evil.exe", {"mp3"})


def test_duration_cap_is_enforced(monkeypatch):
    import pytest
    monkeypatch.setattr(media, "probe_duration", lambda p: 16 * 60.0)
    with pytest.raises(media.MediaError, match="at most 15 minutes"):
        media.prepare_chunks(Path("x.mp3"), 1)
