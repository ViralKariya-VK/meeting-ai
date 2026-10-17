from datetime import date

from services.extraction import normalize_deadline
from services.preprocessing import parse_transcript, sentences_from_segments, split_sentences

MEETING = date(2026, 10, 12)  # a Monday


def test_parse_bracket_lines_with_speakers():
    segs = parse_transcript("[00:00:05] Rahul: Let's start.\n[00:01:10] Ananya: I'll send it by Friday.")
    assert [s["speaker"] for s in segs] == ["Rahul", "Ananya"]
    assert segs[1]["start"] == 70 and segs[0]["end"] == 70


def test_parse_vtt():
    vtt = "WEBVTT\n\n00:00:01.000 --> 00:00:04.000\n<v Mark>I will fix it.</v>\n\n00:00:05.000 --> 00:00:07.000\nOkay.\n"
    segs = parse_transcript(vtt, "x.vtt")
    assert segs[0]["speaker"] == "Mark" and segs[0]["text"] == "I will fix it."
    assert segs[1]["speaker"] is None


def test_sentence_split_keeps_abbreviations_and_negation():
    s = split_sentences("Dr. Rao will not send the report. We agreed! Why?")
    assert s[0] == "Dr. Rao will not send the report."
    assert len(s) == 3


def test_sentences_remember_segment():
    segs = parse_transcript("A: One. Two.\nB: Three.")
    out = sentences_from_segments(segs)
    assert [o["seg_idx"] for o in out] == [0, 0, 1]


def test_deadline_normalisation():
    assert normalize_deadline("tomorrow", MEETING) == (date(2026, 10, 13), "ok")
    assert normalize_deadline("by Friday", MEETING) == (date(2026, 10, 16), "ok")
    assert normalize_deadline("next Tuesday", MEETING) == (date(2026, 10, 20), "ok")
    assert normalize_deadline("by the 15th", MEETING) == (date(2026, 10, 15), "ok")
    assert normalize_deadline("by 12 October", MEETING) == (date(2026, 10, 12), "ok")
    assert normalize_deadline("by the end of the month", MEETING) == (date(2026, 10, 31), "ok")
    assert normalize_deadline("soon", MEETING) == (None, "vague")
    assert normalize_deadline("before the next meeting", MEETING) == (None, "vague")
    assert normalize_deadline(None, MEETING) == (None, "none")


def test_same_weekday_means_next_week_not_today():
    assert normalize_deadline("on Monday", MEETING) == (date(2026, 10, 19), "ok")
