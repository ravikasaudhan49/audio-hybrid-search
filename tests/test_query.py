"""Speaker/role detection in queries and the relevance threshold (no API, no DB)."""
import pytest

from audiosearch.query import parse
from audiosearch.search import Result, apply_thresholds

SPEAKERS = ["Gary Jordan", "Abba Zubair", "Leah Cheshire", "Gene Kranz", "Patrick O'Neil"]


@pytest.mark.parametrize("q,speaker,role,text", [
    ("what did Jordan say about stem cells", "Gary Jordan", None, "stem cells"),
    ("what did Gary say about CASIS", "Gary Jordan", None, "CASIS"),
    ("Gene Kranz's views on failure", "Gene Kranz", None, "failure"),
    ("Kranz on accountability", "Gene Kranz", None, "accountability"),
    ("what did Patrick O'Neil say about pharmaceutical companies", "Patrick O'Neil", None, "pharmaceutical companies"),
    ("what did the guest say about trust", None, "guest", "trust"),
    ("what did the host ask about Nigeria", None, "host", "Nigeria"),
    ("the guest's opinion on microgravity", None, "guest", "microgravity"),
    ("what did Kranz say?", "Gene Kranz", None, "what did Kranz say?"),   # no topic: search whole question
    ("what did Gene say about the moon landing", "Gene Kranz", None, "moon landing"),
    ("according to Patrick, why use the station", "Patrick O'Neil", None, "use station"),
])
def test_detects_speaker_or_role_and_strips_framing(q, speaker, role, text):
    p = parse(q, SPEAKERS)
    assert (p.speaker, p.role, p.text) == (speaker, role, text)


@pytest.mark.parametrize("q", ["Jerry Bostick", "failure is not an option", "guest list",
                               "hostile environment of space", "Apollo one fire",
                               '"affect gene expression"', "gene therapy in space",   # "Gene" is also a word
                               "Jordan River", "Patrick space station"])
def test_leaves_ordinary_queries_alone(q):
    p = parse(q, SPEAKERS)
    assert p.speaker is None and p.role is None and p.text == q


def test_ambiguous_first_name_is_not_guessed():
    p = parse("what did Gary say about space", ["Gary Jordan", "Gary Smith"])
    assert p.speaker is None


def _r(score):
    return Result(0, "f", "t", "a", "s", 0, 1, 0, "x", "x", score, {}, "f:p00", 0, 1, "x")


def test_threshold_drops_low_scores_even_inside_top_k():
    kept, cutoff = apply_thresholds([_r(0.95), _r(0.64), _r(0.45), _r(0.29)], min_score=0.40, min_relative=0.5)
    assert [r.score for r in kept] == [0.95, 0.64] and cutoff == pytest.approx(0.475)


def test_threshold_can_leave_nothing_for_off_topic_queries():
    kept, _ = apply_thresholds([_r(0.29), _r(0.28), _r(0.25)], min_score=0.40, min_relative=0.5)
    assert kept == []
