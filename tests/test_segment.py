from itertools import pairwise

from audiosearch import config
from audiosearch.segment import Word, chunk_transcript, clean_speakers, smooth_flips


def dg(words):
    """Minimal normalized transcript from (text, start, end, speaker) tuples."""
    return {"provider": "test", "duration": words[-1][2] if words else 0, "speakers": {},
            "words": [{"text": t, "start": s, "end": e, "speaker": sp} for t, s, e, sp in words]}


def speech(speaker, start, n_sentences, words_per=10, word_s=0.4):
    """n sentences of distinct words; returns (words, end_time)."""
    out, t = [], start
    for s in range(n_sentences):
        for i in range(words_per):
            text = f"s{s}w{i}" + ("." if i == words_per - 1 else "")
            out.append((text, t, t + word_s, speaker))
            t += word_s
    return out, t


def test_stray_speaker_labels_are_folded_into_neighbour():
    ws = [Word("a", 0, 1, 0), Word("b", 1, 2, 0), Word("c", 2, 3, 5), Word("d", 3, 4, 1),
          Word("e", 4, 5, 1), Word("f", 5, 6, 1)]
    assert [w.speaker for w in clean_speakers(ws)] == [0, 0, 0, 1, 1, 1]


def test_mid_sentence_flip_is_smoothed_but_real_backchannel_is_kept():
    flip = [Word("we", 0, 1, 0), Word("use", 1, 2, 0), Word("rank", 2, 3, 1), Word("fusion.", 3, 4, 0)]
    assert {w.speaker for w in smooth_flips(flip)} == {0}
    backchannel = [Word("it", 0, 1, 0), Word("works.", 1, 2, 0), Word("Right.", 2, 3, 1), Word("So", 3, 4, 0)]
    assert [w.speaker for w in smooth_flips(backchannel)] == [0, 0, 1, 0]


def test_children_are_speaker_pure_bounded_and_inside_their_parent():
    a, t = speech(0, 0.0, 40)        # long answer
    b, t = speech(1, t, 3)           # question
    c, _ = speech(0, t, 12)
    parents, children = chunk_transcript(dg(a + b + c))
    by_index = {p.index: p for p in parents}
    for ch in children:
        assert len({w.speaker for w in ch.words}) == 1
        assert ch.tokens <= config.CHILD_TOKENS + config.CHILD_MIN_TOKENS   # tail merge may add a little
        p = by_index[ch.parent]
        assert p.start <= ch.start and ch.end <= p.end
    assert all(p.tokens <= config.PARENT_TOKENS for p in parents)
    assert len(parents) >= 2 and len(children) > len(parents)


def test_question_and_answer_share_a_parent_and_render_as_dialogue():
    q, t = speech(1, 0.0, 2)
    a, _ = speech(0, t, 4)
    parents, children = chunk_transcript(dg(q + a))
    assert len(parents) == 1 and {c.parent for c in children} == {0}
    # labels renumber by first appearance: the questioner becomes speaker 0
    assert parents[0].render({0: "Host", 1: "Guest"}).startswith("Host: s0w0")
    assert "\nGuest: " in parents[0].render({0: "Host", 1: "Guest"})


def test_adjacent_children_overlap_within_the_budget():
    a, _ = speech(0, 0.0, 30)
    _, children = chunk_transcript(dg(a))
    for prev, nxt in pairwise(children):
        if prev.parent == nxt.parent:
            shared = [w for w in nxt.words if w.start < prev.end]
            assert 0 < sum(w.tokens for w in shared) <= config.CHILD_OVERLAP_TOKENS


def test_backchannel_turn_gets_no_child_but_stays_in_parent_text():
    a, t = speech(0, 0.0, 3)
    yeah = [("Yeah.", t, t + 0.3, 1)]
    b, _ = speech(0, t + 0.5, 3)
    parents, children = chunk_transcript(dg(a + yeah + b))
    assert all(c.speaker == 0 for c in children)
    assert "Speaker 1: Yeah." in parents[0].render()


def test_run_on_turn_without_punctuation_is_hard_split():
    words = [(f"um{i}", i * 0.3, i * 0.3 + 0.3, 0) for i in range(900)]
    parents, children = chunk_transcript(dg(words))
    assert len(parents) >= 2
    assert all(c.tokens <= config.CHILD_TOKENS + config.CHILD_MIN_TOKENS for c in children)
