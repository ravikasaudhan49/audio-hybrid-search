"""Words and speakers of a normalized transcript, cleaned to exactly two speakers.

  - stray labels are folded into the neighbouring main speaker
  - 1-2 word "turns" that interrupt a sentence of the other speaker are diarization flips
    and are relabeled
"""
from collections import Counter

from ...models import Word
from .params import CHUNKING
from .tokens import count_word_tokens

SENTENCE_END = (".", "?", "!")


def words_from_transcript(t: dict) -> list[Word]:
    words = [Word(w["text"], float(w["start"]), float(w["end"]), int(w["speaker"])) for w in t["words"]]
    for w, n in zip(words, count_word_tokens([w.text for w in words]), strict=True):
        w.tokens = n
    return words


def clean_speakers(words: list[Word], n_speakers: int = CHUNKING.n_speakers) -> list[Word]:
    """Keep the n most frequent speaker labels; relabel the rest to the previous
    main speaker (or the next one, at the start of the file). Then renumber 0..n-1
    in order of first appearance so labels are stable."""
    if not words:
        return words
    main = {s for s, _ in Counter(w.speaker for w in words).most_common(n_speakers)}
    fallback = next(w.speaker for w in words if w.speaker in main)
    prev = None
    for w in words:
        if w.speaker in main:
            prev = w.speaker
        else:
            w.speaker = prev if prev is not None else fallback
    order: dict[int, int] = {}
    for w in words:
        order.setdefault(w.speaker, len(order))
    for w in words:
        w.speaker = order[w.speaker]
    return words


def turns(words: list[Word]) -> list[list[Word]]:
    """Consecutive same-speaker words."""
    out: list[list[Word]] = []
    for w in words:
        if out and out[-1][-1].speaker == w.speaker:
            out[-1].append(w)
        else:
            out.append([w])
    return out


def smooth_flips(words: list[Word], max_words: int = 2) -> list[Word]:
    """A 1-2 word turn that lands mid-sentence of the other speaker (the word before it
    doesn't end a sentence and the same speaker resumes after it) is a diarization flip,
    not an interjection. Relabel it to the surrounding speaker."""
    ts = turns(words)
    for i in range(1, len(ts) - 1):
        before, cur, after = ts[i - 1], ts[i], ts[i + 1]
        if (len(cur) <= max_words and before[0].speaker == after[0].speaker
                and not before[-1].text.endswith(SENTENCE_END)):
            for w in cur:
                w.speaker = before[0].speaker
    return words


def clean_words(t: dict) -> list[Word]:
    return smooth_flips(clean_speakers(words_from_transcript(t)))


def speaker_names(t: dict) -> dict[int, str | None]:
    """Names for the CLEANED speaker labels (0, 1): each cleaned label takes the name of the
    transcript speaker most of its words came from. None where the provider gave no name."""
    original = [int(w["speaker"]) for w in t["words"]]
    cleaned = [w.speaker for w in clean_words(t)]
    names: dict[int, str | None] = {}
    for label in sorted(set(cleaned)):
        source = Counter(o for o, c in zip(original, cleaned, strict=True) if c == label).most_common(1)[0][0]
        names[label] = (t.get("speakers") or {}).get(str(source))
    return names


def display_names(t: dict) -> dict[int, str]:
    """speaker_names with a 'Speaker N' fallback where the provider gave no name."""
    return {k: v or f"Speaker {k + 1}" for k, v in speaker_names(t).items()}


def host_roles(labels, host: int | None) -> dict[int, str] | None:
    """{label: 'host' | 'guest'} given which label is the host; None if unknown."""
    if host is None:
        return None
    return {int(lbl): "host" if int(lbl) == host else "guest" for lbl in labels}


def speaker_turns(t: dict) -> list[dict]:
    """Cleaned diarized transcript as display-ready turns."""
    return [{"speaker": turn[0].speaker, "start": turn[0].start, "end": turn[-1].end,
             "text": " ".join(w.text for w in turn)}
            for turn in turns(clean_words(t))]
