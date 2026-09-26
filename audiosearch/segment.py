"""Parent-child chunking of a two-speaker transcript (provider-neutral format, see transcribe.py).

  1. words      every word with timestamps, speaker label and tiktoken count
  2. speakers   exactly two per file: stray labels are folded into the neighbouring
                main speaker; 1-2 word "turns" that interrupt a sentence of the other
                speaker are diarization flips and are relabeled
  3. turns      consecutive same-speaker words
  4. parents    whole turns packed to ~PARENT_TOKENS, so a question and its answer share
                a parent (a single turn longer than that is split at sentence ends)
  5. children   ~CHILD_TOKENS windows inside one speaker's turn within a parent, cut at a
                sentence end when possible, overlapping by ~CHILD_OVERLAP_TOKENS.
                Turns under MIN_CHILD_WORDS (backchannels) get no child; they remain in
                the parent text.

A child never spans two speakers or two parents, so every search hit has one speaker, a
tight timestamp and a parent to show as context.
"""
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache

from . import config

SENTENCE_END = (".", "?", "!")


@lru_cache(maxsize=1)
def _encoder():
    import tiktoken
    return tiktoken.get_encoding(config.TOKENIZER)


def count_tokens(text: str) -> int:
    return len(_encoder().encode_ordinary(text))


@dataclass
class Word:
    text: str
    start: float
    end: float
    speaker: int
    tokens: int = 1


@dataclass
class Parent:
    index: int
    turns: list[list[Word]]          # whole turns (or sentence-split pieces of a very long one)

    @property
    def start(self) -> float:
        return self.turns[0][0].start

    @property
    def end(self) -> float:
        return self.turns[-1][-1].end

    @property
    def tokens(self) -> int:
        return sum(w.tokens for t in self.turns for w in t)

    def render(self, names: dict[int, str] | None = None) -> str:
        """Dialogue text, one line per turn: 'Host: ...\\nGuest: ...'."""
        names = names or {}
        return "\n".join(f"{names.get(t[0].speaker, f'Speaker {t[0].speaker}')}: {' '.join(w.text for w in t)}"
                         for t in self.turns)


@dataclass
class Child:
    parent: int                      # Parent.index
    speaker: int
    words: list[Word] = field(default_factory=list)

    @property
    def start(self) -> float:
        return self.words[0].start

    @property
    def end(self) -> float:
        return self.words[-1].end

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)

    @property
    def tokens(self) -> int:
        return sum(w.tokens for w in self.words)


# ---------- words + speakers ----------

def words_from_transcript(t: dict) -> list[Word]:
    words = [Word(w["text"], float(w["start"]), float(w["end"]), int(w["speaker"])) for w in t["words"]]
    # Leading space: tokens as the word appears mid-sentence.
    for w, ids in zip(words, _encoder().encode_ordinary_batch([" " + w.text for w in words]), strict=True):
        w.tokens = len(ids)
    return words


def clean_speakers(words: list[Word], n_speakers: int = 2) -> list[Word]:
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


def turns(words: list[Word]) -> list[list[Word]]:
    out: list[list[Word]] = []
    for w in words:
        if out and out[-1][-1].speaker == w.speaker:
            out[-1].append(w)
        else:
            out.append([w])
    return out


# ---------- parents ----------

def _split_long_turn(turn: list[Word], limit: int) -> list[list[Word]]:
    """Split a turn longer than `limit` tokens at sentence ends (hard cut if needed)."""
    pieces, cur, tok = [], [], 0
    for w in turn:
        cur.append(w)
        tok += w.tokens
        if tok >= limit or (tok >= limit * 0.8 and w.text.endswith(SENTENCE_END)):
            pieces.append(cur)
            cur, tok = [], 0
    if cur:
        pieces.append(cur)
    return pieces


def build_parents(words: list[Word]) -> list[Parent]:
    parents: list[Parent] = []
    cur: list[list[Word]] = []
    cur_tok = 0

    def close():
        nonlocal cur, cur_tok
        if cur:
            parents.append(Parent(len(parents), cur))
        cur, cur_tok = [], 0

    for turn in turns(words):
        tok = sum(w.tokens for w in turn)
        if tok > config.PARENT_TOKENS:
            close()
            for piece in _split_long_turn(turn, config.PARENT_TOKENS):
                parents.append(Parent(len(parents), [piece]))
            continue
        if cur and cur_tok + tok > config.PARENT_TOKENS:
            close()
        cur.append(turn)
        cur_tok += tok
    close()
    return parents


# ---------- children ----------

def _windows(piece: list[Word]) -> list[list[Word]]:
    """Token windows over one speaker's words: cut at CHILD_TOKENS, or at a sentence end
    once 75% full; the next window starts CHILD_OVERLAP_TOKENS back."""
    out: list[list[Word]] = []
    i, n = 0, len(piece)
    while i < n:
        j, tok = i, 0
        while j < n:
            tok += piece[j].tokens
            j += 1
            if tok >= config.CHILD_TOKENS or (
                    tok >= 0.75 * config.CHILD_TOKENS and piece[j - 1].text.endswith(SENTENCE_END)):
                break
        out.append(piece[i:j])
        if j >= n:
            break
        k, overlap = j, 0   # step back over whole words while within the overlap budget
        while k - 1 > i and overlap + piece[k - 1].tokens <= config.CHILD_OVERLAP_TOKENS:
            k -= 1
            overlap += piece[k].tokens
        i = k
    # A tiny tail (only its non-overlapping words count) folds into the previous window.
    if len(out) > 1:
        prev_ids = {id(w) for w in out[-2]}
        new = [w for w in out[-1] if id(w) not in prev_ids]
        if sum(w.tokens for w in new) < config.CHILD_MIN_TOKENS:
            out[-2:] = [out[-2] + new]
    return out


def build_children(parents: list[Parent]) -> list[Child]:
    return [Child(p.index, piece[0].speaker, window)
            for p in parents
            for piece in p.turns if len(piece) >= config.MIN_CHILD_WORDS
            for window in _windows(piece)]


# ---------- entry points ----------

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


def chunk_transcript(t: dict) -> tuple[list[Parent], list[Child]]:
    parents = build_parents(clean_words(t))
    return parents, build_children(parents)


def speaker_turns(t: dict) -> list[dict]:
    """Cleaned diarized transcript as display-ready turns."""
    return [{"speaker": turn[0].speaker, "start": turn[0].start, "end": turn[-1].end,
             "text": " ".join(w.text for w in turn)}
            for turn in turns(clean_words(t))]
