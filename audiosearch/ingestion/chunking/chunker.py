"""Parent-child chunking of a two-speaker transcript (provider-neutral format).

  1. words      every word with timestamps, speaker label and tiktoken count
  2. speakers   cleaned to exactly two (see speakers.py)
  3. turns      consecutive same-speaker words
  4. parents    whole turns packed to ~parent_tokens, so a question and its answer share
                a parent (a single turn longer than that is split at sentence ends)
  5. children   ~child_tokens windows inside one speaker's turn within a parent, cut at a
                sentence end when possible, overlapping by ~child_overlap_tokens.
                Turns under min_child_words (backchannels) get no child; they remain in
                the parent text.

A child never spans two speakers or two parents, so every search hit has one speaker, a
tight timestamp and a parent to show as context.
"""
from ...models import Child, Parent, Word
from .params import CHUNKING, ChunkingParams
from .speakers import SENTENCE_END, clean_words, turns


class ParentChildChunker:
    def __init__(self, params: ChunkingParams = CHUNKING):
        self.p = params

    # ---------- parents ----------

    @staticmethod
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

    def build_parents(self, words: list[Word]) -> list[Parent]:
        limit = self.p.parent_tokens
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
            if tok > limit:
                close()
                for piece in self._split_long_turn(turn, limit):
                    parents.append(Parent(len(parents), [piece]))
                continue
            if cur and cur_tok + tok > limit:
                close()
            cur.append(turn)
            cur_tok += tok
        close()
        return parents

    # ---------- children ----------

    def _windows(self, piece: list[Word]) -> list[list[Word]]:
        """Token windows over one speaker's words: cut at child_tokens, or at a sentence end
        once 75% full; the next window starts child_overlap_tokens back."""
        size, overlap_budget = self.p.child_tokens, self.p.child_overlap_tokens
        out: list[list[Word]] = []
        i, n = 0, len(piece)
        while i < n:
            j, tok = i, 0
            while j < n:
                tok += piece[j].tokens
                j += 1
                if tok >= size or (tok >= 0.75 * size and piece[j - 1].text.endswith(SENTENCE_END)):
                    break
            out.append(piece[i:j])
            if j >= n:
                break
            k, overlap = j, 0   # step back over whole words while within the overlap budget
            while k - 1 > i and overlap + piece[k - 1].tokens <= overlap_budget:
                k -= 1
                overlap += piece[k].tokens
            i = k
        # A tiny tail (only its non-overlapping words count) folds into the previous window.
        if len(out) > 1:
            prev_ids = {id(w) for w in out[-2]}
            new = [w for w in out[-1] if id(w) not in prev_ids]
            if sum(w.tokens for w in new) < self.p.child_min_tokens:
                out[-2:] = [out[-2] + new]
        return out

    def build_children(self, parents: list[Parent]) -> list[Child]:
        return [Child(p.index, piece[0].speaker, window)
                for p in parents
                for piece in p.turns if len(piece) >= self.p.min_child_words
                for window in self._windows(piece)]

    # ---------- entry point ----------

    def chunk(self, t: dict) -> tuple[list[Parent], list[Child]]:
        parents = self.build_parents(clean_words(t))
        return parents, self.build_children(parents)


_default = ParentChildChunker()


def chunk_transcript(t: dict) -> tuple[list[Parent], list[Child]]:
    """Chunk with the contract parameters."""
    return _default.chunk(t)
