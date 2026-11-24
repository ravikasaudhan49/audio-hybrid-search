"""The chunking contract (CLAUDE.md rule 5, ADR-0003). Sizes are tiktoken cl100k_base tokens.

Child: the retrieval unit (embedded, keyword-indexed, returned with speaker + timestamp).
  ~128 tokens ≈ 40-60 s of speech, always inside ONE speaker turn, 25-token overlap.
Parent: ~512 tokens ≈ 2-3 min of whole turns (question + answer stay together); context
  for reranking/display and the dedup key. 8-10 min files -> ~4-6 parents, ~20-25 children.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class ChunkingParams:
    tokenizer: str = "cl100k_base"
    child_tokens: int = 128
    child_overlap_tokens: int = 25
    child_min_tokens: int = 32         # a smaller tail is merged into the previous child
    parent_tokens: int = 512
    min_child_words: int = 3           # shorter turns ("Yeah.", "Right.") stay in parent context only
    n_speakers: int = 2


CHUNKING = ChunkingParams()
