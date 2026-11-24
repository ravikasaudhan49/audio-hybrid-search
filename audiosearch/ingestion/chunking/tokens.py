"""Token counting with tiktoken. The encoder is loaded once per process (Singleton via lru_cache)."""
from functools import lru_cache

from .params import CHUNKING


@lru_cache(maxsize=4)
def encoder(name: str = CHUNKING.tokenizer):
    import tiktoken
    return tiktoken.get_encoding(name)


def count_tokens(text: str) -> int:
    return len(encoder().encode_ordinary(text))


def count_word_tokens(words: list[str]) -> list[int]:
    """Tokens per word as it appears mid-sentence (with its leading space)."""
    return [len(ids) for ids in encoder().encode_ordinary_batch([" " + w for w in words])]
