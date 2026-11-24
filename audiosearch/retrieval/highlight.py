"""Query terms, the exact time of the first matched word, and [[highlighting]]."""
from ..core.text import norm_word

STOPWORDS = set("""a an the and or but if of to in on at by for with about from into over
is are was were be been being do does did have has had it its this that these those
i you he she we they me him her us them my your his our their what which who whom how
why when where there here not no so as than then too very can could would should will
just also say said tell talk talked talking""".split())


def query_terms(q: str) -> list[str]:
    return [t for t in (norm_word(w) for w in q.split()) if t and t not in STOPWORDS]


def _matches(word: str, term: str) -> bool:
    # Cheap stem-tolerant match: "embeddings" ~ "embedding", "running" ~ "run".
    stem = term[: max(4, len(term) - 2)] if len(term) > 4 else term
    return word == term or (len(stem) >= 4 and word.startswith(stem))


def locate(words: list, terms: list[str], default_s: float) -> tuple[float, str]:
    """Return (time of first matched query word, text with matched words wrapped in [[ ]])."""
    hit = None
    parts = []
    for w, ws, _ in words:
        nw = norm_word(w)
        if nw and any(_matches(nw, t) for t in terms):
            hit = ws if hit is None else hit
            parts.append(f"[[{w}]]")
        else:
            parts.append(w)
    return (hit if hit is not None else default_s), " ".join(parts)
