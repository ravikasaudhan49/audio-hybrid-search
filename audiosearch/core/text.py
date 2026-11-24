"""Small text and time helpers used by several layers (one copy each)."""
import hashlib
import re


def fmt_ts(s: float | None) -> str:
    """mm:ss, or h:mm:ss past an hour (episodes can run over 60 minutes)."""
    h, rem = divmod(int(s or 0), 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"


def to_seconds(t: str | float) -> float:
    """'00:17:30' / '17:30' / 1050 -> 1050.0"""
    if isinstance(t, int | float):
        return float(t)
    secs = 0.0
    for part in str(t).split(":"):
        secs = secs * 60 + float(part)
    return secs


def norm_word(w: str) -> str:
    """Lowercase alphanumerics only: how words are compared for highlighting and anchors."""
    return re.sub(r"[^a-z0-9]", "", w.lower())


def snippet(highlighted: str, width: int = 40) -> str:
    """Trim long chunk text to a window of words around the first [[highlight]]."""
    words = highlighted.split()
    first = next((i for i, w in enumerate(words) if w.startswith("[[")), 0)
    lo = max(0, first - width // 2)
    out = " ".join(words[lo: lo + width])
    return ("… " if lo else "") + out + (" …" if lo + width < len(words) else "")


def upload_file_id(data: bytes) -> str:
    """Content-derived id of an uploaded file: the same bytes always get the same id, so a
    re-upload reuses its cached transcript."""
    return "up_" + hashlib.sha1(data).hexdigest()[:8]


def split_terms(csv: str) -> list[str]:
    """'a, b,,c' -> ['a', 'b', 'c']"""
    return [t.strip() for t in csv.split(",") if t.strip()]
