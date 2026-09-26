"""Grounded summary answer over the retrieved chunks (Gemini Flash-Lite, see config.ANSWER_MODEL).

The model sees only the top results (each with its speaker, episode timestamp and the parent
dialogue around it) and must answer from them alone, citing results as [1], [2] ... . If the
excerpts don't answer the question it says so instead of guessing. No results (e.g. all below
the relevance threshold) -> no model call at all. Answers are cached on disk per
(model, question, result set), so repeating a query is free.
"""
import hashlib
import json
import re
from dataclasses import dataclass, field

from . import aio, config
from .log import get, timed
from .search import Result

log = get(__name__)

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
SYSTEM = """You answer questions about podcast recordings using ONLY the numbered excerpts provided.
Rules:
- Answer in 2-4 sentences, plainly. Attribute claims to the speaker by name.
- Cite the excerpts you used as [1], [2] right after the claim they support.
- Use only facts stated in the excerpts. Do not add outside knowledge.
- If the excerpts do not answer the question, reply exactly: "The recordings don't cover this."
"""


@dataclass
class Answer:
    text: str
    model: str
    citations: list[int] = field(default_factory=list)   # 1-based result numbers cited
    cached: bool = False


def _context(results: list[Result]) -> str:
    blocks = []
    for i, r in enumerate(results, 1):
        blocks.append(f"[{i}] {r.file_title} · {r.speaker} at {r.timestamp}\n"
                      f"Matched passage: {r.child_text}\nSurrounding dialogue:\n{r.parent_text}")
    return "\n\n".join(blocks)


def _citations(text: str, n: int) -> list[int]:
    return sorted({int(m) for m in re.findall(r"\[(\d+)\]", text) if 1 <= int(m) <= n})


async def generate_answer(question: str, results: list[Result], top_n: int | None = None,
                          model: str | None = None) -> Answer | None:
    """Summary answer grounded in `results`, or None when there is nothing to answer from."""
    model = model or config.ANSWER_MODEL
    results = results[: top_n or config.ANSWER_TOP_N]
    if not results:
        return None
    if not config.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set in .env")
    key = hashlib.sha1(json.dumps([model, question, [r.segment_id for r in results]]).encode()).hexdigest()
    path = config.CACHE_DIR / "answers" / f"{key}.json"
    if path.exists():
        log.info("answer cache hit model=%s", model)
        return Answer(**{**json.loads(path.read_text(encoding="utf-8")), "cached": True})

    body = {
        "systemInstruction": {"parts": [{"text": SYSTEM}]},
        "contents": [{"role": "user", "parts": [{"text": f"Excerpts:\n\n{_context(results)}\n\n"
                                                          f"Question: {question}"}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 400},
    }
    with timed(log, "answer generation", model=model, excerpts=len(results)):
        r = await aio.http().post(GEMINI_URL.format(model=model), json=body, timeout=60,
                                  headers={"x-goog-api-key": config.GEMINI_API_KEY})
        if r.status_code >= 400:
            log.error("gemini answer error status=%d body=%s", r.status_code, r.text[:300])
        r.raise_for_status()
    parts = r.json()["candidates"][0]["content"].get("parts", [])
    text = "".join(p.get("text", "") for p in parts).strip() or "The recordings don't cover this."
    answer = Answer(text=text, model=model, citations=_citations(text, len(results)))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"text": answer.text, "model": model, "citations": answer.citations}),
                    encoding="utf-8")
    return answer
