"""Grounded summary answer over the retrieved chunks (Gemini Flash-Lite, see Settings.answer_model).

The model sees only the top results (each with its speaker, episode timestamp and the parent
dialogue around it) and must answer from them alone, citing results as [1], [2] ... . If the
excerpts don't answer the question it says so instead of guessing. No results (e.g. all below
the relevance threshold) -> no model call at all. Answers are cached on disk per
(model, question, result set), so repeating a query is free.
"""
import re

from ..core import runtime
from ..core.cache import JsonDiskCache
from ..core.errors import MissingApiKeyError
from ..core.log import get, timed
from ..core.settings import Settings, get_settings
from ..models import Answer, Result

log = get(__name__)

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
NO_ANSWER = "The recordings don't cover this."
SYSTEM = f"""You answer questions about podcast recordings using ONLY the numbered excerpts provided.
Rules:
- Answer in 2-4 sentences, plainly. Attribute claims to the speaker by name.
- Cite the excerpts you used as [1], [2] right after the claim they support.
- Use only facts stated in the excerpts. Do not add outside knowledge.
- If the excerpts do not answer the question, reply exactly: "{NO_ANSWER}"
"""


def build_context(results: list[Result]) -> str:
    blocks = []
    for i, r in enumerate(results, 1):
        blocks.append(f"[{i}] {r.file_title} · {r.speaker} at {r.timestamp}\n"
                      f"Matched passage: {r.child_text}\nSurrounding dialogue:\n{r.parent_text}")
    return "\n\n".join(blocks)


def parse_citations(text: str, n: int) -> list[int]:
    """[n] markers in `text` that point at one of the n excerpts, deduplicated and sorted."""
    return sorted({int(m) for m in re.findall(r"\[(\d+)\]", text) if 1 <= int(m) <= n})


class AnswerService:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._cache = JsonDiskCache(self.settings.cache_dir / "answers")

    async def generate(self, question: str, results: list[Result], top_n: int | None = None,
                       model: str | None = None) -> Answer | None:
        """Summary answer grounded in `results`, or None when there is nothing to answer from."""
        model = model or self.settings.answer_model
        results = results[: top_n or self.settings.answer_top_n]
        if not results:
            return None
        if not self.settings.gemini_api_key:
            raise MissingApiKeyError("GEMINI_API_KEY is not set in .env")
        key = JsonDiskCache.key_of(model, question, [r.segment_id for r in results])
        if (hit := self._cache.get(key)) is not None:
            log.info("answer cache hit model=%s", model)
            return Answer(**{**hit, "cached": True})

        body = {
            "systemInstruction": {"parts": [{"text": SYSTEM}]},
            "contents": [{"role": "user", "parts": [{"text": f"Excerpts:\n\n{build_context(results)}\n\n"
                                                              f"Question: {question}"}]}],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 400},
        }
        with timed(log, "answer generation", model=model, excerpts=len(results)):
            r = await runtime.http().post(GEMINI_URL.format(model=model), json=body, timeout=60,
                                          headers={"x-goog-api-key": self.settings.gemini_api_key})
            if r.status_code >= 400:
                log.error("gemini answer error status=%d body=%s", r.status_code, r.text[:300])
            r.raise_for_status()
        parts = r.json()["candidates"][0]["content"].get("parts", [])
        text = "".join(p.get("text", "") for p in parts).strip() or NO_ANSWER
        answer = Answer(text=text, model=model, citations=parse_citations(text, len(results)))
        self._cache.put(key, {"text": answer.text, "model": model, "citations": answer.citations})
        return answer
