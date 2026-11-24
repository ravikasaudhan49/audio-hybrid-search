"""Deepgram Nova-3 + diarization: anonymous speaker numbers only (names come from the user)."""
import asyncio
from pathlib import Path
from typing import ClassVar

from ...core import runtime
from ...core.errors import MissingApiKeyError
from ...core.log import get, timed
from .base import CONTENT_TYPES, Transcriber, number_speakers

log = get(__name__)


class DeepgramTranscriber(Transcriber):
    name = "deepgram"
    URL = "https://api.deepgram.com/v1/listen"
    PARAMS: ClassVar[dict[str, str]] = {
        "model": "nova-3",
        "language": "en",
        "diarize": "true",
        "utterances": "true",
        "smart_format": "true",
        "punctuate": "true",
        "utt_split": "0.8",
    }

    async def request(self, file_id: str, audio_path: Path, keyterms: list[str]) -> dict:
        if not self._api_key:
            raise MissingApiKeyError("DEEPGRAM_API_KEY is not set in .env")
        # Keyterm prompting boosts rare names/jargon, which are exactly what users search for.
        params = list(self.PARAMS.items()) + [("keyterm", k) for k in keyterms]
        body = await asyncio.to_thread(audio_path.read_bytes)
        log.info("deepgram request file=%s bytes=%d keyterms=%d", file_id, len(body), len(keyterms))
        with timed(log, "deepgram transcription", file=file_id):
            resp = await runtime.http().post(
                self.URL, params=params, content=body, timeout=600,
                headers={"Authorization": f"Token {self._api_key}",
                         "Content-Type": CONTENT_TYPES.get(audio_path.suffix.lower(), "application/octet-stream")})
            if resp.status_code >= 400:
                log.error("deepgram error file=%s status=%d body=%s", file_id, resp.status_code, resp.text[:300])
            resp.raise_for_status()
        return resp.json()

    @staticmethod
    def normalize(raw: dict) -> dict:
        words = raw["results"]["channels"][0]["alternatives"][0]["words"]
        order = number_speakers([w.get("speaker", 0) for w in words])
        return {
            "provider": "deepgram",
            "duration": raw.get("metadata", {}).get("duration"),
            "speakers": {str(i): None for i in order.values()},  # diarization only: no names
            "words": [{"text": w.get("punctuated_word") or w["word"], "start": float(w["start"]),
                       "end": float(w["end"]), "speaker": order[w.get("speaker", 0)],
                       "confidence": w.get("confidence")} for w in words],
        }
