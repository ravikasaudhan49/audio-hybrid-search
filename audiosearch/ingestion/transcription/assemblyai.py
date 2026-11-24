"""AssemblyAI: Universal model + speaker labels + Speaker Identification. Names are inferred
from the conversation ("I'm Gary Jordan", "welcome back, Patrick") with no voice enrollment,
so results can say who spoke. The default provider (ADR-0002)."""
import asyncio
from pathlib import Path

from ...core import runtime
from ...core.errors import MissingApiKeyError
from ...core.log import get, timed
from .base import Transcriber, number_speakers

log = get(__name__)


class AssemblyAITranscriber(Transcriber):
    name = "assemblyai"
    URL = "https://api.assemblyai.com/v2"
    POLL_S = 3

    async def request(self, file_id: str, audio_path: Path, keyterms: list[str]) -> dict:
        if not self._api_key:
            raise MissingApiKeyError("ASSEMBLYAI_API_KEY is not set in .env")
        http, headers = runtime.http(), {"authorization": self._api_key}
        body = await asyncio.to_thread(audio_path.read_bytes)
        with timed(log, "assemblyai transcription", file=file_id):
            up = await http.post(f"{self.URL}/upload", headers=headers, content=body, timeout=600)
            up.raise_for_status()
            request = {
                "audio_url": up.json()["upload_url"],
                "language_code": "en",
                "speaker_labels": True,
                "speakers_expected": 2,
                # Names are inferred from the conversation itself; none need to be supplied.
                "speech_understanding": {"request": {"speaker_identification": {"speaker_type": "name"}}},
            }
            if keyterms:
                request["keyterms_prompt"] = keyterms
            log.info("assemblyai request file=%s bytes=%d keyterms=%d", file_id, len(body), len(keyterms))
            r = await http.post(f"{self.URL}/transcript", headers=headers, json=request, timeout=60)
            if r.status_code >= 400:
                log.error("assemblyai error file=%s status=%d body=%s", file_id, r.status_code, r.text[:300])
            r.raise_for_status()
            tid = r.json()["id"]
            while True:
                resp = (await http.get(f"{self.URL}/transcript/{tid}", headers=headers, timeout=60)).json()
                if resp["status"] in ("completed", "error"):
                    break
                await asyncio.sleep(self.POLL_S)
            if resp["status"] == "error":
                log.error("assemblyai failed file=%s: %s", file_id, resp.get("error"))
                raise RuntimeError(f"AssemblyAI transcription failed: {resp.get('error')}")
        sid = (resp.get("speech_understanding") or {}).get("response", {}).get("speaker_identification", {})
        log.info("assemblyai speaker identification file=%s status=%s mapping=%s",
                 file_id, sid.get("status"), sid.get("mapping"))
        return resp

    @staticmethod
    def normalize(raw: dict) -> dict:
        words = raw.get("words") or []
        order = number_speakers([w.get("speaker") for w in words])
        identified = (raw.get("speech_understanding") or {}).get("response", {}) \
            .get("speaker_identification", {}).get("status") == "success"
        return {
            "provider": "assemblyai",
            "duration": raw.get("audio_duration"),
            # With Speaker Identification the labels ARE names; otherwise they are "A", "B".
            "speakers": {str(i): (label if identified else None) for label, i in order.items()},
            "words": [{"text": w["text"], "start": w["start"] / 1000, "end": w["end"] / 1000,
                       "speaker": order[w.get("speaker")], "confidence": w.get("confidence")} for w in words],
        }
