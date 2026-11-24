"""TranscriptStore: the on-disk transcript cache (raw provider responses + normalized form).

  data/transcripts/<id>.<provider>.json     raw response, exactly as the provider sent it
  data/transcripts/<id>.transcript.json     provider-neutral transcript (see base.py)
"""
import json
from pathlib import Path

from ...core.log import get
from ...core.text import to_seconds
from .assemblyai import AssemblyAITranscriber
from .base import number_speakers
from .deepgram import DeepgramTranscriber

log = get(__name__)

# Provider name -> its normalize function. Normalizing a cached raw response needs no API key.
NORMALIZERS = {
    AssemblyAITranscriber.name: AssemblyAITranscriber.normalize,
    DeepgramTranscriber.name: DeepgramTranscriber.normalize,
}


class TranscriptStore:
    def __init__(self, directory: Path):
        self.directory = directory

    def path(self, file_id: str) -> Path:
        return self.directory / f"{file_id}.transcript.json"

    def raw_path(self, file_id: str, provider: str) -> Path:
        return self.directory / f"{file_id}.{provider}.json"

    def exists(self, file_id: str) -> bool:
        """A normalized transcript is on disk (not just a raw response)."""
        return self.path(file_id).exists()

    def has(self, file_id: str) -> bool:
        """A transcript is available without an API call (normalized or raw)."""
        return self.exists(file_id) or any(self.raw_path(file_id, p).exists() for p in NORMALIZERS)

    @staticmethod
    def _write(path: Path, data: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=1), encoding="utf-8")

    def save(self, file_id: str, transcript: dict) -> None:
        self._write(self.path(file_id), transcript)

    def save_raw(self, file_id: str, provider: str, raw: dict) -> None:
        self._write(self.raw_path(file_id, provider), raw)

    def load(self, file_id: str) -> dict:
        """Normalized transcript; built (locally, no API call) from a cached raw response if needed."""
        path = self.path(file_id)
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        for provider, normalize in NORMALIZERS.items():
            raw = self.raw_path(file_id, provider)
            if raw.exists():
                t = normalize(json.loads(raw.read_text(encoding="utf-8")))
                self.save(file_id, t)
                log.info("normalized cached %s transcript file=%s", provider, file_id)
                return t
        raise FileNotFoundError(f"no transcript for {file_id}")

    def slice(self, source_id: str, dest_id: str, start: str | float, duration: float,
              names: dict | None = None) -> dict:
        """Cut a clip's transcript out of an already-transcribed full recording (no API call).
        Keeps words fully inside [start, start + duration), shifts times so the clip starts at 0,
        renumbers speakers by first appearance in the clip and carries names across.
        `names` pins names by the SOURCE transcript's speaker numbers, e.g. {"0": "Host"}."""
        src = self.load(source_id)
        start_s = to_seconds(start)
        end_s = start_s + float(duration)
        words = [w for w in src["words"] if w["start"] >= start_s and w["end"] <= end_s]
        if not words:
            raise ValueError(f"no words in {source_id} between {start_s:.0f}s and {end_s:.0f}s")
        source_names = {**(src.get("speakers") or {}), **{str(k): v for k, v in (names or {}).items()}}
        order = number_speakers([w["speaker"] for w in words])
        t = {
            "provider": src["provider"],
            "duration": float(duration),
            "speakers": {str(i): source_names.get(str(orig)) for orig, i in order.items()},
            "sliced_from": {"id": source_id, "start_s": start_s, "end_s": end_s},
            "words": [{**w, "start": round(w["start"] - start_s, 3), "end": round(w["end"] - start_s, 3),
                       "speaker": order[w["speaker"]]} for w in words],
        }
        self.save(dest_id, t)
        log.info("sliced transcript %s -> %s [%.0fs, %.0fs) words=%d speakers=%s", source_id, dest_id,
                 start_s, end_s, len(words), t["speakers"])
        return t
