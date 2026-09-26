"""Speech-to-text with diarization, behind one provider-neutral transcript format.

Providers (TRANSCRIBER in .env):
  assemblyai  Universal model + speaker labels + Speaker Identification: names are inferred
              from the conversation ("I'm Gary Jordan", "welcome back, Patrick") with no
              voice enrollment, so results can say who spoke. Default.
  deepgram    Nova-3 + diarization: anonymous speaker numbers only.

Every raw provider response is cached as data/transcripts/<id>.<provider>.json and
normalized to data/transcripts/<id>.transcript.json:

  {"provider": "assemblyai", "duration": 180.0,
   "speakers": {"0": "Gary Jordan", "1": "Patrick O'Neill"},   # names may be null
   "words": [{"text": "Houston,", "start": 0.51, "end": 0.96, "speaker": 0, "confidence": 0.99}, ...]}

Speaker numbers follow order of first appearance. Nothing downstream reads raw responses,
so the provider can change without touching chunking, search or the UI.
"""
import asyncio
import json
from pathlib import Path

from . import aio, config
from .log import get, timed

log = get(__name__)

DEEPGRAM_URL = "https://api.deepgram.com/v1/listen"
DEEPGRAM_PARAMS = {
    "model": "nova-3",
    "language": "en",
    "diarize": "true",
    "utterances": "true",
    "smart_format": "true",
    "punctuate": "true",
    "utt_split": "0.8",
}
ASSEMBLYAI_URL = "https://api.assemblyai.com/v2"
ASSEMBLYAI_POLL_S = 3
CONTENT_TYPES = {".mp3": "audio/mpeg", ".wav": "audio/wav", ".m4a": "audio/mp4"}
PROVIDERS = ("assemblyai", "deepgram")


# ---------- cache + normalized format ----------

def transcript_path(file_id: str) -> Path:
    return config.TRANSCRIPT_DIR / f"{file_id}.transcript.json"


def raw_path(file_id: str, provider: str) -> Path:
    return config.TRANSCRIPT_DIR / f"{file_id}.{provider}.json"


def has_transcript(file_id: str) -> bool:
    return transcript_path(file_id).exists() or any(raw_path(file_id, p).exists() for p in PROVIDERS)


def _save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1), encoding="utf-8")


def load_transcript(file_id: str) -> dict:
    """Normalized transcript; built (locally, no API call) from a cached raw response if needed."""
    path = transcript_path(file_id)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    for provider in PROVIDERS:
        raw = raw_path(file_id, provider)
        if raw.exists():
            t = NORMALIZERS[provider](json.loads(raw.read_text(encoding="utf-8")))
            _save(path, t)
            log.info("normalized cached %s transcript file=%s", provider, file_id)
            return t
    raise FileNotFoundError(f"no transcript for {file_id}")


def _number_speakers(labels: list) -> dict:
    """Provider speaker labels -> 0, 1, ... in order of first appearance."""
    order: dict = {}
    for label in labels:
        order.setdefault(label, len(order))
    return order


def normalize_deepgram(resp: dict) -> dict:
    words = resp["results"]["channels"][0]["alternatives"][0]["words"]
    order = _number_speakers([w.get("speaker", 0) for w in words])
    return {
        "provider": "deepgram",
        "duration": resp.get("metadata", {}).get("duration"),
        "speakers": {str(i): None for i in order.values()},  # diarization only: no names
        "words": [{"text": w.get("punctuated_word") or w["word"], "start": float(w["start"]),
                   "end": float(w["end"]), "speaker": order[w.get("speaker", 0)],
                   "confidence": w.get("confidence")} for w in words],
    }


def normalize_assemblyai(resp: dict) -> dict:
    words = resp.get("words") or []
    order = _number_speakers([w.get("speaker") for w in words])
    identified = (resp.get("speech_understanding") or {}).get("response", {}) \
        .get("speaker_identification", {}).get("status") == "success"
    return {
        "provider": "assemblyai",
        "duration": resp.get("audio_duration"),
        # With Speaker Identification the labels ARE names; otherwise they are "A", "B".
        "speakers": {str(i): (label if identified else None) for label, i in order.items()},
        "words": [{"text": w["text"], "start": w["start"] / 1000, "end": w["end"] / 1000,
                   "speaker": order[w.get("speaker")], "confidence": w.get("confidence")} for w in words],
    }


NORMALIZERS = {"deepgram": normalize_deepgram, "assemblyai": normalize_assemblyai}


def to_seconds(t: str | float) -> float:
    """'00:17:30' / '17:30' / 1050 -> 1050.0"""
    if isinstance(t, int | float):
        return float(t)
    secs = 0.0
    for part in str(t).split(":"):
        secs = secs * 60 + float(part)
    return secs


def slice_transcript(source_id: str, dest_id: str, start: str | float, duration: float,
                     names: dict | None = None) -> dict:
    """Cut a clip's transcript out of an already-transcribed full recording (no API call).
    Keeps words fully inside [start, start + duration), shifts times so the clip starts at 0,
    renumbers speakers by first appearance in the clip and carries names across.
    `names` pins names by the SOURCE transcript's speaker numbers, e.g. {"0": "Host"}."""
    src = load_transcript(source_id)
    start_s = to_seconds(start)
    end_s = start_s + float(duration)
    words = [w for w in src["words"] if w["start"] >= start_s and w["end"] <= end_s]
    if not words:
        raise ValueError(f"no words in {source_id} between {start_s:.0f}s and {end_s:.0f}s")
    source_names = {**(src.get("speakers") or {}), **{str(k): v for k, v in (names or {}).items()}}
    order = _number_speakers([w["speaker"] for w in words])
    t = {
        "provider": src["provider"],
        "duration": float(duration),
        "speakers": {str(i): source_names.get(str(orig)) for orig, i in order.items()},
        "sliced_from": {"id": source_id, "start_s": start_s, "end_s": end_s},
        "words": [{**w, "start": round(w["start"] - start_s, 3), "end": round(w["end"] - start_s, 3),
                   "speaker": order[w["speaker"]]} for w in words],
    }
    _save(transcript_path(dest_id), t)
    log.info("sliced transcript %s -> %s [%.0fs, %.0fs) words=%d speakers=%s", source_id, dest_id,
             start_s, end_s, len(words), t["speakers"])
    return t


# ---------- providers ----------

async def _deepgram(file_id: str, audio_path: Path, keyterms: list[str]) -> dict:
    if not config.DEEPGRAM_API_KEY:
        raise RuntimeError("DEEPGRAM_API_KEY is not set in .env")
    # Keyterm prompting boosts rare names/jargon, which are exactly what users search for.
    params = list(DEEPGRAM_PARAMS.items()) + [("keyterm", k) for k in keyterms]
    body = await asyncio.to_thread(audio_path.read_bytes)
    log.info("deepgram request file=%s bytes=%d keyterms=%d", file_id, len(body), len(keyterms))
    with timed(log, "deepgram transcription", file=file_id):
        resp = await aio.http().post(
            DEEPGRAM_URL, params=params, content=body, timeout=600,
            headers={"Authorization": f"Token {config.DEEPGRAM_API_KEY}",
                     "Content-Type": CONTENT_TYPES.get(audio_path.suffix.lower(), "application/octet-stream")})
        if resp.status_code >= 400:
            log.error("deepgram error file=%s status=%d body=%s", file_id, resp.status_code, resp.text[:300])
        resp.raise_for_status()
    return resp.json()


async def _assemblyai(file_id: str, audio_path: Path, keyterms: list[str]) -> dict:
    if not config.ASSEMBLYAI_API_KEY:
        raise RuntimeError("ASSEMBLYAI_API_KEY is not set in .env")
    http, headers = aio.http(), {"authorization": config.ASSEMBLYAI_API_KEY}
    body = await asyncio.to_thread(audio_path.read_bytes)
    with timed(log, "assemblyai transcription", file=file_id):
        up = await http.post(f"{ASSEMBLYAI_URL}/upload", headers=headers, content=body, timeout=600)
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
        r = await http.post(f"{ASSEMBLYAI_URL}/transcript", headers=headers, json=request, timeout=60)
        if r.status_code >= 400:
            log.error("assemblyai error file=%s status=%d body=%s", file_id, r.status_code, r.text[:300])
        r.raise_for_status()
        tid = r.json()["id"]
        while True:
            resp = (await http.get(f"{ASSEMBLYAI_URL}/transcript/{tid}", headers=headers, timeout=60)).json()
            if resp["status"] in ("completed", "error"):
                break
            await asyncio.sleep(ASSEMBLYAI_POLL_S)
        if resp["status"] == "error":
            log.error("assemblyai failed file=%s: %s", file_id, resp.get("error"))
            raise RuntimeError(f"AssemblyAI transcription failed: {resp.get('error')}")
    sid = (resp.get("speech_understanding") or {}).get("response", {}).get("speaker_identification", {})
    log.info("assemblyai speaker identification file=%s status=%s mapping=%s",
             file_id, sid.get("status"), sid.get("mapping"))
    return resp


PROVIDER_CALLS = {"deepgram": _deepgram, "assemblyai": _assemblyai}


async def transcribe(file_id: str, audio_path: Path, keyterms: list[str] | None = None,
                     force: bool = False, provider: str | None = None) -> dict:
    """Normalized transcript for `file_id`; calls the provider only when nothing is cached."""
    if has_transcript(file_id) and not force:
        log.info("transcript cache hit file=%s", file_id)
        return await asyncio.to_thread(load_transcript, file_id)
    provider = provider or config.TRANSCRIBER
    if provider not in PROVIDER_CALLS:
        raise ValueError(f"unknown transcriber {provider!r}; use one of {PROVIDERS}")
    raw = await PROVIDER_CALLS[provider](file_id, audio_path, keyterms or [])
    await asyncio.to_thread(_save, raw_path(file_id, provider), raw)
    t = NORMALIZERS[provider](raw)
    await asyncio.to_thread(_save, transcript_path(file_id), t)
    log.info("transcript file=%s provider=%s duration=%.1fs words=%d speakers=%s", file_id, provider,
             t["duration"] or 0, len(t["words"]), t["speakers"])
    return t
