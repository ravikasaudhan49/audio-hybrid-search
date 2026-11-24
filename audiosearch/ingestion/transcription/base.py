"""Transcriber interface: speech-to-text with diarization, behind one provider-neutral format.

Every provider returns its raw response from `request()` (cached by the store as
data/transcripts/<id>.<provider>.json) and converts it with `normalize()` into
data/transcripts/<id>.transcript.json:

  {"provider": "assemblyai", "duration": 180.0,
   "speakers": {"0": "Gary Jordan", "1": "Patrick O'Neill"},   # names may be null
   "words": [{"text": "Houston,", "start": 0.51, "end": 0.96, "speaker": 0, "confidence": 0.99}, ...]}

Speaker numbers follow order of first appearance. Nothing downstream reads raw responses
(CLAUDE.md rule 6), so the provider can change without touching chunking, search or the UI.
`normalize` is a staticmethod: re-normalizing a cached response needs no API key.
"""
from abc import ABC, abstractmethod
from pathlib import Path

CONTENT_TYPES = {".mp3": "audio/mpeg", ".wav": "audio/wav", ".m4a": "audio/mp4"}


def number_speakers(labels: list) -> dict:
    """Provider speaker labels -> 0, 1, ... in order of first appearance."""
    order: dict = {}
    for label in labels:
        order.setdefault(label, len(order))
    return order


class Transcriber(ABC):
    name: str

    def __init__(self, api_key: str):
        self._api_key = api_key

    @abstractmethod
    async def request(self, file_id: str, audio_path: Path, keyterms: list[str]) -> dict:
        """Call the provider; returns its raw response."""

    @staticmethod
    @abstractmethod
    def normalize(raw: dict) -> dict:
        """Raw provider response -> provider-neutral transcript."""
