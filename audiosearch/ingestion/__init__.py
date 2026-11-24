"""Ingestion pipeline: audio -> transcript -> chunks -> Postgres -> embeddings.

  pipeline.py      IngestionService (the steps, in order) and Manifest
  audio.py         ffmpeg: any audio -> 16 kHz mono mp3
  transcription/   AssemblyAI / Deepgram, and the on-disk transcript store
  chunking/        speaker clean-up and parent/child chunks
"""
from .pipeline import IngestionService, Manifest

__all__ = ["IngestionService", "Manifest"]
