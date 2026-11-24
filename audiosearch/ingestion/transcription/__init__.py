"""Speech-to-text with diarization (TRANSCRIBER=assemblyai | deepgram), behind one
provider-neutral transcript format. The app builds the transcriber it uses in AppContainer."""
from .assemblyai import AssemblyAITranscriber
from .base import Transcriber, number_speakers
from .deepgram import DeepgramTranscriber
from .store import NORMALIZERS, TranscriptStore

__all__ = ["NORMALIZERS", "AssemblyAITranscriber", "DeepgramTranscriber", "Transcriber", "TranscriptStore",
           "number_speakers"]
