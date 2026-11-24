"""AudioNormalizer: convert (and optionally trim) any ffmpeg-readable audio to 16 kHz mono mp3.

ffmpeg runs in a worker thread: asyncio subprocesses are unavailable on Windows' selector loop.
"""
import asyncio
import subprocess
from pathlib import Path

from ..core.log import get, timed

log = get(__name__)


class AudioNormalizer:
    SAMPLE_RATE = 16000
    BITRATE = "64k"

    def _ffmpeg(self, src: Path, out: Path, start=None, duration=None) -> None:
        import imageio_ffmpeg
        out.parent.mkdir(parents=True, exist_ok=True)
        trim = (["-ss", str(start)] if start is not None else []) + \
               (["-t", str(duration)] if duration is not None else [])
        subprocess.run(
            [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", *trim, "-i", str(src),
             "-ac", "1", "-ar", str(self.SAMPLE_RATE), "-b:a", self.BITRATE, str(out)],
            check=True,
        )

    async def normalize(self, src: Path, out: Path, start=None, duration=None) -> None:
        with timed(log, "ffmpeg normalize", src=src.name, out=out.name, start=start, duration=duration):
            await asyncio.to_thread(self._ffmpeg, src, out, start, duration)
