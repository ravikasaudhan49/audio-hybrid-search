import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Route test-run logs to logs/test.log so they don't mix with the real API/UI logs.
from audiosearch import log

log.setup("test")
