"""Build data/queries.json (labeled spans) from data/golden_spec.json (queries + anchors).

For each query, the anchor phrase (copied verbatim from the clip transcript) is located in the
clip's words. The relevant span is the ANSWER PASSAGE around it: the same speaker's turn,
limited to PASSAGE_S seconds either side of the anchor (about one child chunk). It is converted
to EPISODE time (the clip's offset in the source episode) and carries the speaker's name.
Phrase queries ("...") accept ANY place in the clip where the exact phrase is said ("match":
"any"). Labels are reproducible from the transcript instead of hand-typed timestamps
(ADR-0008). No API calls.

(v1 used only the anchor's own 2-3 seconds; answers found a sentence later in the same turn were
scored as misses.)
"""
import json
from dataclasses import dataclass, field

from ..core.settings import Settings, get_settings
from ..core.text import norm_word
from ..ingestion.chunking import CHUNKING, clean_words, speaker_names
from ..ingestion.transcription import TranscriptStore

PASSAGE_S = 30.0   # answer passage: anchor +/- this much, within the same speaker turn


@dataclass
class LabelReport:
    queries: list[dict]
    problems: list[str] = field(default_factory=list)

    @property
    def by_type(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for q in self.queries:
            out[q["type"]] = out.get(q["type"], 0) + 1
        return out

    @property
    def ok(self) -> bool:
        return not any("not found" in p for p in self.problems)


def _occurrences(words, phrase: str) -> tuple[list[int], int]:
    """Start indexes of every exact occurrence of `phrase`, and its length in words."""
    target = [x for x in (norm_word(w) for w in phrase.split()) if x]
    norm = [norm_word(w.text) for w in words]
    return [i for i in range(len(norm) - len(target) + 1) if norm[i:i + len(target)] == target], len(target)


def _passage(words, i: int, n: int) -> tuple[int, int]:
    """Expand words[i:i+n] to the surrounding same-speaker passage, at most PASSAGE_S each side.
    A backchannel from the other speaker ("Right.", "Yeah." - fewer than min_child_words words)
    doesn't end the passage, mirroring the chunker, which gives backchannels no chunk of their own."""
    spk, a_start, a_end = words[i].speaker, words[i].start, words[i + n - 1].end

    def backchannel(j: int, step: int) -> int:
        """Length of an other-speaker run starting at j (going `step`) if it is a backchannel, else 0."""
        k = j
        while 0 <= k < len(words) and words[k].speaker != spk:
            k += step
        run = abs(k - j)
        return run if run < CHUNKING.min_child_words and 0 <= k < len(words) else 0

    lo, hi = i, i + n - 1
    while lo > 0 and words[lo - 1].start >= a_start - PASSAGE_S:
        if words[lo - 1].speaker == spk:
            lo -= 1
        elif (b := backchannel(lo - 1, -1)):
            lo -= b
        else:
            break
    while hi < len(words) - 1 and words[hi + 1].end <= a_end + PASSAGE_S:
        if words[hi + 1].speaker == spk:
            hi += 1
        elif (b := backchannel(hi + 1, 1)):
            hi += b
        else:
            break
    return lo, hi


class GoldenLabelBuilder:
    def __init__(self, settings: Settings | None = None, transcripts: TranscriptStore | None = None):
        self.settings = settings or get_settings()
        self.transcripts = transcripts or TranscriptStore(self.settings.transcript_dir)

    def locate(self, file_id: str, phrase: str, every: bool = False) -> list[dict]:
        """Answer passage(s) for `phrase` in a clip: the first occurrence, or all when every=True."""
        t = self.transcripts.load(file_id)
        words = clean_words(t)
        names = speaker_names(t)
        offset = (t.get("sliced_from") or {}).get("start_s", 0.0)
        hits, n = _occurrences(words, phrase)
        if not hits:
            raise ValueError(f"not found in {file_id}: {phrase!r}")
        out = []
        for i in hits if every else hits[:1]:
            lo, hi = _passage(words, i, n)
            out.append({"file": file_id,
                        "start": round(words[lo].start + offset, 2), "end": round(words[hi].end + offset, 2),
                        "speaker": names.get(words[i].speaker), "anchor": phrase})
        return out

    def build(self) -> LabelReport:
        """Label every spec query and write data/queries.json."""
        spec = json.loads(self.settings.golden_spec_path.read_text(encoding="utf-8"))
        report = LabelReport([])
        for q in spec["queries"]:
            item = {k: q[k] for k in ("id", "type", "query")}
            if q["type"] == "negative":
                item["relevant"] = []
            else:
                try:
                    item["relevant"] = self.locate(q["file"], q["anchor"])
                    if q["type"] == "phrase":
                        # Any place the exact quoted phrase is said counts as a correct hit.
                        quoted = q["query"].strip().strip('"')
                        extra = [r for r in self.locate(q["file"], quoted, every=True)
                                 if not any(abs(r["start"] - x["start"]) < 1 for x in item["relevant"])]
                        if extra:
                            item["relevant"] += extra
                            item["match"] = "any"
                except ValueError as e:
                    report.problems.append(f"{q['id']}: {e}")
                    continue
            report.queries.append(item)
        self.settings.queries_path.write_text(
            json.dumps({"collection": spec.get("collection"), "queries": report.queries},
                       indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        return report
