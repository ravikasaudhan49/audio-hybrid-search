"""Build data/queries.json (labeled spans) from data/golden_spec.json (queries + anchors).

For each query, the anchor phrase (copied verbatim from the clip transcript) is located in the
clip's words. The relevant span is the ANSWER PASSAGE around it: the same speaker's turn,
limited to PASSAGE_S seconds either side of the anchor (about one child chunk). It is converted
to EPISODE time (the clip's offset in the source episode) and carries the speaker's name.
Phrase queries ("...") accept ANY place in the clip where the exact phrase is said ("match":
"any"). Labels are reproducible from the transcript instead of hand-typed timestamps.
No API calls.

(v1 used only the anchor's own 2-3 seconds; answers found a sentence later in the same turn were
scored as misses. See AGENT_LOG for both results.)

Run:  python -m eval.build_golden
"""
import json
import re
import sys

from audiosearch import config
from audiosearch.segment import clean_words, speaker_names
from audiosearch.transcribe import load_transcript

SPEC = config.DATA_DIR / "golden_spec.json"
PASSAGE_S = 30.0   # answer passage: anchor +/- this much, within the same speaker turn


def _norm(w: str) -> str:
    return re.sub(r"[^a-z0-9]", "", w.lower())


def _occurrences(words, phrase: str) -> tuple[list[int], int]:
    """Start indexes of every exact occurrence of `phrase`, and its length in words."""
    target = [x for x in (_norm(w) for w in phrase.split()) if x]
    norm = [_norm(w.text) for w in words]
    return [i for i in range(len(norm) - len(target) + 1) if norm[i:i + len(target)] == target], len(target)


def _passage(words, i: int, n: int) -> tuple[int, int]:
    """Expand words[i:i+n] to the surrounding same-speaker passage, at most PASSAGE_S each side.
    A backchannel from the other speaker ("Right.", "Yeah." - fewer than MIN_CHILD_WORDS words)
    doesn't end the passage, mirroring the chunker, which gives backchannels no chunk of their own."""
    spk, a_start, a_end = words[i].speaker, words[i].start, words[i + n - 1].end

    def backchannel(j: int, step: int) -> int:
        """Length of an other-speaker run starting at j (going `step`) if it is a backchannel, else 0."""
        k = j
        while 0 <= k < len(words) and words[k].speaker != spk:
            k += step
        run = abs(k - j)
        return run if run < config.MIN_CHILD_WORDS and 0 <= k < len(words) else 0

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


def locate(file_id: str, phrase: str, every: bool = False) -> list[dict]:
    """Answer passage(s) for `phrase` in a clip: the first occurrence, or all when every=True."""
    t = load_transcript(file_id)
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


def main() -> int:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    out, problems = [], []
    for q in spec["queries"]:
        item = {k: q[k] for k in ("id", "type", "query")}
        if q["type"] == "negative":
            item["relevant"] = []
        else:
            try:
                item["relevant"] = locate(q["file"], q["anchor"])
                if q["type"] == "phrase":
                    # Any place the exact quoted phrase is said counts as a correct hit.
                    quoted = q["query"].strip().strip('"')
                    extra = [r for r in locate(q["file"], quoted, every=True)
                             if not any(abs(r["start"] - x["start"]) < 1 for x in item["relevant"])]
                    if extra:
                        item["relevant"] += extra
                        item["match"] = "any"
            except ValueError as e:
                problems.append(f"{q['id']}: {e}")
                continue
        out.append(item)
    config.QUERIES_PATH.write_text(json.dumps({"collection": spec.get("collection"), "queries": out},
                                              indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    by_type: dict[str, int] = {}
    for q in out:
        by_type[q["type"]] = by_type.get(q["type"], 0) + 1
    print(f"wrote {config.QUERIES_PATH.name}: {len(out)} queries {by_type}")
    for p in problems:
        print("  !", p)
    return 1 if any("not found" in p for p in problems) else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
