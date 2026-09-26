"""Query understanding: detect WHO a query is about, so "what did Jordan say about X" searches
only Gary Jordan's words for X. No API calls: names come from the speakers table.

  "what did Jordan say about stem cells"   -> speaker="Gary Jordan", text="stem cells"
  "what did the guest say about trust"     -> role="guest",          text="trust"
  "Kranz on accountability"                -> speaker="Gene Kranz",  text="accountability"
  "guest list"                             -> nothing detected (roles need "the guest" / "guest's")

A full name always matches. A single first or last name (3+ letters) only matches when the
query uses it as a speaker: "Jordan's", "what did Gene ...", "Kranz on/said/about ...",
"according to Patrick", because such names are also ordinary words ("gene expression").
If a partial name fits several different people, no speaker is detected (ambiguous).
Rankers get the stripped text; the reranker still sees the full query, where the name helps
it read "Speaker: ..." context.
"""
import re
from dataclasses import dataclass

ROLE_WORDS = {"host": "host", "interviewer": "host", "presenter": "host",
              "guest": "guest", "interviewee": "guest"}
# Framing around the speaker that carries no topic: "what did X say about", "X's views on" ...
FRAMING = re.compile(
    r"\b(what|how|why|when|did|does|do|has|have|was|is|the|a|say|says|said|saying|talk|talks|talked|"
    r"talking|mention|mentions|mentioned|tell|tells|told|think|thinks|thought|explain|explains|explained|"
    r"ask|asks|asked|asking|"
    r"describe|describes|described|discuss|discusses|discussed|views?|opinions?|take|thoughts?|"
    r"according|to|about|on|regarding|re|of)\b", re.IGNORECASE)


@dataclass
class ParsedQuery:
    text: str                    # what the rankers search for
    speaker: str | None = None   # full speaker name to filter on
    role: str | None = None      # host | guest
    matched: str | None = None   # the words in the query that identified the speaker/role


def _name_forms(name: str) -> list[str]:
    parts = [p for p in re.split(r"[\s\-]+", name) if len(p.strip(".'")) >= 3]
    return [name, *parts]


def _find(term: str, q: str) -> re.Match | None:
    return re.search(rf"(?<![\w']){re.escape(term)}(?:'s|’s)?(?![\w'])", q, re.IGNORECASE)


# A single first/last name is also an ordinary word ("gene expression", "Jordan River"), so it
# only counts as a speaker when the query uses it as one: possessive, "what did X ...",
# "X said/on/about ...", "according to X".
SPEECH_VERBS = (r"say|says|said|saying|talk|talks|talked|talking|mention|mentions|mentioned|think|thinks|"
                r"thought|explain|explains|explained|describe|describes|described|discuss|discusses|"
                r"discussed|ask|asks|asked|tell|tells|told|on|about|regarding")


def _as_speaker(part: str, q: str) -> re.Match | None:
    name = re.escape(part)
    patterns = (rf"(?<![\w']){name}(?:'s|’s)(?![\w'])",
                rf"\b(?:what|how|why|when|did|does|do|has|have)\s+(?:\w+\s+)?{name}(?![\w'])",
                rf"(?<![\w']){name}\s+(?:{SPEECH_VERBS})\b",
                rf"\baccording\s+to\s+{name}(?![\w'])")
    for pattern in patterns:
        if re.search(pattern, q, re.IGNORECASE):
            return _find(part, q)
    return None


def parse(q: str, speakers: list[str]) -> ParsedQuery:
    """Detect a known speaker name or a host/guest role in `q`."""
    speaker = matched = role = None
    # Full names first, then single name parts; a part must point to exactly one person.
    for name in sorted(set(speakers), key=len, reverse=True):
        if (m := _find(name, q)):
            speaker, matched = name, m.group(0)
            break
    if speaker is None:
        hits: dict[str, str] = {}
        for name in set(speakers):
            for part in _name_forms(name)[1:]:
                if (m := _as_speaker(part, q)):
                    hits[name] = m.group(0)
        if len(hits) == 1:
            speaker, matched = next(iter(hits.items()))
    if speaker is None:
        # Roles need "the guest" / "guest's", so topics like "guest list" aren't read as a role.
        for word, r in ROLE_WORDS.items():
            pattern = rf"\bthe {word}(?:'s|’s)?(?![\w'])|\b{word}(?:'s|’s)(?![\w'])"
            if (m := re.search(pattern, q, re.IGNORECASE)):
                role, matched = r, m.group(0)
                break
    if matched is None:
        return ParsedQuery(q)
    topic = re.sub(r"\s+", " ", FRAMING.sub(" ", q.replace(matched, " "))).strip(" ?.,!")
    # Nothing left but framing ("what did Kranz say?"): search the whole question within the filter.
    return ParsedQuery(topic or q, speaker, role, matched)
