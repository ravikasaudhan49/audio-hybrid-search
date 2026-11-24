"""Domain models shared by the library, the API and the tools (plain dataclasses, no I/O).

  transcript / chunking   Word, Parent, Child
  query understanding     ParsedQuery
  search                  Filters, SearchRequest, Result, SearchOutput
  answers                 Answer

API request/response shapes live separately in backend/schemas.py (pydantic): these are the
library's own types and know nothing about HTTP.
"""
from dataclasses import dataclass, field
from typing import Literal

from .core.text import fmt_ts

SearchMethod = Literal["hybrid", "keyword", "vector", "fuzzy"]
METHODS: tuple[str, ...] = ("hybrid", "keyword", "vector", "fuzzy")


# ---------- transcript / chunking ----------

@dataclass
class Word:
    text: str
    start: float
    end: float
    speaker: int
    tokens: int = 1


@dataclass
class Parent:
    """~512 tokens of whole turns: the context shown with a hit and the dedup key."""
    index: int
    turns: list[list[Word]]          # whole turns (or sentence-split pieces of a very long one)

    @property
    def start(self) -> float:
        return self.turns[0][0].start

    @property
    def end(self) -> float:
        return self.turns[-1][-1].end

    @property
    def tokens(self) -> int:
        return sum(w.tokens for t in self.turns for w in t)

    def render(self, names: dict[int, str] | None = None) -> str:
        """Dialogue text, one line per turn: 'Host: ...\\nGuest: ...'."""
        names = names or {}
        return "\n".join(f"{names.get(t[0].speaker, f'Speaker {t[0].speaker}')}: {' '.join(w.text for w in t)}"
                         for t in self.turns)


@dataclass
class Child:
    """~128 tokens inside one speaker's turn: the unit that is embedded, indexed and returned."""
    parent: int                      # Parent.index
    speaker: int
    words: list[Word] = field(default_factory=list)

    @property
    def start(self) -> float:
        return self.words[0].start

    @property
    def end(self) -> float:
        return self.words[-1].end

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)

    @property
    def tokens(self) -> int:
        return sum(w.tokens for w in self.words)


# ---------- query understanding ----------

@dataclass
class ParsedQuery:
    text: str                    # what the rankers search for
    speaker: str | None = None   # full speaker name to filter on
    role: str | None = None      # host | guest
    matched: str | None = None   # the words in the query that identified the speaker/role


# ---------- search ----------

@dataclass(frozen=True)
class Filters:
    speaker: str | None = None     # case-insensitive substring of the speaker name
    file_id: str | None = None
    origin: str | None = None      # 'golden' | 'upload'
    role: str | None = None        # 'host' | 'guest'


@dataclass
class SearchRequest:
    """Everything one search needs. `None` means "use the configured default"."""
    q: str
    method: str = "hybrid"
    k: int = 10
    speaker: str | None = None
    file_id: str | None = None
    rerank: bool | None = None        # None -> Settings.rerank_default
    mmr: bool = False
    mmr_lambda: float | None = None   # None -> Settings.mmr_lambda
    group: bool = True                # keep one (best) child per parent
    role: str | None = None
    auto_speaker: bool = True         # detect "what did <name>/the guest say about X"
    threshold: bool = True            # drop results below the reranker relevance cut-off
    patient_rerank: bool = False      # wait out API rate limits (evaluation) instead of falling back


@dataclass
class Result:
    segment_id: int
    file_id: str
    file_title: str
    audio_path: str
    speaker: str
    start_s: float
    end_s: float
    hit_s: float          # exact word time for keyword hits, else segment start
    child_text: str
    highlighted: str      # child_text with matched words wrapped in [[ ]]
    score: float          # fused RRF score, or the reranker's relevance score when reranked
    sources: dict         # stage -> rank (1-based): keyword/fuzzy/vector, then rerank, mmr
    parent_id: str
    parent_start_s: float
    parent_end_s: float
    parent_text: str
    grouped: int = 1      # children of this parent that matched (collapsed into this result)
    audio_url: str | None = None    # the original episode's audio
    source_offset_s: float = 0.0    # where the stored clip starts in that episode

    # start_s / end_s / hit_s are CLIP time (seek position in the stored clip);
    # episode_* are what users see: time in the original episode.
    @property
    def episode_start_s(self) -> float:
        return self.start_s + self.source_offset_s

    @property
    def episode_end_s(self) -> float:
        return self.end_s + self.source_offset_s

    @property
    def episode_hit_s(self) -> float:
        return self.hit_s + self.source_offset_s

    @property
    def parent_episode_start_s(self) -> float:
        return self.parent_start_s + self.source_offset_s

    @property
    def parent_episode_end_s(self) -> float:
        return self.parent_end_s + self.source_offset_s

    @property
    def timestamp(self) -> str:
        """Hit time in the original episode, e.g. '28:51'."""
        return fmt_ts(self.episode_hit_s)


@dataclass
class SearchOutput:
    results: list[Result]
    reranked: bool = False            # True only if the reranker actually ran
    reranker: str | None = None       # model name when reranked
    mmr: bool = False
    grouped: bool = False             # results were deduplicated by parent_id
    search_text: str = ""             # what the rankers searched (query minus a detected speaker)
    speaker_detected: str | None = None
    role_detected: str | None = None
    below_threshold: int = 0          # results dropped for low relevance
    timings_ms: dict = field(default_factory=dict)   # per stage: understand, embed, retrieve, rerank, post, total
    notes: list[str] = field(default_factory=list)   # e.g. "rerank skipped: rate limit"


# ---------- library ----------

@dataclass
class FileRecord:
    """One stored file with its speaker names and chunk count."""
    id: str
    title: str
    origin: str                      # golden | upload
    duration_s: float | None
    audio_url: str | None
    source_offset_s: float
    speakers: list[str]
    segments: int


# ---------- answers ----------

@dataclass
class Answer:
    text: str
    model: str
    citations: list[int] = field(default_factory=list)   # 1-based result numbers cited
    cached: bool = False
