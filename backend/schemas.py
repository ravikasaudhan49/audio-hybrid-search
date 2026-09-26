"""API request/response models (these also drive the OpenAPI docs at /docs)."""
from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    segment_id: int
    file_id: str
    file_title: str
    speaker: str
    start_s: float                   # clip time (seek position in the stored clip)
    end_s: float
    hit_s: float                     # clip time of the matched word
    timestamp: str                   # hit time in the ORIGINAL EPISODE, e.g. "28:51"
    episode_start_s: float           # child span in episode time
    episode_end_s: float
    episode_hit_s: float
    child_text: str                  # the matched child chunk (~128 tokens, one speaker)
    highlighted: str                 # child_text with matched words wrapped in [[ ]]
    score: float                     # RRF score, or reranker relevance when reranked
    sources: dict[str, int]          # stage -> rank, e.g. {"keyword": 1, "vector": 3, "rerank": 1}
    audio_url: str | None            # the original episode's audio (dummy URL for now)
    stream_url: str                  # local clip stream (supports seeking), for the player
    source_offset_s: float           # where the clip starts in the episode
    parent_id: str                   # dedup key; children of one parent share it
    parent_start_s: float
    parent_end_s: float
    parent_episode_start_s: float
    parent_episode_end_s: float
    parent_text: str                 # ~512-token dialogue around the child ("Host: ...\nGuest: ...")
    grouped: int                     # matching children collapsed into this result
    relevant: bool | None = None     # set when the query is a labeled golden query


class AnswerOut(BaseModel):
    text: str                        # grounded summary, citing results as [1], [2] ...
    model: str
    citations: list[int]             # 1-based result numbers the answer cites
    cached: bool


class SearchResponse(BaseModel):
    query: str
    method: str
    embedder: str
    took_ms: float
    golden_query: bool
    reranked: bool                   # True only if the reranker actually ran
    reranker: str | None             # e.g. "rerank-v4.0-pro"
    mmr: bool
    grouped: bool                    # one result per parent_id
    search_text: str                 # query sent to the rankers (minus a detected speaker/role)
    speaker_detected: str | None     # e.g. "Gene Kranz" from "what did Kranz say about ..."
    role_detected: str | None        # "host" | "guest" from "what did the guest say ..."
    below_threshold: int             # results hidden for low relevance
    answer: AnswerOut | None = None  # summary answer (when requested and there are results)
    timings_ms: dict[str, float]     # understand, embed, retrieve, rerank, post, total (+ answer)
    notes: list[str]                 # e.g. "rerank skipped, showing hybrid order: rate limit"
    results: list[SearchResult]


class Turn(BaseModel):
    speaker: int
    speaker_name: str
    start: float
    end: float
    text: str


class FileInfo(BaseModel):
    id: str
    title: str
    origin: str                      # golden | upload
    duration_s: float | None
    speakers: list[str]
    segments: int
    audio_url: str | None            # the original episode's audio (dummy URL for now)
    stream_url: str                  # local clip stream
    source_offset_s: float           # clip start in the episode; turn times below are episode time


class FileDetail(FileInfo):
    turns: list[Turn]


class CollectionInfo(BaseModel):
    name: str
    description: str | None
    created_at: str
    files: int
    segments: int


class CollectionCreate(BaseModel):
    name: str = Field(pattern=r"^[a-z][a-z0-9_]{0,39}$",
                      description="1-40 lowercase letters, digits or '_', starting with a letter")
    description: str | None = None


class Health(BaseModel):
    database: bool
    collection: str                  # the collection the counts below refer to
    collections: list[str]           # all collections
    files: int
    segments: int
    embeddings: dict[str, int]       # model -> vector count
    embedding_dims: dict[str, int]   # model -> vector dimensions
    default_embedder: str
    embedders: list[str]
    reranker: str
    rerank_default: bool
    pool: dict[str, int]             # connection pool stats: size, available, requests waiting


class Job(BaseModel):
    id: str
    kind: str                        # transcribe | index | eval
    status: str                      # running | done | error
    stage: str
    progress: float                  # 0..1
    log: list[str]
    result: dict | None = None
    error: str | None = None


class IndexRequest(BaseModel):
    title: str
    speakers: dict[int, str]         # diarization label -> display name
    host: int | None = 0             # label of the host; the other speaker is the guest
    embedder: str | None = None
    collection: str | None = None    # target collection (default: DEFAULT_COLLECTION)
