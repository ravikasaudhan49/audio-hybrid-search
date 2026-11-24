"""AppContainer: builds everything the app needs ONCE, then hands it to the services.

When the app starts, the container reads `.env` and creates:
  - the embedder     (EMBEDDER=gemini)
  - the reranker     (RERANK_PROVIDER=cohere | local)
  - the transcriber  (TRANSCRIBER=assemblyai | deepgram)
  - the answer service, the transcript store and (on start) the database pool.

Services never choose a provider themselves; they get these objects in their constructor.
The API keeps one container on `app.state.container`; each CLI command and tool opens one:

    async with AppContainer.open(max_size=4) as app:
        out = await app.searcher(collection="nasa").search(SearchRequest("..."))

Tests pass their own objects instead, e.g. `AppContainer(settings, embedder=FakeEmbedder())`.
"""
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from psycopg_pool import AsyncConnectionPool

from .core import runtime
from .core.errors import CollectionError, UnknownProviderError
from .core.log import get
from .core.settings import Settings, get_settings
from .db import CollectionRepository, Database, schema_name
from .embeddings import Embedder
from .embeddings.gemini import GeminiEmbedder
from .evaluation import Evaluator
from .ingestion import IngestionService
from .ingestion.transcription import AssemblyAITranscriber, DeepgramTranscriber, Transcriber, TranscriptStore
from .retrieval import SearchService, SpeakerNameCache
from .retrieval.answer import AnswerService
from .retrieval.rerankers import Reranker
from .retrieval.rerankers.cohere import CohereReranker
from .retrieval.rerankers.local import LocalReranker

log = get(__name__)


# ---------- the one place that turns a name from .env into an object ----------

def build_embedder(name: str, s: Settings) -> Embedder:
    if name == "gemini":
        return GeminiEmbedder(s.gemini_api_key, s.embed_concurrency, s.cache_dir, s.vector_min_sim)
    raise UnknownProviderError(f"unknown embedder {name!r}; use gemini")


def build_reranker(s: Settings) -> Reranker:
    if s.rerank_provider == "cohere":
        return CohereReranker(s.cohere_api_key, s.cohere_rerank_model, s.cache_dir)
    if s.rerank_provider == "local":
        return LocalReranker(s.cache_dir)
    raise UnknownProviderError(f"unknown rerank provider {s.rerank_provider!r}; use cohere | local")


def build_transcriber(s: Settings) -> Transcriber:
    if s.transcriber == "assemblyai":
        return AssemblyAITranscriber(s.assemblyai_api_key)
    if s.transcriber == "deepgram":
        return DeepgramTranscriber(s.deepgram_api_key)
    raise UnknownProviderError(f"unknown transcriber {s.transcriber!r}; use assemblyai | deepgram")


class AppContainer:
    def __init__(self, settings: Settings | None = None, *, embedder: Embedder | None = None):
        """embedder: use this one instead of building EMBEDDER from .env (tests pass a fake)."""
        self.settings = settings or get_settings()
        # Chosen once from .env, used everywhere.
        self.embedder = embedder or build_embedder(self.settings.embedder, self.settings)
        self.reranker = build_reranker(self.settings)
        self.transcriber = build_transcriber(self.settings)
        self.database = Database(self.settings)
        self.transcripts = TranscriptStore(self.settings.transcript_dir)
        self.answers = AnswerService(self.settings)
        self._pool: AsyncConnectionPool | None = None
        self._speakers: SpeakerNameCache | None = None

    # ---------- lifecycle ----------

    async def start(self, min_size: int | None = None, max_size: int | None = None) -> "AppContainer":
        self._pool = await self.database.open_pool(min_size, max_size)
        self._speakers = SpeakerNameCache(self._pool)
        return self

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
        await runtime.close_http()

    @classmethod
    @asynccontextmanager
    async def open(cls, settings: Settings | None = None, min_size: int | None = None,
                   max_size: int | None = None) -> AsyncIterator["AppContainer"]:
        app = cls(settings)
        await app.start(min_size, max_size)
        try:
            yield app
        finally:
            await app.close()

    @property
    def pool(self) -> AsyncConnectionPool:
        if self._pool is None:
            raise RuntimeError("AppContainer is not started; use `async with AppContainer.open()`")
        return self._pool

    # ---------- collections ----------

    @property
    def collections(self) -> CollectionRepository:
        return CollectionRepository(self.pool)

    def collection_or_default(self, name: str | None) -> str:
        return name or self.settings.default_collection

    async def resolve_collection(self, name: str | None) -> str:
        """Validated, existing collection (default if None). Raises CollectionError."""
        name = self.collection_or_default(name)
        schema_name(name)
        if not await self.collections.exists(name):
            raise CollectionError(f"unknown collection {name!r}")
        return name

    # ---------- services (cheap, per use) ----------

    def searcher(self, collection: str | None = None, origin: str | None = None) -> SearchService:
        return SearchService(self.pool, embedder=self.embedder, reranker=self.reranker,
                             collection=collection, origin=origin, settings=self.settings, speakers=self._speakers)

    def ingestion(self, collection: str | None = None) -> IngestionService:
        return IngestionService(self.pool, embedder=self.embedder, transcriber=self.transcriber,
                                collection=collection, settings=self.settings, transcripts=self.transcripts)

    def evaluator(self, collection: str | None = None) -> Evaluator:
        return Evaluator(self, collection)
