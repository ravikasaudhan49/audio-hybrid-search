"""How providers are chosen and handed to the services (no paid API calls, no DB)."""
import pytest

from audiosearch.core import runtime
from audiosearch.core.errors import MissingApiKeyError, RerankUnavailable
from audiosearch.embeddings.gemini import GeminiEmbedder
from audiosearch.retrieval.rerankers.cohere import CohereReranker


def test_gemini_embedder_starts_without_key_and_fails_on_first_call():
    # Break caught: the key check moving back into __init__ would stop the app starting without keys.
    embedder = GeminiEmbedder("")
    with pytest.raises(MissingApiKeyError):
        runtime.run(embedder.embed_query("stem cells"))


def test_cohere_reranker_starts_without_key_and_is_unavailable_on_use(tmp_path):
    # Break caught: a missing key at startup must not crash; search falls back on RerankUnavailable.
    reranker = CohereReranker("", "rerank-v4.0-pro", tmp_path)
    with pytest.raises(RerankUnavailable):
        runtime.run(reranker.scores("q", ["doc"]))


def test_transcript_store_normalizes_a_cached_raw_response_offline(tmp_path):
    # Break caught: a provider missing from NORMALIZERS, or loading needing a transcriber/key.
    from audiosearch.ingestion.transcription.store import NORMALIZERS, TranscriptStore
    store = TranscriptStore(tmp_path)
    store.save_raw("ep1", "deepgram", {"metadata": {"duration": 1.0}, "results": {"channels": [{"alternatives": [
        {"words": [{"word": "hi", "punctuated_word": "Hi.", "start": 0.0, "end": 0.4, "speaker": 1}]}]}]}})
    t = store.load("ep1")
    assert t["provider"] == "deepgram" and t["words"][0]["text"] == "Hi." and t["words"][0]["speaker"] == 0
    assert store.exists("ep1")                      # normalized form is written next to the raw one
    assert set(NORMALIZERS) == {"assemblyai", "deepgram"}


def test_container_builds_each_provider_once_from_settings():
    # Break caught: a wrong branch (e.g. 'local' building Cohere) or a provider ignored from .env.
    import dataclasses

    from audiosearch.container import AppContainer
    from audiosearch.core.settings import get_settings
    s = dataclasses.replace(get_settings(), embedder="gemini", rerank_provider="local", transcriber="deepgram")
    app = AppContainer(s)
    assert app.embedder.name == "gemini-embedding-001"
    assert app.reranker.name == "bge-reranker-base"
    assert app.transcriber.name == "deepgram"


@pytest.mark.parametrize("field,value", [("embedder", "gemni"), ("rerank_provider", "cohre"),
                                         ("transcriber", "whisper")])
def test_container_rejects_a_misspelled_provider_name(field, value):
    # Break caught: a typo in .env silently leaving the provider as None until first use.
    import dataclasses

    from audiosearch.container import AppContainer
    from audiosearch.core.errors import UnknownProviderError
    from audiosearch.core.settings import get_settings
    with pytest.raises(UnknownProviderError, match=value):
        AppContainer(dataclasses.replace(get_settings(), **{field: value}))


class FakeEmbedder:
    """Stands in for a real embedder; only its identity matters in these tests."""
    name, dim, min_similarity = "fake-wiring", 3, 0.0


def test_container_uses_an_embedder_it_is_given_instead_of_building_one():
    # Break caught: the container ignoring the injected embedder (tests would then hit Gemini).
    import dataclasses

    from audiosearch.container import AppContainer
    from audiosearch.core.settings import get_settings
    fake = FakeEmbedder()
    app = AppContainer(dataclasses.replace(get_settings(), embedder="not-a-model"), embedder=fake)
    assert app.embedder is fake            # and the unknown .env name was never built


def test_only_the_gemini_embedder_is_supported():
    # Break caught: a local model (BGE, ~400 MB of weights) creeping back in as a choice.
    import dataclasses

    from audiosearch.container import AppContainer
    from audiosearch.core.errors import UnknownProviderError
    from audiosearch.core.settings import get_settings
    for name in ("bge-base", "bge-small"):
        with pytest.raises(UnknownProviderError, match="use gemini"):
            AppContainer(dataclasses.replace(get_settings(), embedder=name))


def test_search_service_reranks_with_the_reranker_it_is_given():
    # Break caught: SearchService looking up its own reranker instead of using the one it was given.
    from audiosearch.models import Result, SearchOutput
    from audiosearch.retrieval import SearchService
    from audiosearch.retrieval.rerankers import Reranker

    class Reverse(Reranker):
        name = "reverse-test"

        async def _scores(self, query, docs, patient):
            return [0.1, 0.5, 0.9]          # the last candidate is the best

    out = SearchOutput([Result(i, "f", "t", "a", "s", 0, 1, 0, "x", "x", 1.0, {}, f"f:p{i:02d}", 0, 1, f"p{i}")
                        for i in range(3)], grouped=True)
    service = SearchService(pool=None, embedder=None, reranker=Reverse(), speakers=object())
    runtime.run(service._rerank("q", out, patient=False))
    assert [r.segment_id for r in out.results] == [2, 1, 0]
    assert out.reranked and out.reranker == "reverse-test" and out.results[0].score == 0.9


def test_ingestion_transcribes_with_the_transcriber_it_is_given_and_only_once(tmp_path):
    # Break caught: calling the paid provider again for a cached file, or ignoring the given transcriber.
    from audiosearch.ingestion import IngestionService
    from audiosearch.ingestion.transcription import Transcriber, TranscriptStore

    class CountingTranscriber(Transcriber):
        name = "fake-asr"

        def __init__(self):
            super().__init__(api_key="unused")
            self.requests = 0

        async def request(self, file_id, audio_path, keyterms):
            self.requests += 1
            return {"words": [["Hello.", 0.0, 0.5]]}

        @staticmethod
        def normalize(raw):
            text, start, end = raw["words"][0]
            return {"provider": "fake-asr", "duration": end, "speakers": {"0": None},
                    "words": [{"text": text, "start": start, "end": end, "speaker": 0}]}

    asr = CountingTranscriber()
    store = TranscriptStore(tmp_path)
    ingestion = IngestionService(pool=None, embedder=FakeEmbedder(), transcriber=asr, transcripts=store)
    first = runtime.run(ingestion.transcribe("ep1", tmp_path / "ep1.mp3", ["Kranz"]))
    second = runtime.run(ingestion.transcribe("ep1", tmp_path / "ep1.mp3"))
    assert asr.requests == 1                                   # second call came from the cache
    assert first == second and first["words"][0]["text"] == "Hello."
    assert store.raw_path("ep1", "fake-asr").exists()          # raw response cached under the provider's name
