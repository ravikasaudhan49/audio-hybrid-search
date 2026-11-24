import time

from fastapi import APIRouter, HTTPException, Query

from audiosearch.core.log import get
from audiosearch.evaluation import golden_set, is_hit
from audiosearch.models import METHODS, SearchRequest

from ..dependencies import CollectionDep, ContainerDep
from ..schemas import AnswerOut, SearchResponse, SearchResult

router = APIRouter(tags=["search"])
log = get("backend.api")


@router.get("/search", response_model=SearchResponse)
async def search(app: ContainerDep, collection: CollectionDep, q: str, method: str = "hybrid",
                 k: int = Query(10, ge=1, le=50),
                 speaker: str | None = None, file_id: str | None = None,
                 rerank: bool | None = Query(None, description="Rerank top candidates (Cohere by default); "
                                                               "default: on"),
                 mmr: bool = Query(False, description="Diversify results with Maximal Marginal Relevance"),
                 mmr_lambda: float | None = Query(None, ge=0, le=1,
                                                  description="1.0 = pure relevance, 0.0 = pure diversity"),
                 group: bool = Query(True, description="One result per parent chunk (best-matching child)"),
                 role: str | None = Query(None, pattern="^(host|guest)$", description="Only this speaker role"),
                 auto_speaker: bool = Query(True, description="Detect 'what did <name>/the guest say' in q"),
                 threshold: bool = Query(True, description="Hide results below the relevance threshold"),
                 answer: bool = Query(False, description="Add a grounded summary answer (Gemini) above the results"),
                 origin: str | None = Query(None, pattern="^(golden|upload)$",
                                            description="Only golden (evaluation) or uploaded files")):
    if method not in METHODS:
        raise HTTPException(400, f"method must be one of {list(METHODS)}")
    t0 = time.perf_counter()
    out = await app.searcher(collection, origin).search(SearchRequest(
        q, method, k, speaker, file_id, rerank=rerank, mmr=mmr, mmr_lambda=mmr_lambda, group=group, role=role,
        auto_speaker=auto_speaker, threshold=threshold))
    summary = None
    if answer and out.results:
        t_answer = time.perf_counter()
        try:
            a = await app.answers.generate(q, out.results)
            summary = AnswerOut(text=a.text, model=a.model, citations=a.citations, cached=a.cached) if a else None
        except Exception as e:  # the results are still useful without a summary
            log.warning("answer generation failed: %s: %s", type(e).__name__, e)
            out.notes.append(f"summary answer unavailable: {type(e).__name__}")
        out.timings_ms["answer"] = round((time.perf_counter() - t_answer) * 1000, 1)
    took = (time.perf_counter() - t0) * 1000
    spans = golden_set().relevant_spans(q)
    return SearchResponse(
        query=q, method=method, embedder=app.settings.embedder, took_ms=round(took, 1), golden_query=spans is not None,
        reranked=out.reranked, reranker=out.reranker, mmr=out.mmr, grouped=out.grouped, notes=out.notes,
        search_text=out.search_text, speaker_detected=out.speaker_detected, role_detected=out.role_detected,
        below_threshold=out.below_threshold, answer=summary, timings_ms=out.timings_ms,
        results=[SearchResult.from_result(r, collection,
                                          None if spans is None else any(is_hit(r, x) for x in spans))
                 for r in out.results])


@router.get("/queries", tags=["evaluation"])
async def golden_queries():
    """The labeled evaluation queries (id, query text, type, relevant spans)."""
    return golden_set().queries()
