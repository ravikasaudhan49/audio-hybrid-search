"""One handler per CLI command. Each receives the parsed arguments; async work runs through
`runtime.run` inside an `AppContainer` (which opens and closes the pool and HTTP client)."""
import argparse
import re
import subprocess
import sys
import textwrap
from pathlib import Path

from ..container import AppContainer
from ..core import runtime
from ..core.progress import PrintReporter
from ..core.settings import get_settings
from ..core.text import snippet, split_terms, upload_file_id
from ..db import EmbeddingRepository, collection_conn
from ..ingestion import Manifest
from ..ingestion.chunking import clean_speakers, display_names, speaker_names, turns, words_from_transcript
from ..ingestion.transcription import TranscriptStore
from ..models import SearchRequest

ROOT = get_settings().root
API_CMD = [sys.executable, "-m", "uvicorn", "backend.main:app", "--port", "8000",
           "--loop", "asyncio:SelectorEventLoop"]
UI_CMD = [sys.executable, "-m", "streamlit", "run", "frontend/app.py"]


# ---------- search ----------

async def _search(a: argparse.Namespace) -> None:
    async with AppContainer.open(min_size=1, max_size=4) as app:
        collection = await app.resolve_collection(a.collection)
        out = await app.searcher(collection).search(SearchRequest(
            a.query, a.method, a.k, a.speaker, a.file, rerank=a.rerank, mmr=a.mmr, mmr_lambda=a.mmr_lambda,
            group=a.group, role=a.role, auto_speaker=a.auto_speaker, threshold=a.threshold))
    if out.speaker_detected or out.role_detected:
        print(f"detected {out.speaker_detected or 'the ' + out.role_detected}: searching {out.search_text!r} "
              "in their words only")
    if out.below_threshold:
        print(f"{out.below_threshold} low-relevance result(s) hidden")
    for note in out.notes:
        print(f"note: {note}")
    if out.reranked:
        print(f"reranked by {out.reranker}" + (" · MMR applied" if out.mmr else ""))
    if not out.results:
        print("no results")
    for i, r in enumerate(out.results, 1):
        print(f"{i}. {r.file_id} · {r.file_title}  [{r.timestamp}]  {r.speaker}   {r.audio_url}\n   "
              f"(score {r.score:.4f}, via {r.sources}, parent {r.parent_id}"
              + (f", {r.grouped} children matched" if r.grouped > 1 else "") + ")")
        print(textwrap.indent(textwrap.fill(snippet(r.highlighted), 100), "     "))
        if a.context:
            print(textwrap.indent(r.parent_text, "       | "))


def search(a: argparse.Namespace) -> None:
    runtime.run(_search(a))


# ---------- doctor ----------

async def _doctor() -> None:
    """Environment check for a fresh checkout: database, keys, data, indexes."""
    s = get_settings()
    ok = True

    def line(good: bool, what: str, hint: str = "") -> None:
        nonlocal ok
        ok &= good
        print(f"  {'OK ' if good else 'FIX'}  {what}" + (f"  ->  {hint}" if not good and hint else ""))

    print("audiosearch doctor")
    keys = {"GEMINI_API_KEY": (s.gemini_api_key, "query/document embeddings (EMBEDDER=gemini) and summary answers"),
            "COHERE_API_KEY": (s.cohere_api_key, "reranking (or set RERANK_PROVIDER=local)"),
            "ASSEMBLYAI_API_KEY": (s.assemblyai_api_key,
                                   "transcribing NEW audio with speaker names (TRANSCRIBER=assemblyai)")}
    for key, (value, why) in keys.items():
        line(bool(value), f"{key} set", f"add it to .env: {why}")
    line(s.manifest_path.exists(), "golden manifest data/manifest.json")
    line(s.queries_path.exists() or s.golden_spec_path.exists(), "golden query set")
    app = AppContainer(s)
    try:
        await app.start(min_size=1, max_size=2)
    except Exception as e:  # database unreachable
        line(False, "Postgres + pgvector reachable", f"docker compose up -d ({type(e).__name__})")
        raise SystemExit(1) from None
    try:
        line(True, "Postgres + pgvector reachable")
        for c in await app.collections.list_all():
            async with collection_conn(app.pool, c["name"]) as conn:
                emb = {m: n for m, n, _ in await EmbeddingRepository(conn).stats()}
            line(True, f"collection {c['name']!r}: {c['files']} files, {c['segments']} chunks, "
                       f"embeddings {emb or 'none'}")
    finally:
        await app.close()
    print("ready" if ok else "fix the items marked FIX, then re-run: python -m audiosearch doctor")


def doctor(a: argparse.Namespace) -> None:
    runtime.run(_doctor())


# ---------- collections ----------

async def _collections(a: argparse.Namespace) -> None:
    async with AppContainer.open(min_size=1, max_size=2) as app:
        if a.action == "create":
            if not a.name:
                raise SystemExit("usage: collections create NAME [--description TEXT]")
            await app.collections.create(a.name, a.description)
            print(f"collection {a.name!r} ready")
        for c in await app.collections.list_all():
            print(f"{c['name']:20s} files={c['files']:<4} chunks={c['segments']:<6} {c['description'] or ''}")


def collections(a: argparse.Namespace) -> None:
    runtime.run(_collections(a))


# ---------- ingest / add ----------

async def _ingest(a: argparse.Namespace) -> None:
    s = get_settings()
    manifest = Manifest(s.manifest_path)
    # The manifest may name its collection; --collection overrides; else the default.
    collection = a.collection or manifest.collection or s.default_collection
    async with AppContainer.open() as app:
        ingestion = app.ingestion(collection)
        entries = manifest.select(a.only)
        print(f"ingesting {len(entries)} file(s) into collection {collection!r}")
        await ingestion.ingest_manifest(entries, a.rechunk, on_file=lambda fid, n: print(f"{fid}: {n} segments"))
        print(f"{s.embedder}: embedded {await ingestion.embed_missing()} new segments")


def ingest(a: argparse.Namespace) -> None:
    runtime.run(_ingest(a))


async def _add(a: argparse.Namespace) -> None:
    """Ingest one audio file end to end (the same service as the UI upload)."""
    src = Path(a.file)
    if not src.exists():
        raise SystemExit(f"error: no such file {src}")
    fid = upload_file_id(src.read_bytes())
    async with AppContainer.open(min_size=1, max_size=4) as app:
        collection = await app.resolve_collection(a.collection)
        ingestion = app.ingestion(collection)
        reporter = PrintReporter()
        print(f"{src.name} -> {fid}")
        t, _ = await ingestion.transcribe_upload(fid, src, split_terms(a.keyterms), reporter)
        names = display_names(t)
        print(f"   speakers: {names} · host: speaker {a.host}")
        n, _ = await ingestion.index_upload(fid, a.title or src.stem, names, a.host,
                                            audio_url=app.settings.audio_url_base + src.name,
                                            reporter=reporter, transcript=t)
        print(f"done: {fid} ({n} chunks) is searchable -> "
              f"python -m audiosearch search \"...\" --collection {collection}")


def add(a: argparse.Namespace) -> None:
    runtime.run(_add(a))


# ---------- preview ----------

def preview(a: argparse.Namespace) -> None:
    """First turns per speaker label, to decide which label is which person."""
    t = TranscriptStore(get_settings().transcript_dir).load(a.file_id)
    raw = words_from_transcript(t)
    print(f"provider: {t['provider']} · raw speaker labels: {sorted({w.speaker for w in raw})}"
          f" · identified names: {speaker_names(t)}")
    shown: dict[int, int] = {}
    for turn in turns(clean_speakers(raw)):
        spk = turn[0].speaker
        if shown.get(spk, 0) >= 3 or len(turn) < 8:
            continue
        shown[spk] = shown.get(spk, 0) + 1
        text = re.sub(r"\s+", " ", " ".join(w.text for w in turn))[:220]
        print(f"speaker {spk} @ {turn[0].start:6.1f}s: {text}")


# ---------- evaluation / tooling ----------

def evaluate(a: argparse.Namespace) -> None:
    from eval import build_golden
    from eval import evaluate as evaluate_script
    if build_golden.main():
        raise SystemExit("error: golden labels could not be built")
    runtime.run(evaluate_script.amain(a.rerank))


def bench(a: argparse.Namespace) -> None:
    from eval import bench as bench_script
    bench_script.main(["--url", a.url, "--concurrency", str(a.concurrency), "--rounds", str(a.rounds)])


def report(a: argparse.Namespace) -> None:
    from eval import report as report_script
    raise SystemExit(report_script.main())


def test(a: argparse.Namespace) -> None:
    raise SystemExit(subprocess.call([sys.executable, "-m", "pytest", "-q"], cwd=ROOT))


def serve(a: argparse.Namespace) -> None:
    """Run the FastAPI backend and the Streamlit frontend together; Ctrl+C stops both."""
    procs = [subprocess.Popen(API_CMD, cwd=ROOT), subprocess.Popen(UI_CMD, cwd=ROOT)]
    try:
        for proc in procs:
            proc.wait()
    except KeyboardInterrupt:
        for proc in procs:
            proc.terminate()
