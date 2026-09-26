"""CLI:  python -m audiosearch <command>

  doctor                         check database, API keys, data and indexes
  collections [create NAME [--description TEXT]]   list collections, or create one
  ingest   [--embedders gemini bge-base] [--rechunk] [--only ep01] [--collection NAME]
                                 ingest the golden manifest (data/manifest.json)
  add      FILE [--collection NAME] [--title T] [--host 0|1] [--keyterms "a,b"]
                                 ingest any audio file: transcribe (+ speaker names), chunk, embed
  eval     [--rerank] [--embedders gemini bge-base]   build golden labels + recall@k report
  bench    [--concurrency 5] [--rounds 10] [--url http://127.0.0.1:8000]   latency p50/p90/p99
  test                           run the automated test suite (incl. the recall@k gate)
  report                         write all evidence to test-results/
                                 (eval tables, per-query hits, pytest output, latency)
  preview  ep01                  show first turns per speaker label (to name speakers)
  search   "query" [--method hybrid|keyword|vector|fuzzy] [-k 5] [--speaker NAME] [--file ep01]
                   [--no-rerank] [--mmr [--mmr-lambda 0.7]] [--no-group] [--context]
                   [--role host|guest] [--no-auto-speaker] [--no-threshold]
  serve                          API on :8000 (docs at /docs) + Streamlit UI on :8501
"""
import argparse
import re
import subprocess
import sys
import textwrap
from pathlib import Path

from . import aio, config, db
from . import log as logs

ROOT = Path(__file__).resolve().parent.parent
API_CMD = [sys.executable, "-m", "uvicorn", "backend.api:app", "--port", "8000",
           "--loop", "asyncio:SelectorEventLoop"]
UI_CMD = [sys.executable, "-m", "streamlit", "run", "frontend/app.py"]


def main():
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252
    p = argparse.ArgumentParser(prog="audiosearch")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("ingest")
    pi.add_argument("--embedders", nargs="*", help="gemini bge-base bge-small (default: EMBEDDER from .env)")
    pi.add_argument("--rechunk", action="store_true")
    pi.add_argument("--only")
    pi.add_argument("--collection", help="target collection (default: manifest 'collection' or DEFAULT_COLLECTION)")

    pc = sub.add_parser("collections")
    pc.add_argument("action", nargs="?", choices=["list", "create"], default="list")
    pc.add_argument("name", nargs="?")
    pc.add_argument("--description")

    pp = sub.add_parser("preview")
    pp.add_argument("file_id")

    ps = sub.add_parser("search")
    ps.add_argument("query")
    ps.add_argument("--method", default="hybrid", choices=["hybrid", "keyword", "vector", "fuzzy"])
    ps.add_argument("-k", type=int, default=5)
    ps.add_argument("--speaker")
    ps.add_argument("--file")
    ps.add_argument("--embedder")
    ps.add_argument("--no-rerank", dest="rerank", action="store_false", help="skip the reranker (on by default)")
    ps.add_argument("--mmr", action="store_true", help="diversify results with Maximal Marginal Relevance")
    ps.add_argument("--mmr-lambda", type=float, default=0.7, help="1.0 = pure relevance, 0.0 = pure diversity")
    ps.add_argument("--no-group", dest="group", action="store_false", help="don't collapse children by parent")
    ps.add_argument("--context", action="store_true", help="also print each result's parent dialogue")
    ps.add_argument("--role", choices=["host", "guest"])
    ps.add_argument("--collection", help="collection to search (default: DEFAULT_COLLECTION)")
    ps.add_argument("--no-auto-speaker", dest="auto_speaker", action="store_false",
                    help="don't detect 'what did <name>/the guest say' in the query")
    ps.add_argument("--no-threshold", dest="threshold", action="store_false",
                    help="keep low-relevance results")

    sub.add_parser("serve")
    sub.add_parser("doctor")
    sub.add_parser("test")
    sub.add_parser("report")

    pa = sub.add_parser("add")
    pa.add_argument("file")
    pa.add_argument("--collection", help="default: DEFAULT_COLLECTION")
    pa.add_argument("--title")
    pa.add_argument("--host", type=int, choices=[0, 1], default=0,
                    help="which speaker label is the host (0 = first to speak)")
    pa.add_argument("--keyterms", default="", help="comma-separated names/jargon to help transcription")
    pa.add_argument("--embedder")

    pe = sub.add_parser("eval")
    pe.add_argument("--embedders", nargs="*", default=[config.EMBEDDER])
    pe.add_argument("--rerank", action="store_true", help="include rerank and rerank+MMR configurations")

    pb = sub.add_parser("bench")
    pb.add_argument("--url", default="http://127.0.0.1:8000")
    pb.add_argument("--concurrency", type=int, default=5)
    pb.add_argument("--rounds", type=int, default=10)

    a = p.parse_args()
    if a.cmd != "serve":  # serve's child processes set up their own logs
        logs.setup("cli")
        logs.get("audiosearch.cli").info("command: %s", " ".join(sys.argv[1:]))
    try:
        dispatch(a)
    except db.CollectionError as e:
        raise SystemExit(f"error: {e}") from None


def dispatch(a: argparse.Namespace) -> None:
    if a.cmd == "doctor":
        aio.run(doctor())
        return
    if a.cmd == "report":
        from eval import report
        raise SystemExit(report.main())
    if a.cmd == "test":
        raise SystemExit(subprocess.call([sys.executable, "-m", "pytest", "-q"], cwd=ROOT))
    if a.cmd == "add":
        aio.run(add_file(a))
        return
    if a.cmd == "eval":
        from eval import build_golden, evaluate
        if build_golden.main():
            raise SystemExit("error: golden labels could not be built")
        aio.run(evaluate.amain(a.embedders, a.rerank))
        return
    if a.cmd == "bench":
        sys.argv = ["bench", "--url", a.url, "--concurrency", str(a.concurrency), "--rounds", str(a.rounds)]
        from eval import bench
        bench.main()
        return
    if a.cmd == "ingest":
        from .ingest import run
        aio.run(run(a.embedders, rechunk=a.rechunk, only=a.only, collection=a.collection))
    elif a.cmd == "collections":
        aio.run(collections(a))
    elif a.cmd == "preview":
        preview(a.file_id)
    elif a.cmd == "search":
        aio.run(search(a))
    elif a.cmd == "serve":
        serve()


async def search(a: argparse.Namespace) -> None:
    from .search import Searcher
    pool = await db.open_pool(min_size=1, max_size=4)
    try:
        await db.require_collection(pool, a.collection or config.DEFAULT_COLLECTION)
        searcher = Searcher(pool, a.embedder, collection=a.collection)
        out = await searcher.search(a.query, a.method, a.k, a.speaker, a.file,
                                    rerank=a.rerank, mmr=a.mmr, mmr_lambda=a.mmr_lambda,
                                    group=a.group, role=a.role, auto_speaker=a.auto_speaker,
                                    threshold=a.threshold)
    finally:
        await pool.close()
        await aio.close_http()
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


async def doctor() -> None:
    """Environment check for a fresh checkout: database, keys, data, indexes."""
    ok = True

    def line(good: bool, what: str, hint: str = "") -> None:
        nonlocal ok
        ok &= good
        print(f"  {'OK ' if good else 'FIX'}  {what}" + (f"  ->  {hint}" if not good and hint else ""))

    print("audiosearch doctor")
    keys = {"GEMINI_API_KEY": "query/document embeddings (EMBEDDER=gemini) and summary answers",
            "COHERE_API_KEY": "reranking (or set RERANK_PROVIDER=local)",
            "ASSEMBLYAI_API_KEY": "transcribing NEW audio with speaker names (TRANSCRIBER=assemblyai)"}
    for key, why in keys.items():
        line(bool(getattr(config, key)), f"{key} set", f"add it to .env: {why}")
    line(config.MANIFEST_PATH.exists(), "golden manifest data/manifest.json")
    line(config.QUERIES_PATH.exists() or (config.DATA_DIR / "golden_spec.json").exists(), "golden query set")
    try:
        pool = await db.open_pool(min_size=1, max_size=2)
    except Exception as e:  # database unreachable
        line(False, "Postgres + pgvector reachable", f"docker compose up -d ({type(e).__name__})")
        raise SystemExit(1) from None
    line(True, "Postgres + pgvector reachable")
    for c in await db.list_collections(pool):
        async with db.collection_conn(pool, c["name"]) as conn:
            cur = await conn.execute("SELECT model, count(*) FROM embeddings GROUP BY model")
            emb = dict(await cur.fetchall())
        line(True, f"collection {c['name']!r}: {c['files']} files, {c['segments']} chunks, embeddings {emb or 'none'}")
    await pool.close()
    print("ready" if ok else "fix the items marked FIX, then re-run: python -m audiosearch doctor")


async def add_file(a: argparse.Namespace) -> None:
    """Ingest one audio file end to end (same path as the UI upload)."""
    import hashlib

    from .ingest import embed_missing, normalize_audio, save_file
    from .segment import chunk_transcript, speaker_names
    from .transcribe import transcribe
    src = Path(a.file)
    if not src.exists():
        raise SystemExit(f"error: no such file {src}")
    collection = a.collection or config.DEFAULT_COLLECTION
    fid = "up_" + hashlib.sha1(src.read_bytes()).hexdigest()[:8]
    audio = config.AUDIO_DIR / f"{fid}.mp3"
    pool = await db.open_pool(min_size=1, max_size=4)
    try:
        await db.require_collection(pool, collection)
        print(f"[1/4] normalizing {src.name} -> {audio.name}")
        await normalize_audio(src, audio)
        print(f"[2/4] transcribing with {config.TRANSCRIBER} (diarization + speaker names)")
        t = await transcribe(fid, audio, [k.strip() for k in a.keyterms.split(",") if k.strip()])
        names = {k: v or f"Speaker {k + 1}" for k, v in speaker_names(t).items()}
        roles = {k: "host" if k == a.host else "guest" for k in names}
        parents, children = chunk_transcript(t)
        print(f"      speakers: {names} · roles: {roles} · {len(parents)} parents / {len(children)} chunks")
        print(f"[3/4] storing in collection {collection!r}")
        n = await save_file(pool, fid, a.title or src.stem, audio.name, t, names, origin="upload",
                            audio_url=config.AUDIO_URL_BASE + src.name, roles=roles, collection=collection)
        print(f"[4/4] embedding {n} chunks")
        await embed_missing(pool, a.embedder, collection=collection)
        print(f"done: {fid} is searchable -> python -m audiosearch search \"...\" --collection {collection}")
    finally:
        await pool.close()
        await aio.close_http()


async def collections(a: argparse.Namespace) -> None:
    pool = await db.open_pool(min_size=1, max_size=2)
    try:
        if a.action == "create":
            if not a.name:
                raise SystemExit("usage: collections create NAME [--description TEXT]")
            await db.create_collection(pool, a.name, a.description)
            print(f"collection {a.name!r} ready")
        for c in await db.list_collections(pool):
            print(f"{c['name']:20s} files={c['files']:<4} chunks={c['segments']:<6} {c['description'] or ''}")
    finally:
        await pool.close()


def snippet(highlighted: str, width: int = 40) -> str:
    """Trim long chunk text to a window of words around the first highlight."""
    words = highlighted.split()
    first = next((i for i, w in enumerate(words) if w.startswith("[[")), 0)
    lo = max(0, first - width // 2)
    out = " ".join(words[lo: lo + width])
    return ("… " if lo else "") + out + (" …" if lo + width < len(words) else "")


def preview(file_id: str) -> None:
    from .segment import clean_speakers, speaker_names, turns, words_from_transcript
    from .transcribe import load_transcript
    t = load_transcript(file_id)
    raw = words_from_transcript(t)
    print(f"provider: {t['provider']} · raw speaker labels: {sorted({w.speaker for w in raw})}"
          f" · identified names: {speaker_names(t)}")
    shown: dict[int, int] = {}
    for t in turns(clean_speakers(raw)):
        spk = t[0].speaker
        if shown.get(spk, 0) >= 3 or len(t) < 8:
            continue
        shown[spk] = shown.get(spk, 0) + 1
        text = re.sub(r"\s+", " ", " ".join(w.text for w in t))[:220]
        print(f"speaker {spk} @ {t[0].start:6.1f}s: {text}")


def serve() -> None:
    """Run the FastAPI backend and the Streamlit frontend together; Ctrl+C stops both."""
    procs = [subprocess.Popen(API_CMD, cwd=ROOT), subprocess.Popen(UI_CMD, cwd=ROOT)]
    try:
        for proc in procs:
            proc.wait()
    except KeyboardInterrupt:
        for proc in procs:
            proc.terminate()


if __name__ == "__main__":
    main()
