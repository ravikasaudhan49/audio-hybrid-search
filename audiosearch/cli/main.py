"""CLI:  python -m audiosearch <command>

  doctor                         check database, API keys, data and indexes
  collections [create NAME [--description TEXT]]   list collections, or create one
  ingest   [--rechunk] [--only ep01] [--collection NAME]
                                 ingest the golden manifest (data/manifest.json)
  add      FILE [--collection NAME] [--title T] [--host 0|1] [--keyterms "a,b"]
                                 ingest any audio file: transcribe (+ speaker names), chunk, embed
  eval     [--rerank]                    build golden labels + recall@k report
  bench    [--concurrency 5] [--rounds 10] [--url http://127.0.0.1:8000]   latency p50/p90/p99
  test                           run the automated test suite (incl. the recall@k gate)
  report                         write all evidence to test-results/
                                 (eval tables, per-query hits, pytest output, latency)
  preview  ep01                  show first turns per speaker label (to name speakers)
  search   "query" [--method hybrid|keyword|vector|fuzzy] [-k 5] [--speaker NAME] [--file ep01]
                   [--no-rerank] [--mmr [--mmr-lambda 0.6]] [--no-group] [--context]
                   [--role host|guest] [--no-auto-speaker] [--no-threshold]
  serve                          API on :8000 (docs at /docs) + Streamlit UI on :8501
"""
import argparse
import sys
from collections.abc import Callable

from ..core import log as logs
from ..core.errors import CollectionError
from ..core.settings import get_settings
from ..models import METHODS
from . import commands

Handler = Callable[[argparse.Namespace], None]


def build_parser() -> argparse.ArgumentParser:
    s = get_settings()
    p = argparse.ArgumentParser(prog="audiosearch")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("ingest")
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
    ps.add_argument("--method", default="hybrid", choices=list(METHODS))
    ps.add_argument("-k", type=int, default=5)
    ps.add_argument("--speaker")
    ps.add_argument("--file")
    ps.add_argument("--no-rerank", dest="rerank", action="store_false", help="skip the reranker (on by default)")
    ps.add_argument("--mmr", action="store_true", help="diversify results with Maximal Marginal Relevance")
    ps.add_argument("--mmr-lambda", type=float, default=s.mmr_lambda,
                    help="1.0 = pure relevance, 0.0 = pure diversity")
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

    pe = sub.add_parser("eval")
    pe.add_argument("--rerank", action="store_true", help="include rerank and rerank+MMR configurations")

    pb = sub.add_parser("bench")
    pb.add_argument("--url", default="http://127.0.0.1:8000")
    pb.add_argument("--concurrency", type=int, default=5)
    pb.add_argument("--rounds", type=int, default=10)
    return p


COMMANDS: dict[str, Handler] = {
    "doctor": commands.doctor,
    "collections": commands.collections,
    "ingest": commands.ingest,
    "add": commands.add,
    "eval": commands.evaluate,
    "bench": commands.bench,
    "test": commands.test,
    "report": commands.report,
    "preview": commands.preview,
    "search": commands.search,
    "serve": commands.serve,
}


def main(argv: list[str] | None = None) -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252
    a = build_parser().parse_args(argv)
    if a.cmd != "serve":  # serve's child processes set up their own logs
        logs.setup("cli")
        logs.get("audiosearch.cli").info("command: %s", " ".join(sys.argv[1:]))
    try:
        COMMANDS[a.cmd](a)
    except CollectionError as e:
        raise SystemExit(f"error: {e}") from None
