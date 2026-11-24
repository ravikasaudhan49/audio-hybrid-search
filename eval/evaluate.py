"""Recall@k / MRR evaluation over the labeled query set (golden files only).

The evaluator lives in audiosearch/evaluation/evaluator.py; this is the command-line entry point.

Run:  python -m eval.evaluate [--rerank]
"""
import argparse

from audiosearch.container import AppContainer
from audiosearch.core import log as logs
from audiosearch.core import runtime
from audiosearch.evaluation import to_markdown


async def amain(rerank: bool) -> None:
    async with AppContainer.open() as app:
        report = await app.evaluator().run(rerank)
    print("\n" + to_markdown(report))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rerank", action="store_true")
    a = p.parse_args(argv)
    logs.setup("eval")
    runtime.run(amain(a.rerank))


if __name__ == "__main__":
    main()
