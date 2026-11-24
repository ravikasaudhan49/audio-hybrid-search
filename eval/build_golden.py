"""Build data/queries.json (labeled spans) from data/golden_spec.json (queries + anchors).

The labelling method lives in audiosearch/evaluation/labels.py (ADR-0008); this is the
command-line entry point.

Run:  python -m eval.build_golden
"""
import sys

from audiosearch.evaluation import GoldenLabelBuilder


def main() -> int:
    report = GoldenLabelBuilder().build()
    print(f"wrote queries.json: {len(report.queries)} queries {report.by_type}")
    for p in report.problems:
        print("  !", p)
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
