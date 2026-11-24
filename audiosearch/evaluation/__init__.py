"""Golden set, label building (from transcript anchors) and recall@k evaluation."""
from .evaluator import KS, EvalConfig, Evaluator, configurations, evaluate, to_markdown
from .golden import GoldenSet, golden_set, is_hit, load_queries
from .labels import GoldenLabelBuilder, LabelReport

__all__ = ["KS", "EvalConfig", "Evaluator", "GoldenLabelBuilder", "GoldenSet", "LabelReport", "configurations",
           "evaluate", "golden_set", "is_hit", "load_queries", "to_markdown"]
