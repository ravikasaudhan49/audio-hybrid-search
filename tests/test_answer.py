"""Summary answer: no results -> no model call; citations parsed and bounded (no API)."""
from audiosearch.core import runtime
from audiosearch.retrieval.answer import AnswerService, parse_citations


def test_no_results_means_no_answer_and_no_api_call():
    assert runtime.run(AnswerService().generate("anything", [])) is None


def test_citations_are_parsed_deduplicated_and_bounded():
    assert parse_citations("Kranz said X [2]. Also [1] and [2], not [9].", n=3) == [1, 2]
