"""Summary answer: no results -> no model call; citations parsed and bounded (no API)."""
from audiosearch import aio
from audiosearch.answer import _citations, generate_answer


def test_no_results_means_no_answer_and_no_api_call():
    assert aio.run(generate_answer("anything", [])) is None


def test_citations_are_parsed_deduplicated_and_bounded():
    assert _citations("Kranz said X [2]. Also [1] and [2], not [9].", n=3) == [1, 2]
