"""Retrieval pipeline: query -> rankers -> fusion -> rerank -> threshold -> MMR -> answer.

  pipeline.py      SearchService (the steps, in order)
  query_parser.py  "what did <name> say about X" -> speaker filter
  rankers.py       keyword, fuzzy and vector SQL
  fusion.py        RRF, parent grouping, thresholds, MMR
  highlight.py     matched words and their timestamps
  rerankers/       Cohere / local cross-encoder
  answer.py        grounded summary answer
"""
from .pipeline import SearchService, SpeakerNameCache

__all__ = ["SearchService", "SpeakerNameCache"]
