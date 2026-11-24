"""audiosearch: hybrid (keyword + semantic) search over two-speaker podcast audio.

Package map (see docs/ARCHITECTURE.md):
  core/           settings, logging, async runtime, disk cache, errors, text helpers
  models.py       domain dataclasses
  db/             pool, collections, repositories, schema.sql
  ingestion/      INGESTION PIPELINE: audio, transcription, chunking, storage, embedding
  retrieval/      RETRIEVAL PIPELINE: query parsing, rankers, fusion, rerankers, answers
  embeddings/     Gemini embeddings (used by both pipelines)
  evaluation/     golden set, labels, evaluator
  container.py    AppContainer: builds the embedder, reranker, transcriber once; creates services
  cli/            `python -m audiosearch`
"""
