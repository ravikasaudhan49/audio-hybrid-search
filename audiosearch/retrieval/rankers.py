"""First-stage rankers (Strategy pattern): each builds ONE self-contained SQL statement that
returns segment ids in rank order. The SQL is unchanged from the original single-module
implementation.

All rankers of one search run sequentially on ONE pooled connection (psycopg pipeline mode
was measured 7x slower here, ~48 ms vs 6.6 ms, due to TCP delayed-ACK on Windows->Docker;
ADR-0006), so a search costs one connection regardless of how many rankers run.
"""
from abc import ABC, abstractmethod

import numpy as np

from ..embeddings import Embedder
from ..models import Filters

Query = tuple[str, dict]
CANDIDATES = 50       # per-ranker candidate depth fed into fusion


def filter_clause(f: Filters) -> tuple[str, dict]:
    """WHERE-clause fragment over `segments s`, with named parameters."""
    sql, params = "", {}
    if f.origin:
        sql += " AND s.file_id IN (SELECT id FROM files WHERE origin = %(f_origin)s)"
        params["f_origin"] = f.origin
    if f.file_id:
        sql += " AND s.file_id = %(f_file)s"
        params["f_file"] = f.file_id
    if f.speaker:
        sql += " AND s.speaker_id IN (SELECT id FROM speakers WHERE name ILIKE %(f_speaker)s)"
        params["f_speaker"] = f"%{f.speaker}%"
    if f.role:
        sql += " AND s.speaker_id IN (SELECT id FROM speakers WHERE role = %(f_role)s)"
        params["f_role"] = f.role
    return sql, params


class Ranker(ABC):
    name: str
    weight: float = 1.0           # weight in Reciprocal Rank Fusion

    def __init__(self, k: int = CANDIDATES):
        self.k = k

    @abstractmethod
    def query(self, text: str, f: Filters) -> Query:
        """(sql, params) returning segment ids, best first."""


class KeywordRanker(Ranker):
    """Full-text ranker. A quoted phrase or all-terms match ranks first, then chunks by how
    many query terms they cover, then by cover density (terms appearing close together).

    A chunk sharing one common word with a long question is a weak keyword signal, yet in RRF
    only rank counts. So 3+ term queries must match at least half their terms; 1-2 term
    queries (names, jargon) accept any match. Lexemes are the query after stemming and
    stopword removal, e.g. "how to evaluate search" -> {evalu, search}."""
    name = "keyword"
    weight = 1.0

    def query(self, text: str, f: Filters) -> Query:
        fsql, fparams = filter_clause(f)
        sql = f"""
            WITH q AS (
                SELECT websearch_to_tsquery('english', %(q)s) AS strict,
                       CASE WHEN %(phrase)s THEN websearch_to_tsquery('english', %(q)s)
                            ELSE nullif(replace(plainto_tsquery('english', %(q)s)::text, '&', '|'), '')::tsquery
                       END AS loose,
                       tsvector_to_array(to_tsvector('english', %(q)s)) AS lex),
            scored AS (
                SELECT s.id, s.tsv,
                       (SELECT count(*) FROM unnest(q.lex) l WHERE s.tsv @@ quote_literal(l)::tsquery) AS cover
                FROM segments s, q
                WHERE s.tsv @@ q.loose {fsql})
            SELECT scored.id FROM scored, q
            WHERE %(phrase)s
               OR cover >= CASE WHEN cardinality(q.lex) <= 2 THEN 1 ELSE ceil(cardinality(q.lex) / 2.0) END
            ORDER BY (scored.tsv @@ q.strict) DESC, cover DESC, ts_rank_cd(scored.tsv, q.loose, 32) DESC, scored.id
            LIMIT %(k)s"""
        return sql, {"q": text, "phrase": '"' in text, "k": self.k, **fparams}


class FuzzyRanker(Ranker):
    """Trigram word-similarity ranker: catches misspellings and ASR spelling variants."""
    name = "fuzzy"
    weight = 0.5

    def query(self, text: str, f: Filters) -> Query:
        fsql, fparams = filter_clause(f)
        sql = f"""
            SELECT s.id FROM segments s
            WHERE %(q)s <%% s.child_text {fsql}
            ORDER BY word_similarity(%(q)s, s.child_text) DESC, s.id
            LIMIT %(k)s"""
        return sql, {"q": text, "k": self.k, **fparams}


class VectorRanker(Ranker):
    """Cosine k-NN over this model's partial HNSW index (same cast + WHERE as the index).
    Candidates below the model's similarity floor are dropped (cosine distance = 1 - sim).
    The query vector is computed BEFORE a connection is taken (CLAUDE.md rule 4)."""
    name = "vector"
    weight = 1.0

    def __init__(self, embedder: Embedder, query_vector: np.ndarray, k: int = CANDIDATES):
        super().__init__(k)
        self.embedder = embedder
        self.query_vector = query_vector

    def query(self, text: str, f: Filters) -> Query:
        fsql, fparams = filter_clause(f)
        cast = f"vector({self.embedder.dim})"
        sql = f"""
            SELECT s.id FROM embeddings e JOIN segments s ON s.id = e.segment_id
            WHERE e.model = %(model)s {fsql}
              AND (e.embedding::{cast} <=> %(qv)s::{cast}) <= %(max_dist)s
            ORDER BY e.embedding::{cast} <=> %(qv)s::{cast}
            LIMIT %(k)s"""
        return sql, {"model": self.embedder.name, "qv": self.query_vector, "k": self.k,
                     "max_dist": 1 - self.embedder.min_similarity, **fparams}
