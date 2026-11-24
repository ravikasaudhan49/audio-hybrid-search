"""Database layer: pool and collections (`database`), SQL per table (`repositories`).

All table access goes through `collection_conn(pool, collection)` (CLAUDE.md rule 3).
"""
from .database import Database, collection_conn, is_valid_name, schema_name
from .repositories import (
    CollectionRepository,
    EmbeddingRepository,
    FileRepository,
    SegmentRepository,
    SpeakerRepository,
)

__all__ = ["CollectionRepository", "Database", "EmbeddingRepository", "FileRepository", "SegmentRepository",
           "SpeakerRepository", "collection_conn", "is_valid_name", "schema_name"]
