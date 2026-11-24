"""Embeddings via the Gemini API (EMBEDDER=gemini). AppContainer builds the embedder once."""
from .base import Embedder, normalize_rows

__all__ = ["Embedder", "normalize_rows"]
