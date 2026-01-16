# src/models/__init__.py
from .llm import get_llm
from .embeddings import get_embeddings

__all__ = ["get_llm", "get_embeddings"]
