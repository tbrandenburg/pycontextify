"""Embedders package for PyContextify.

This package provides an extensible embedding system with support for
multiple embedding providers including sentence-transformers, Ollama, and OpenAI.
"""

from .embedder_base import BaseEmbedder
from .embedder_factory import EmbedderFactory
from .embedder_sentence_transformers_embedder import SentenceTransformersEmbedder

__all__ = ["BaseEmbedder", "EmbedderFactory", "SentenceTransformersEmbedder"]
