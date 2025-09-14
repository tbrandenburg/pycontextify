"""Embedders package for PyContextify.

This package provides an extensible embedding system with support for
multiple embedding providers including sentence-transformers, Ollama, and OpenAI.
"""

from .base import BaseEmbedder
from .factory import EmbedderFactory
from .sentence_transformers_embedder import SentenceTransformersEmbedder

__all__ = ['BaseEmbedder', 'EmbedderFactory', 'SentenceTransformersEmbedder']