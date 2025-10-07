"""Index package for PyContextify.

This package contains all the core indexing functionality including
embedding providers, vector storage, content loading, text chunking,
and lightweight knowledge graph capabilities.
"""

from ..indexer.manager import IndexManager
from ..orchestrator.config import Config
from ..storage.metadata import ChunkMetadata, MetadataStore, SourceType

__all__ = [
    "IndexManager",
    "Config",
    "ChunkMetadata",
    "SourceType",
    "MetadataStore",
]
