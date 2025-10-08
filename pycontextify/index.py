"""Index package for PyContextify.

This package contains all the core indexing functionality including
embedding providers, vector storage, content loading, text chunking,
and lightweight knowledge graph capabilities.
"""

from .indexer_manager import IndexManager
from .orchestrator_config import Config
from .storage_metadata import ChunkMetadata, MetadataStore, SourceType

__all__ = [
    "IndexManager",
    "Config",
    "ChunkMetadata",
    "SourceType",
    "MetadataStore",
]
