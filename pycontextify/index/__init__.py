"""Index package for PyContextify.

This package contains all the core indexing functionality including
embedding providers, vector storage, content loading, text chunking,
and lightweight knowledge graph capabilities.
"""

from .manager import IndexManager
from .config import Config
from .metadata import ChunkMetadata, SourceType, MetadataStore
from .relationship_store import RelationshipStore

__all__ = [
    "IndexManager",
    "Config", 
    "ChunkMetadata",
    "SourceType",
    "MetadataStore",
    "RelationshipStore"
]