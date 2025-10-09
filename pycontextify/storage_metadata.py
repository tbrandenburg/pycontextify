"""Metadata structures for PyContextify.

This module defines the data structures for storing chunk metadata with
lightweight knowledge graph capabilities, including relationship tracking
and entity references.
"""

import gzip
import pickle
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .types import Chunk, SourceType


@dataclass
class ChunkMetadata:
    """Metadata for a text chunk with relationship information.

    This dataclass stores all information about a chunk including its
    content, source, position, and relationship data for lightweight
    knowledge graph capabilities.
    """

    # Core chunk information
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_path: str = ""
    source_type: SourceType = SourceType.DOCUMENT
    chunk_text: str = ""
    start_char: int = 0
    end_char: int = 0
    created_at: datetime = field(default_factory=datetime.now)

    # File-specific information
    file_extension: Optional[str] = None

    # Embedding information
    embedding_provider: str = "sentence_transformers"
    embedding_model: str = "all-mpnet-base-v2"

    # Lightweight knowledge graph fields
    tags: List[str] = field(default_factory=list)
    references: List[str] = field(default_factory=list)
    parent_section: Optional[str] = None
    code_symbols: List[str] = field(default_factory=list)

    # Additional metadata for different content types
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_tag(self, tag: str) -> None:
        """Add a tag if not already present."""
        if tag not in self.tags:
            self.tags.append(tag)

    def add_reference(self, reference: str) -> None:
        """Add a reference if not already present."""
        if reference not in self.references:
            self.references.append(reference)

    def add_code_symbol(self, symbol: str) -> None:
        """Add a code symbol if not already present."""
        if symbol not in self.code_symbols:
            self.code_symbols.append(symbol)

    def get_relationships(self) -> Dict[str, List[str]]:
        """Get all relationship information for this chunk."""
        return {
            "tags": self.tags,
            "references": self.references,
            "code_symbols": self.code_symbols,
            "parent_section": [self.parent_section] if self.parent_section else [],
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "chunk_id": self.chunk_id,
            "source_path": self.source_path,
            "source_type": self.source_type.value,
            "chunk_text": self.chunk_text,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "created_at": self.created_at.isoformat(),
            "file_extension": self.file_extension,
            "embedding_provider": self.embedding_provider,
            "embedding_model": self.embedding_model,
            "tags": self.tags,
            "references": self.references,
            "parent_section": self.parent_section,
            "code_symbols": self.code_symbols,
            "metadata": self.metadata,
        }

    @classmethod
    def from_chunk(cls, chunk: Chunk) -> "ChunkMetadata":
        """Create ChunkMetadata from a Chunk DTO.

        Args:
            chunk: Chunk object from the chunking pipeline

        Returns:
            ChunkMetadata with storage-specific fields (chunk_id, created_at)
        """
        return cls(
            source_path=chunk.source_path,
            source_type=chunk.source_type,
            chunk_text=chunk.chunk_text,
            start_char=chunk.start_char,
            end_char=chunk.end_char,
            file_extension=chunk.file_extension,
            embedding_provider=chunk.embedding_provider,
            embedding_model=chunk.embedding_model,
            tags=chunk.tags.copy(),
            references=chunk.references.copy(),
            parent_section=chunk.parent_section,
            code_symbols=chunk.code_symbols.copy(),
            metadata=chunk.metadata.copy(),
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChunkMetadata":
        """Create from dictionary."""
        # Handle datetime parsing
        created_at = datetime.fromisoformat(data["created_at"])

        # Handle enum parsing
        source_type = SourceType(data["source_type"])

        return cls(
            chunk_id=data["chunk_id"],
            source_path=data["source_path"],
            source_type=source_type,
            chunk_text=data["chunk_text"],
            start_char=data["start_char"],
            end_char=data["end_char"],
            created_at=created_at,
            file_extension=data.get("file_extension"),
            embedding_provider=data.get("embedding_provider", "sentence_transformers"),
            embedding_model=data.get("embedding_model", "all-mpnet-base-v2"),
            tags=data.get("tags", []),
            references=data.get("references", []),
            parent_section=data.get("parent_section"),
            code_symbols=data.get("code_symbols", []),
            metadata=data.get("metadata", {}),
        )


class MetadataStore:
    """Store for managing chunk metadata with FAISS index mapping.

    This class manages the bidirectional mapping between FAISS vector indices
    and chunk metadata, providing persistence and querying capabilities.
    """

    def __init__(self) -> None:
        """Initialize empty metadata store."""
        self._id_to_metadata: Dict[int, ChunkMetadata] = {}
        self._chunk_id_to_faiss_id: Dict[str, int] = {}
        self._next_faiss_id: int = 0

    def add_chunk(self, metadata: ChunkMetadata) -> int:
        """Add chunk metadata and return assigned FAISS ID.

        Args:
            metadata: Chunk metadata to store

        Returns:
            FAISS ID assigned to this chunk
        """
        faiss_id = self._next_faiss_id
        self._id_to_metadata[faiss_id] = metadata
        self._chunk_id_to_faiss_id[metadata.chunk_id] = faiss_id
        self._next_faiss_id += 1
        return faiss_id

    def get_chunk(self, faiss_id: int) -> Optional[ChunkMetadata]:
        """Retrieve chunk metadata by FAISS ID."""
        return self._id_to_metadata.get(faiss_id)

    def get_chunk_by_chunk_id(self, chunk_id: str) -> Optional[ChunkMetadata]:
        """Retrieve chunk metadata by chunk ID."""
        faiss_id = self._chunk_id_to_faiss_id.get(chunk_id)
        if faiss_id is not None:
            return self._id_to_metadata.get(faiss_id)
        return None

    def get_faiss_id(self, chunk_id: str) -> Optional[int]:
        """Get FAISS ID for a chunk ID."""
        return self._chunk_id_to_faiss_id.get(chunk_id)

    def get_all_chunks(self) -> List[ChunkMetadata]:
        """Get all stored metadata."""
        return list(self._id_to_metadata.values())

    def get_chunks_by_source_type(self, source_type: SourceType) -> List[ChunkMetadata]:
        """Get all chunks of a specific source type."""
        return [
            chunk
            for chunk in self._id_to_metadata.values()
            if chunk.source_type == source_type
        ]

    def get_chunks_by_source_path(self, source_path: str) -> List[ChunkMetadata]:
        """Get all chunks from a specific source path."""
        return [
            chunk
            for chunk in self._id_to_metadata.values()
            if chunk.source_path == source_path
        ]

    def find_chunks_by_tag(self, tag: str) -> List[ChunkMetadata]:
        """Find chunks with specific tag."""
        return [chunk for chunk in self._id_to_metadata.values() if tag in chunk.tags]

    def find_chunks_by_reference(self, reference: str) -> List[ChunkMetadata]:
        """Find chunks referencing specific entity."""
        return [
            chunk
            for chunk in self._id_to_metadata.values()
            if reference in chunk.references
        ]

    def find_chunks_by_code_symbol(self, symbol: str) -> List[ChunkMetadata]:
        """Find chunks containing specific code symbol."""
        return [
            chunk
            for chunk in self._id_to_metadata.values()
            if symbol in chunk.code_symbols
        ]

    def get_chunk_relationships(self, chunk_id: str) -> Dict[str, List[str]]:
        """Get all relationships for a specific chunk."""
        chunk = self.get_chunk_by_chunk_id(chunk_id)
        if chunk:
            return chunk.get_relationships()
        return {}

    def get_all_tags(self) -> Set[str]:
        """Get all unique tags across all chunks."""
        tags = set()
        for chunk in self._id_to_metadata.values():
            tags.update(chunk.tags)
        return tags

    def get_all_references(self) -> Set[str]:
        """Get all unique references across all chunks."""
        references = set()
        for chunk in self._id_to_metadata.values():
            references.update(chunk.references)
        return references

    def get_all_code_symbols(self) -> Set[str]:
        """Get all unique code symbols across all chunks."""
        symbols = set()
        for chunk in self._id_to_metadata.values():
            symbols.update(chunk.code_symbols)
        return symbols

    def discover_topics(self) -> List[str]:
        """Discover all unique topics from indexed chunks.

        Returns:
            Sorted list of unique topic names from chunk metadata
        """
        topics = set()
        for chunk in self._id_to_metadata.values():
            # Topic should be stored in chunk.metadata["topic"]
            topic = chunk.metadata.get("topic")
            if topic and isinstance(topic, str):
                topics.add(topic)

        return sorted(topics)

    def validate_embedding_compatibility(self, provider: str, model: str) -> bool:
        """Check if existing chunks are compatible with current embedding settings."""
        if not self._id_to_metadata:
            return True  # No existing chunks, any provider is compatible

        # Check if all existing chunks use the same provider and model
        for chunk in self._id_to_metadata.values():
            if chunk.embedding_provider != provider or chunk.embedding_model != model:
                return False

        return True

    def get_embedding_info(self) -> Dict[str, Any]:
        """Get information about embedding providers used in stored chunks."""
        if not self._id_to_metadata:
            return {"providers": [], "models": [], "total_chunks": 0}

        providers = set()
        models = set()

        for chunk in self._id_to_metadata.values():
            providers.add(chunk.embedding_provider)
            models.add(f"{chunk.embedding_provider}:{chunk.embedding_model}")

        return {
            "providers": list(providers),
            "models": list(models),
            "total_chunks": len(self._id_to_metadata),
        }

    def get_stats(self) -> Dict[str, Any]:
        """Return comprehensive statistics about stored metadata."""
        if not self._id_to_metadata:
            return {
                "total_chunks": 0,
                "source_types": {},
                "embedding_info": {"providers": [], "models": []},
                "relationship_stats": {
                    "total_tags": 0,
                    "total_references": 0,
                    "total_code_symbols": 0,
                },
            }

        # Source type breakdown
        source_type_counts = {}
        for chunk in self._id_to_metadata.values():
            source_type = chunk.source_type.value
            source_type_counts[source_type] = source_type_counts.get(source_type, 0) + 1

        # Relationship statistics
        all_tags = self.get_all_tags()
        all_references = self.get_all_references()
        all_code_symbols = self.get_all_code_symbols()

        return {
            "total_chunks": len(self._id_to_metadata),
            "source_types": source_type_counts,
            "embedding_info": self.get_embedding_info(),
            "relationship_stats": {
                "total_tags": len(all_tags),
                "total_references": len(all_references),
                "total_code_symbols": len(all_code_symbols),
                "unique_entities": len(all_tags | all_references | all_code_symbols),
            },
            "sources": len(
                set(chunk.source_path for chunk in self._id_to_metadata.values())
            ),
        }

    def remove_chunk(self, faiss_id: int) -> bool:
        """Remove chunk metadata by FAISS ID.

        Args:
            faiss_id: FAISS ID of the chunk to remove

        Returns:
            True if chunk was removed, False if not found
        """
        if faiss_id not in self._id_to_metadata:
            return False

        # Get the chunk to remove
        chunk = self._id_to_metadata[faiss_id]

        # Remove from both mappings
        del self._id_to_metadata[faiss_id]
        if chunk.chunk_id in self._chunk_id_to_faiss_id:
            del self._chunk_id_to_faiss_id[chunk.chunk_id]

        return True

    def clear(self) -> None:
        """Clear all metadata."""
        self._id_to_metadata.clear()
        self._chunk_id_to_faiss_id.clear()
        self._next_faiss_id = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "chunks": {
                str(faiss_id): chunk.to_dict()
                for faiss_id, chunk in self._id_to_metadata.items()
            },
            "chunk_id_mapping": self._chunk_id_to_faiss_id,
            "next_faiss_id": self._next_faiss_id,
        }

    def from_dict(self, data: Dict[str, Any]) -> None:
        """Load from dictionary."""
        self.clear()

        # Restore chunks
        for faiss_id_str, chunk_data in data.get("chunks", {}).items():
            faiss_id = int(faiss_id_str)
            chunk = ChunkMetadata.from_dict(chunk_data)
            self._id_to_metadata[faiss_id] = chunk

        # Restore mappings
        self._chunk_id_to_faiss_id = data.get("chunk_id_mapping", {})
        self._next_faiss_id = data.get("next_faiss_id", 0)

    def save_to_file(self, filepath: str, compress: bool = True) -> None:
        """Save metadata to file.

        Args:
            filepath: Path to save metadata
            compress: Whether to compress the file with gzip
        """
        data = self.to_dict()

        if compress:
            with gzip.open(filepath, "wb") as f:
                pickle.dump(data, f)
        else:
            with open(filepath, "wb") as f:
                pickle.dump(data, f)

    def load_from_file(self, filepath: str) -> None:
        """Load metadata from file.

        Args:
            filepath: Path to load metadata from
        """
        file_path = Path(filepath)
        if not file_path.exists():
            return

        try:
            # Try to load as compressed file first
            with gzip.open(filepath, "rb") as f:
                data = pickle.load(f)
        except (gzip.BadGzipFile, OSError):
            # Fall back to uncompressed file
            with open(filepath, "rb") as f:
                data = pickle.load(f)

        self.from_dict(data)
