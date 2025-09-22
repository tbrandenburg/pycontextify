"""Tests for metadata and relationship systems."""

import os
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from pycontextify.index.metadata import ChunkMetadata, MetadataStore, SourceType
from pycontextify.index.relationship_store import RelationshipStore


class TestChunkMetadata:
    """Test the ChunkMetadata dataclass."""

    def test_chunk_metadata_creation(self):
        """Test basic chunk metadata creation."""
        chunk = ChunkMetadata(
            source_path="/test/file.py",
            source_type=SourceType.CODE,
            chunk_text="def hello(): pass",
            start_char=0,
            end_char=17,
            embedding_provider="sentence_transformers",
            embedding_model="all-mpnet-base-v2",
        )

        assert chunk.source_path == "/test/file.py"
        assert chunk.source_type == SourceType.CODE
        assert chunk.chunk_text == "def hello(): pass"
        assert chunk.embedding_provider == "sentence_transformers"
        assert isinstance(chunk.created_at, datetime)
        assert len(chunk.chunk_id) > 0  # UUID should be generated

    def test_chunk_metadata_defaults(self):
        """Test default values for chunk metadata."""
        chunk = ChunkMetadata()

        assert chunk.source_type == SourceType.DOCUMENT
        assert chunk.embedding_provider == "sentence_transformers"
        assert chunk.embedding_model == "all-mpnet-base-v2"
        assert chunk.tags == []
        assert chunk.references == []
        assert chunk.code_symbols == []

    def test_add_tag(self):
        """Test adding tags to chunk metadata."""
        chunk = ChunkMetadata()

        chunk.add_tag("python")
        chunk.add_tag("function")
        chunk.add_tag("python")  # Duplicate

        assert "python" in chunk.tags
        assert "function" in chunk.tags
        assert len(chunk.tags) == 2  # No duplicates

    def test_add_reference(self):
        """Test adding references to chunk metadata."""
        chunk = ChunkMetadata()

        chunk.add_reference("numpy")
        chunk.add_reference("pandas")
        chunk.add_reference("numpy")  # Duplicate

        assert "numpy" in chunk.references
        assert "pandas" in chunk.references
        assert len(chunk.references) == 2  # No duplicates

    def test_add_code_symbol(self):
        """Test adding code symbols to chunk metadata."""
        chunk = ChunkMetadata()

        chunk.add_code_symbol("hello")
        chunk.add_code_symbol("world")
        chunk.add_code_symbol("hello")  # Duplicate

        assert "hello" in chunk.code_symbols
        assert "world" in chunk.code_symbols
        assert len(chunk.code_symbols) == 2  # No duplicates

    def test_get_relationships(self):
        """Test getting all relationships from chunk."""
        chunk = ChunkMetadata(parent_section="Introduction")
        chunk.add_tag("important")
        chunk.add_reference("documentation")
        chunk.add_code_symbol("main")

        relationships = chunk.get_relationships()

        assert "tags" in relationships
        assert "references" in relationships
        assert "code_symbols" in relationships
        assert "parent_section" in relationships

        assert "important" in relationships["tags"]
        assert "documentation" in relationships["references"]
        assert "main" in relationships["code_symbols"]
        assert "Introduction" in relationships["parent_section"]

    def test_to_dict_serialization(self):
        """Test converting chunk to dictionary."""
        chunk = ChunkMetadata(
            source_path="/test/file.py",
            source_type=SourceType.CODE,
            chunk_text="def hello(): pass",
        )

        data = chunk.to_dict()

        assert isinstance(data, dict)
        assert data["source_path"] == "/test/file.py"
        assert data["source_type"] == "code"
        assert data["chunk_text"] == "def hello(): pass"
        assert "created_at" in data
        assert "chunk_id" in data

    def test_from_dict_deserialization(self):
        """Test creating chunk from dictionary."""
        data = {
            "chunk_id": "test-id",
            "source_path": "/test/file.py",
            "source_type": "code",
            "chunk_text": "def hello(): pass",
            "start_char": 0,
            "end_char": 17,
            "created_at": "2024-01-01T00:00:00",
            "embedding_provider": "sentence_transformers",
            "embedding_model": "all-mpnet-base-v2",
            "tags": ["python"],
            "references": ["test"],
            "code_symbols": ["hello"],
        }

        chunk = ChunkMetadata.from_dict(data)

        assert chunk.chunk_id == "test-id"
        assert chunk.source_path == "/test/file.py"
        assert chunk.source_type == SourceType.CODE
        assert chunk.tags == ["python"]
        assert chunk.references == ["test"]
        assert chunk.code_symbols == ["hello"]


class TestMetadataStore:
    """Test the MetadataStore class."""

    def test_empty_store_initialization(self):
        """Test initializing empty metadata store."""
        store = MetadataStore()

        assert store.get_all_chunks() == []
        assert store.get_stats()["total_chunks"] == 0

    def test_add_and_retrieve_chunk(self):
        """Test adding and retrieving chunk metadata."""
        store = MetadataStore()
        chunk = ChunkMetadata(
            source_path="/test/file.py", chunk_text="def hello(): pass"
        )

        faiss_id = store.add_chunk(chunk)

        assert isinstance(faiss_id, int)
        assert faiss_id >= 0

        retrieved = store.get_chunk(faiss_id)
        assert retrieved is not None
        assert retrieved.source_path == "/test/file.py"
        assert retrieved.chunk_text == "def hello(): pass"

    def test_get_chunk_by_chunk_id(self):
        """Test retrieving chunk by chunk ID."""
        store = MetadataStore()
        chunk = ChunkMetadata(chunk_text="test")

        faiss_id = store.add_chunk(chunk)

        retrieved = store.get_chunk_by_chunk_id(chunk.chunk_id)
        assert retrieved is not None
        assert retrieved.chunk_id == chunk.chunk_id

    def test_get_faiss_id(self):
        """Test getting FAISS ID for chunk ID."""
        store = MetadataStore()
        chunk = ChunkMetadata(chunk_text="test")

        faiss_id = store.add_chunk(chunk)

        retrieved_faiss_id = store.get_faiss_id(chunk.chunk_id)
        assert retrieved_faiss_id == faiss_id

    def test_chunks_by_source_type(self):
        """Test filtering chunks by source type."""
        store = MetadataStore()

        code_chunk = ChunkMetadata(source_type=SourceType.CODE, chunk_text="code")
        doc_chunk = ChunkMetadata(source_type=SourceType.DOCUMENT, chunk_text="doc")
        web_chunk = ChunkMetadata(source_type=SourceType.WEBPAGE, chunk_text="web")

        store.add_chunk(code_chunk)
        store.add_chunk(doc_chunk)
        store.add_chunk(web_chunk)

        code_chunks = store.get_chunks_by_source_type(SourceType.CODE)
        doc_chunks = store.get_chunks_by_source_type(SourceType.DOCUMENT)

        assert len(code_chunks) == 1
        assert len(doc_chunks) == 1
        assert code_chunks[0].chunk_text == "code"
        assert doc_chunks[0].chunk_text == "doc"

    def test_find_chunks_by_tag(self):
        """Test finding chunks by tag."""
        store = MetadataStore()

        chunk1 = ChunkMetadata(chunk_text="chunk1")
        chunk1.add_tag("important")
        chunk2 = ChunkMetadata(chunk_text="chunk2")
        chunk2.add_tag("test")

        store.add_chunk(chunk1)
        store.add_chunk(chunk2)

        important_chunks = store.find_chunks_by_tag("important")
        test_chunks = store.find_chunks_by_tag("test")

        assert len(important_chunks) == 1
        assert len(test_chunks) == 1
        assert important_chunks[0].chunk_text == "chunk1"
        assert test_chunks[0].chunk_text == "chunk2"

    def test_validate_embedding_compatibility(self):
        """Test embedding compatibility validation."""
        store = MetadataStore()

        # Empty store should be compatible with anything
        assert store.validate_embedding_compatibility(
            "openai", "text-embedding-3-small"
        )

        # Add chunk with specific embedding
        chunk = ChunkMetadata(
            chunk_text="test",
            embedding_provider="sentence_transformers",
            embedding_model="all-mpnet-base-v2",
        )
        store.add_chunk(chunk)

        # Same embedding should be compatible
        assert store.validate_embedding_compatibility(
            "sentence_transformers", "all-mpnet-base-v2"
        )

        # Different embedding should not be compatible
        assert not store.validate_embedding_compatibility(
            "openai", "text-embedding-3-small"
        )

    def test_get_stats(self):
        """Test getting comprehensive statistics."""
        store = MetadataStore()

        # Empty stats
        stats = store.get_stats()
        assert stats["total_chunks"] == 0

        # Add some chunks
        chunk1 = ChunkMetadata(source_type=SourceType.CODE, chunk_text="code")
        chunk1.add_tag("python")
        chunk1.add_reference("numpy")

        chunk2 = ChunkMetadata(source_type=SourceType.DOCUMENT, chunk_text="doc")
        chunk2.add_tag("important")

        store.add_chunk(chunk1)
        store.add_chunk(chunk2)

        stats = store.get_stats()

        assert stats["total_chunks"] == 2
        assert "code" in stats["source_types"]
        assert "document" in stats["source_types"]
        assert stats["relationship_stats"]["total_tags"] == 2
        assert stats["relationship_stats"]["total_references"] == 1

    def test_persistence(self):
        """Test saving and loading metadata."""
        store = MetadataStore()

        # Add some chunks
        chunk1 = ChunkMetadata(source_path="/test1.py", chunk_text="test1")
        chunk2 = ChunkMetadata(source_path="/test2.py", chunk_text="test2")

        store.add_chunk(chunk1)
        store.add_chunk(chunk2)

        # Use tempfile.mktemp to avoid Windows permission issues
        import tempfile

        temp_file = tempfile.mktemp(suffix=".pkl")
        try:
            # Save
            store.save_to_file(temp_file)

            # Create new store and load
            new_store = MetadataStore()
            new_store.load_from_file(temp_file)

            # Verify data was loaded
            assert len(new_store.get_all_chunks()) == 2
            loaded_chunks = new_store.get_all_chunks()
            paths = [chunk.source_path for chunk in loaded_chunks]
            assert "/test1.py" in paths
            assert "/test2.py" in paths
        finally:
            # Clean up
            try:
                os.unlink(temp_file)
            except (OSError, FileNotFoundError):
                pass

    def test_clear(self):
        """Test clearing all metadata."""
        store = MetadataStore()

        # Add some data
        chunk = ChunkMetadata(chunk_text="test")
        store.add_chunk(chunk)

        assert len(store.get_all_chunks()) == 1

        # Clear and verify empty
        store.clear()

        assert len(store.get_all_chunks()) == 0
        assert store.get_stats()["total_chunks"] == 0


class TestRelationshipStore:
    """Test the RelationshipStore class."""

    def test_empty_store_initialization(self):
        """Test initializing empty relationship store."""
        store = RelationshipStore()

        assert store.get_all_entities() == []
        assert store.get_relationship_types() == []

    def test_add_relationship(self):
        """Test adding relationships."""
        store = RelationshipStore()

        store.add_relationship("numpy", "chunk1", RelationshipStore.IMPORT)
        store.add_relationship("pandas", "chunk1", RelationshipStore.IMPORT)
        store.add_relationship("hello", "chunk2", RelationshipStore.FUNCTION_CALL)

        entities = store.get_all_entities()
        assert "numpy" in entities
        assert "pandas" in entities
        assert "hello" in entities

        types = store.get_relationship_types()
        assert RelationshipStore.IMPORT in types
        assert RelationshipStore.FUNCTION_CALL in types

    def test_get_related_chunks(self):
        """Test getting chunks related to an entity."""
        store = RelationshipStore()

        store.add_relationship("numpy", "chunk1", RelationshipStore.IMPORT)
        store.add_relationship("numpy", "chunk2", RelationshipStore.REFERENCE)

        # Get all relationships
        chunks = store.get_related_chunks("numpy")
        assert "chunk1" in chunks
        assert "chunk2" in chunks
        assert len(chunks) == 2

        # Get specific relationship type
        import_chunks = store.get_related_chunks("numpy", RelationshipStore.IMPORT)
        assert "chunk1" in import_chunks
        assert "chunk2" not in import_chunks
        assert len(import_chunks) == 1

    def test_get_chunk_entities(self):
        """Test getting entities related to a chunk."""
        store = RelationshipStore()

        store.add_relationship("numpy", "chunk1", RelationshipStore.IMPORT)
        store.add_relationship("pandas", "chunk1", RelationshipStore.IMPORT)
        store.add_relationship("hello", "chunk1", RelationshipStore.FUNCTION_CALL)

        entities = store.get_chunk_entities("chunk1")

        assert RelationshipStore.IMPORT in entities
        assert RelationshipStore.FUNCTION_CALL in entities
        assert "numpy" in entities[RelationshipStore.IMPORT]
        assert "pandas" in entities[RelationshipStore.IMPORT]
        assert "hello" in entities[RelationshipStore.FUNCTION_CALL]

    def test_find_related_entities(self):
        """Test finding entities related through relationships."""
        store = RelationshipStore()

        # Create relationship chain: numpy -> chunk1 <- pandas -> chunk2 <- scipy
        store.add_relationship("numpy", "chunk1", RelationshipStore.IMPORT)
        store.add_relationship("pandas", "chunk1", RelationshipStore.IMPORT)
        store.add_relationship("pandas", "chunk2", RelationshipStore.IMPORT)
        store.add_relationship("scipy", "chunk2", RelationshipStore.IMPORT)

        # Find entities related to numpy
        related = store.find_related_entities("numpy", max_depth=1)

        assert "pandas" in related  # Both are in chunk1
        # scipy should not be directly related at depth 1

    def test_calculate_relationship_strength(self):
        """Test calculating relationship strength between entities."""
        store = RelationshipStore()

        # Two entities with shared chunks
        store.add_relationship("numpy", "chunk1", RelationshipStore.IMPORT)
        store.add_relationship("numpy", "chunk2", RelationshipStore.IMPORT)
        store.add_relationship("pandas", "chunk1", RelationshipStore.IMPORT)  # Shared
        store.add_relationship("pandas", "chunk3", RelationshipStore.IMPORT)

        # Should have some relationship strength (Jaccard similarity)
        strength = store.calculate_relationship_strength("numpy", "pandas")
        assert 0 < strength <= 1

        # Entities with no shared chunks should have 0 strength
        store.add_relationship("scipy", "chunk4", RelationshipStore.IMPORT)
        strength = store.calculate_relationship_strength("numpy", "scipy")
        assert strength == 0

    def test_code_relationship_extraction(self):
        """Test extracting relationships from code chunks."""
        store = RelationshipStore()

        chunk = ChunkMetadata(
            source_type=SourceType.CODE,
            chunk_text="""
import numpy as np
from pandas import DataFrame

def hello():
    return "world"

class TestClass:
    pass
            """,
            chunk_id="test-chunk",
        )

        relationships = store.extract_code_relationships(chunk)

        # Should extract imports and function/class definitions
        entities = [rel[0] for rel in relationships]
        assert "numpy" in entities
        assert "pandas" in entities
        assert "hello" in entities
        assert "TestClass" in entities

    def test_persistence(self):
        """Test saving and loading relationships."""
        store = RelationshipStore()

        # Add some relationships
        store.add_relationship("numpy", "chunk1", RelationshipStore.IMPORT)
        store.add_relationship("pandas", "chunk2", RelationshipStore.REFERENCE)

        # Use tempfile.mktemp to avoid Windows permission issues
        import os
        import tempfile

        temp_file = tempfile.mktemp(suffix=".pkl")
        try:
            # Save
            store.save_to_file(temp_file)

            # Create new store and load
            new_store = RelationshipStore()
            new_store.load_from_file(temp_file)

            # Verify data was loaded
            entities = new_store.get_all_entities()
            assert "numpy" in entities
            assert "pandas" in entities

            chunks = new_store.get_related_chunks("numpy")
            assert "chunk1" in chunks
        finally:
            # Clean up
            try:
                os.unlink(temp_file)
            except (OSError, FileNotFoundError):
                pass

    def test_get_stats(self):
        """Test getting relationship statistics."""
        store = RelationshipStore()

        # Empty stats
        stats = store.get_stats()
        assert stats["total_entities"] == 0
        assert stats["total_relationships"] == 0

        # Add relationships
        store.add_relationship("numpy", "chunk1", RelationshipStore.IMPORT)
        store.add_relationship("pandas", "chunk2", RelationshipStore.IMPORT)
        store.add_relationship("hello", "chunk1", RelationshipStore.FUNCTION_CALL)

        stats = store.get_stats()
        assert stats["total_entities"] == 3
        assert stats["total_relationships"] == 3
        assert RelationshipStore.IMPORT in stats["relationship_types"]
        assert RelationshipStore.FUNCTION_CALL in stats["relationship_types"]
