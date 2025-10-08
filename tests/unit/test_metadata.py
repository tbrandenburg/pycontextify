"""Tests for metadata and relationship systems."""

import os
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from pycontextify.storage_metadata import ChunkMetadata, MetadataStore
from pycontextify.types import Chunk, SourceType

# RelationshipStore import removed - feature was deleted


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

    def test_from_chunk_conversion_basic(self):
        """Test converting Chunk DTO to ChunkMetadata."""
        chunk = Chunk(
            chunk_text="test content",
            source_path="/test/file.py",
            source_type=SourceType.CODE,
            start_char=0,
            end_char=12,
            embedding_provider="openai",
            embedding_model="text-embedding-3-small",
        )

        metadata = ChunkMetadata.from_chunk(chunk)

        # Verify all fields transferred correctly
        assert metadata.chunk_text == chunk.chunk_text
        assert metadata.source_path == chunk.source_path
        assert metadata.source_type == chunk.source_type
        assert metadata.start_char == chunk.start_char
        assert metadata.end_char == chunk.end_char
        assert metadata.embedding_provider == chunk.embedding_provider
        assert metadata.embedding_model == chunk.embedding_model

        # Verify storage-specific fields are generated
        assert metadata.chunk_id  # UUID generated
        assert isinstance(metadata.created_at, datetime)

    def test_from_chunk_conversion_with_metadata(self):
        """Test converting Chunk with full metadata fields."""
        chunk = Chunk(
            chunk_text="def hello(): pass",
            source_path="/test/file.py",
            source_type=SourceType.CODE,
            start_char=0,
            end_char=17,
            file_extension=".py",
            tags=["python", "function"],
            references=["stdlib", "builtins"],
            parent_section="module_main",
            code_symbols=["hello"],
            metadata={"complexity": "low", "lines": 1},
        )

        metadata = ChunkMetadata.from_chunk(chunk)

        # Verify all optional fields transferred
        assert metadata.file_extension == ".py"
        assert metadata.tags == ["python", "function"]
        assert metadata.references == ["stdlib", "builtins"]
        assert metadata.parent_section == "module_main"
        assert metadata.code_symbols == ["hello"]
        assert metadata.metadata == {"complexity": "low", "lines": 1}

    def test_from_chunk_conversion_independence(self):
        """Test that from_chunk creates independent copies of lists."""
        chunk = Chunk(
            chunk_text="test",
            source_path="/test.py",
            source_type=SourceType.CODE,
            tags=["original"],
            references=["ref1"],
            code_symbols=["sym1"],
        )

        metadata = ChunkMetadata.from_chunk(chunk)

        # Modify the original chunk's lists
        chunk.tags.append("new_tag")
        chunk.references.append("new_ref")
        chunk.code_symbols.append("new_sym")

        # Verify metadata lists are independent
        assert "new_tag" not in metadata.tags
        assert "new_ref" not in metadata.references
        assert "new_sym" not in metadata.code_symbols

        # Verify original values are preserved
        assert metadata.tags == ["original"]
        assert metadata.references == ["ref1"]
        assert metadata.code_symbols == ["sym1"]

    def test_from_chunk_conversion_empty_optionals(self):
        """Test converting Chunk with empty optional fields."""
        chunk = Chunk(
            chunk_text="minimal chunk",
            source_path="/test.txt",
            source_type=SourceType.DOCUMENT,
        )

        metadata = ChunkMetadata.from_chunk(chunk)

        # Verify optional fields are properly initialized
        assert metadata.file_extension is None
        assert metadata.tags == []
        assert metadata.references == []
        assert metadata.parent_section is None
        assert metadata.code_symbols == []
        assert metadata.metadata == {}


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


# TestRelationshipStore class removed - RelationshipStore feature was deleted
