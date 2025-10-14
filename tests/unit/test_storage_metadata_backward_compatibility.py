"""Tests for backward compatibility of storage metadata.

These tests ensure that old metadata files with deprecated SourceType
values can still be loaded without causing server startup failures.
"""

import logging
from datetime import datetime

import pytest

from pycontextify.storage_metadata import ChunkMetadata
from pycontextify.types import SourceType


class TestSourceTypeBackwardCompatibility:
    """Test backward compatibility for SourceType enum changes."""

    def test_deprecated_webpage_sourcetype_converted_to_document(self, caplog):
        """Test that deprecated 'webpage' SourceType is converted to DOCUMENT."""
        # Create metadata dict with deprecated 'webpage' SourceType
        metadata_dict = {
            "chunk_id": "test_chunk_123",
            "source_path": "/test/path.html",
            "source_type": "webpage",  # Deprecated value
            "chunk_text": "Test webpage content",
            "start_char": 0,
            "end_char": 20,
            "created_at": datetime.now().isoformat(),
            "file_extension": ".html",
            "embedding_provider": "sentence_transformers",
            "embedding_model": "all-mpnet-base-v2",
            "tags": ["web", "content"],
            "references": [],
            "parent_section": None,
            "code_symbols": [],
            "metadata": {},
        }

        # Capture log messages
        with caplog.at_level(logging.WARNING):
            # This should not raise an exception
            chunk_metadata = ChunkMetadata.from_dict(metadata_dict)

        # Verify the deprecated SourceType was converted
        assert chunk_metadata.source_type == SourceType.DOCUMENT
        assert chunk_metadata.chunk_id == "test_chunk_123"
        assert chunk_metadata.source_path == "/test/path.html"

        # Verify warning was logged
        assert len(caplog.records) == 1
        assert "Converting deprecated SourceType 'webpage' to 'document'" in caplog.text
        assert "test_chunk_123" in caplog.text

    def test_unknown_sourcetype_converted_to_document(self, caplog):
        """Test that unknown SourceType values are converted to DOCUMENT."""
        metadata_dict = {
            "chunk_id": "test_chunk_456",
            "source_path": "/test/path.unknown",
            "source_type": "unknown_type",  # Unknown value
            "chunk_text": "Test content",
            "start_char": 0,
            "end_char": 12,
            "created_at": datetime.now().isoformat(),
            "file_extension": ".unknown",
            "embedding_provider": "sentence_transformers",
            "embedding_model": "all-mpnet-base-v2",
            "tags": [],
            "references": [],
            "parent_section": None,
            "code_symbols": [],
            "metadata": {},
        }

        with caplog.at_level(logging.WARNING):
            chunk_metadata = ChunkMetadata.from_dict(metadata_dict)

        # Verify the unknown SourceType was converted
        assert chunk_metadata.source_type == SourceType.DOCUMENT
        assert chunk_metadata.chunk_id == "test_chunk_456"

        # Verify warning was logged
        assert len(caplog.records) == 1
        assert (
            "Unknown SourceType 'unknown_type' converted to 'document'" in caplog.text
        )
        assert "test_chunk_456" in caplog.text

    def test_valid_sourcetype_unchanged(self):
        """Test that valid SourceType values are not modified."""
        metadata_dict = {
            "chunk_id": "test_chunk_789",
            "source_path": "/test/path.py",
            "source_type": "code",  # Valid value
            "chunk_text": "def test(): pass",
            "start_char": 0,
            "end_char": 16,
            "created_at": datetime.now().isoformat(),
            "file_extension": ".py",
            "embedding_provider": "sentence_transformers",
            "embedding_model": "all-mpnet-base-v2",
            "tags": [],
            "references": [],
            "parent_section": None,
            "code_symbols": ["test"],
            "metadata": {},
        }

        # This should work without any warnings
        chunk_metadata = ChunkMetadata.from_dict(metadata_dict)

        # Verify the valid SourceType is preserved
        assert chunk_metadata.source_type == SourceType.CODE
        assert chunk_metadata.chunk_id == "test_chunk_789"
        assert chunk_metadata.source_path == "/test/path.py"

    def test_multiple_chunks_with_mixed_sourcetypes(self, caplog):
        """Test loading multiple chunks with mixed valid and invalid SourceTypes."""
        chunks_data = [
            {
                "chunk_id": "valid_chunk",
                "source_path": "/test/valid.py",
                "source_type": "code",
                "chunk_text": "Valid code",
                "start_char": 0,
                "end_char": 10,
                "created_at": datetime.now().isoformat(),
            },
            {
                "chunk_id": "deprecated_chunk",
                "source_path": "/test/old.html",
                "source_type": "webpage",
                "chunk_text": "Old webpage",
                "start_char": 0,
                "end_char": 11,
                "created_at": datetime.now().isoformat(),
            },
            {
                "chunk_id": "unknown_chunk",
                "source_path": "/test/unknown.xyz",
                "source_type": "mystery_type",
                "chunk_text": "Mystery content",
                "start_char": 0,
                "end_char": 15,
                "created_at": datetime.now().isoformat(),
            },
        ]

        with caplog.at_level(logging.WARNING):
            chunks = []
            for chunk_data in chunks_data:
                # Add required fields with defaults
                chunk_data.setdefault("file_extension", None)
                chunk_data.setdefault("embedding_provider", "sentence_transformers")
                chunk_data.setdefault("embedding_model", "all-mpnet-base-v2")
                chunk_data.setdefault("tags", [])
                chunk_data.setdefault("references", [])
                chunk_data.setdefault("parent_section", None)
                chunk_data.setdefault("code_symbols", [])
                chunk_data.setdefault("metadata", {})

                chunks.append(ChunkMetadata.from_dict(chunk_data))

        # Verify all chunks were created successfully
        assert len(chunks) == 3

        # Check first chunk (valid)
        assert chunks[0].source_type == SourceType.CODE
        assert chunks[0].chunk_id == "valid_chunk"

        # Check second chunk (deprecated 'webpage' -> DOCUMENT)
        assert chunks[1].source_type == SourceType.DOCUMENT
        assert chunks[1].chunk_id == "deprecated_chunk"

        # Check third chunk (unknown -> DOCUMENT)
        assert chunks[2].source_type == SourceType.DOCUMENT
        assert chunks[2].chunk_id == "unknown_chunk"

        # Verify warnings were logged for deprecated and unknown types
        assert len(caplog.records) == 2
        warning_messages = [record.message for record in caplog.records]
        assert any(
            "webpage" in msg and "deprecated_chunk" in msg for msg in warning_messages
        )
        assert any(
            "mystery_type" in msg and "unknown_chunk" in msg for msg in warning_messages
        )
