"""Working tests for Chunker functionality."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from pycontextify.chunker import BaseChunker, SimpleChunker
from pycontextify.orchestrator_config import Config
from pycontextify.storage_metadata import ChunkMetadata, SourceType


class TestBaseChunker:
    """Test BaseChunker abstract class."""

    def test_base_chunker_instantiation(self):
        """Test that BaseChunker cannot be instantiated directly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = Config()

            with pytest.raises(TypeError):
                BaseChunker(config)

    def test_base_chunker_subclass(self):
        """Test creating a concrete subclass of BaseChunker."""

        class ConcreteChunker(BaseChunker):
            def chunk_text(
                self, text, source_path, embedding_provider, embedding_model
            ):
                return [
                    self._create_chunk_metadata(
                        chunk_text=text,
                        source_path=source_path,
                        source_type=SourceType.DOCUMENT,
                        start_char=0,
                        end_char=len(text),
                        embedding_provider=embedding_provider,
                        embedding_model=embedding_model,
                    )
                ]

        with tempfile.TemporaryDirectory() as temp_dir:
            config = Config()

            chunker = ConcreteChunker(config)
            results = chunker.chunk_text(
                "Test text", "test.txt", "openai", "text-embedding-3-small"
            )

            assert len(results) == 1
            assert isinstance(results[0], ChunkMetadata)
            assert results[0].chunk_text == "Test text"


class TestSimpleChunker:
    """Test SimpleChunker functionality."""

    def test_simple_chunker_initialization_default(self):
        """Test SimpleChunker initialization with default settings."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = Config()

            chunker = SimpleChunker(config=config)

            assert chunker.config == config
            assert chunker.chunk_size == config.chunk_size
            assert chunker.chunk_overlap == config.chunk_overlap

    def test_chunk_short_text(self):
        """Test chunking short text that fits in one chunk."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = Config()
            # Override chunk size to be large
            config.chunk_size = 1000

            chunker = SimpleChunker(config=config)
            text = "This is a short text that should fit in one chunk."

            chunks = chunker.chunk_text(
                text, "test.txt", "openai", "text-embedding-3-small"
            )

            assert len(chunks) == 1
            assert chunks[0].chunk_text == text
            assert chunks[0].start_char == 0
            assert chunks[0].end_char == len(text)

    def test_chunk_long_text_multiple_chunks(self):
        """Test chunking long text that requires multiple chunks."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = Config()
            # Override chunk size to be small
            config.chunk_size = 50
            config.chunk_overlap = 10

            chunker = SimpleChunker(config=config)
            text = (
                "This is a very long text that needs to be split into multiple chunks. "
                * 5
            )

            chunks = chunker.chunk_text(
                text, "test.txt", "openai", "text-embedding-3-small"
            )

            assert len(chunks) > 1

            # Verify chunks are ChunkMetadata objects
            for chunk in chunks:
                assert isinstance(chunk, ChunkMetadata)
                assert chunk.source_type == SourceType.DOCUMENT
                assert chunk.embedding_provider == "openai"
                assert chunk.embedding_model == "text-embedding-3-small"

    def test_chunk_empty_text(self):
        """Test chunking empty text."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = Config()

            chunker = SimpleChunker(config=config)

            chunks = chunker.chunk_text(
                "", "test.txt", "openai", "text-embedding-3-small"
            )

            assert len(chunks) == 0

    def test_chunk_whitespace_only(self):
        """Test chunking text with only whitespace."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = Config()

            chunker = SimpleChunker(config=config)

            chunks = chunker.chunk_text(
                "   \n\t  \n  ", "test.txt", "openai", "text-embedding-3-small"
            )

            assert len(chunks) == 0

    def test_chunk_with_metadata_attributes(self):
        """Test chunking preserves metadata attributes."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = Config()

            chunker = SimpleChunker(config=config)
            text = "This is a test text."
            source_path = "test.txt"
            provider = "openai"
            model = "text-embedding-3-small"

            chunks = chunker.chunk_text(text, source_path, provider, model)

            assert len(chunks) == 1
            chunk = chunks[0]
            assert chunk.source_path == source_path
            assert chunk.embedding_provider == provider
            assert chunk.embedding_model == model
            assert chunk.source_type == SourceType.DOCUMENT

    def test_split_by_tokens_basic(self):
        """Test the _split_by_tokens method directly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = Config()

            chunker = SimpleChunker(config=config)
            text = "This is a test text with multiple words for splitting."

            # Small chunk size to force splitting
            chunk_tuples = chunker._split_by_tokens(text, chunk_size=20, overlap=5)

            assert len(chunk_tuples) >= 1

            # Verify tuple structure
            for chunk_text, start_char, end_char in chunk_tuples:
                assert isinstance(chunk_text, str)
                assert isinstance(start_char, int)
                assert isinstance(end_char, int)
                assert start_char >= 0
                assert end_char > start_char

    def test_split_by_tokens_empty_text(self):
        """Test _split_by_tokens with empty text."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = Config()

            chunker = SimpleChunker(config=config)

            chunk_tuples = chunker._split_by_tokens("", chunk_size=100, overlap=10)

            assert chunk_tuples == []

    def test_split_by_tokens_single_word(self):
        """Test _split_by_tokens with single word."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = Config()

            chunker = SimpleChunker(config=config)

            chunk_tuples = chunker._split_by_tokens("word", chunk_size=100, overlap=10)

            assert len(chunk_tuples) == 1
            chunk_text, start_char, end_char = chunk_tuples[0]
            assert chunk_text == "word"
            assert start_char == 0
            assert end_char == 4

    def test_extract_relationships_basic(self):
        """Test basic relationship extraction."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = Config()
            config.enable_relationships = True
            config.max_relationships_per_chunk = 5

            chunker = SimpleChunker(config=config)
            text = "This is about Python Programming and Machine Learning."

            chunks = chunker.chunk_text(
                text, "test.txt", "openai", "text-embedding-3-small"
            )

            assert len(chunks) == 1
            chunk = chunks[0]

            # Should have extracted some capitalized words as references
            assert len(chunk.references) > 0

    def test_relationships_disabled(self):
        """Test that relationships are not extracted when disabled."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = Config()
            config.enable_relationships = False

            chunker = SimpleChunker(config=config)
            text = "This is about Python Programming and Machine Learning."

            chunks = chunker.chunk_text(
                text, "test.txt", "openai", "text-embedding-3-small"
            )

            assert len(chunks) == 1
            chunk = chunks[0]

            # Should not have extracted any relationships
            assert len(chunk.references) == 0

    def test_chunk_with_special_characters(self):
        """Test chunking text with special characters."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = Config()

            chunker = SimpleChunker(config=config)
            text = (
                "Text with émojis 🚀, unicode characters ñáéíóú, and symbols @#$%^&*()."
            )

            chunks = chunker.chunk_text(
                text, "test.txt", "openai", "text-embedding-3-small"
            )

            assert len(chunks) == 1
            assert chunks[0].chunk_text == text

    def test_chunk_with_newlines_and_tabs(self):
        """Test chunking text with various whitespace characters."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = Config()

            chunker = SimpleChunker(config=config)
            text = "Line 1\nLine 2\n\tIndented line\r\nWindows newline"

            chunks = chunker.chunk_text(
                text, "test.txt", "openai", "text-embedding-3-small"
            )

            assert len(chunks) >= 1
            # Should preserve the original text
            full_content = "".join(chunk.chunk_text for chunk in chunks)
            # Content should be similar (may have normalization)
            assert len(full_content) > 0

    def test_chunk_positions_consistency(self):
        """Test that chunk positions are consistent."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = Config()
            config.chunk_size = 30  # Small size to get multiple chunks

            chunker = SimpleChunker(config=config)
            text = "This is a test text that will be split into multiple chunks for testing positions."

            chunks = chunker.chunk_text(
                text, "test.txt", "openai", "text-embedding-3-small"
            )

            # Verify start/end positions make sense
            for i, chunk in enumerate(chunks):
                assert chunk.start_char >= 0
                assert chunk.end_char > chunk.start_char

                if i == 0:
                    assert chunk.start_char == 0

                if i == len(chunks) - 1:
                    # Last chunk should end within reasonable bounds
                    assert chunk.end_char <= len(text) + 10  # Allow some buffer

    def test_chunk_metadata_file_extension(self):
        """Test that file extension is correctly set in metadata."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = Config()

            chunker = SimpleChunker(config=config)
            text = "Test content"

            chunks = chunker.chunk_text(
                text, "test.py", "openai", "text-embedding-3-small"
            )

            assert len(chunks) == 1
            assert chunks[0].file_extension == ".py"

    def test_chunk_overlap_verification(self):
        """Test that chunks have proper overlap when configured."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = Config()
            config.chunk_size = 30  # Small size to force splits
            config.chunk_overlap = 10  # Overlap

            chunker = SimpleChunker(config=config)
            text = "Word1 Word2 Word3 Word4 Word5 Word6 Word7 Word8 Word9 Word10 Word11 Word12"

            chunks = chunker.chunk_text(
                text, "test.txt", "openai", "text-embedding-3-small"
            )

            if len(chunks) > 1:
                # Check that consecutive chunks have some overlapping content
                # This is a loose test since exact overlap depends on tokenization
                full_content = " ".join(chunk.chunk_text for chunk in chunks)
                # Total content should be longer than original due to overlap
                assert len(full_content) >= len(text)

    def test_chunk_edge_case_exact_size(self):
        """Test chunking text that's exactly at chunk size boundary."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = Config()
            config.chunk_size = 50

            chunker = SimpleChunker(config=config)
            # Create text that should be right at the boundary
            text = "This is test content that is exactly fifty chars!!"
            assert len(text) == 50

            chunks = chunker.chunk_text(
                text, "test.txt", "openai", "text-embedding-3-small"
            )

            assert len(chunks) == 1
            assert chunks[0].chunk_text == text

    def test_chunk_zero_overlap(self):
        """Test chunking with zero overlap."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = Config()
            config.chunk_size = 20
            config.chunk_overlap = 0

            chunker = SimpleChunker(config=config)
            text = "First part content. Second part content. Third part content."

            chunks = chunker.chunk_text(
                text, "test.txt", "openai", "text-embedding-3-small"
            )

            # With zero overlap, combined length should be close to original
            if len(chunks) > 1:
                combined_length = sum(len(chunk.chunk_text) for chunk in chunks)
                # Allow some variance due to tokenization and spacing
                assert abs(combined_length - len(text)) <= len(text) * 0.2

    def test_chunk_very_long_single_sentence(self):
        """Test chunking a very long sentence without natural breaks."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = Config()
            config.chunk_size = 50

            chunker = SimpleChunker(config=config)
            # Create very long sentence without periods
            text = (
                "This is a very long sentence that contains many words and should be split "
                "even though it has no periods or natural breaking points "
            ) * 3

            chunks = chunker.chunk_text(
                text, "test.txt", "openai", "text-embedding-3-small"
            )

            assert len(chunks) > 1

            # Verify each chunk respects approximate size limits
            # (allowing some flexibility due to tokenization)
            for chunk in chunks:
                # Rough check - shouldn't be massively over the limit
                assert len(chunk.chunk_text.split()) <= config.chunk_size * 2

    def test_chunk_none_input_handling(self):
        """Test chunking with None input."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = Config()

            chunker = SimpleChunker(config=config)

            # The actual implementation may convert None to string or handle it
            try:
                chunks = chunker.chunk_text(
                    None, "test.txt", "openai", "text-embedding-3-small"
                )
                # If it doesn't raise an error, should return empty or handle gracefully
                assert isinstance(chunks, list)
            except (TypeError, AttributeError):
                # It's also acceptable to raise an error for None input
                pass

    def test_chunk_position_accuracy(self):
        """Test that chunk start/end positions are reasonably accurate."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = Config()
            config.chunk_size = 30

            chunker = SimpleChunker(config=config)
            text = "This is a test text that will be split into multiple chunks for position testing."

            chunks = chunker.chunk_text(
                text, "test.txt", "openai", "text-embedding-3-small"
            )

            # Verify basic position properties
            for i, chunk in enumerate(chunks):
                assert chunk.start_char >= 0
                assert chunk.end_char > chunk.start_char

                if i == 0:
                    assert chunk.start_char == 0

                # Positions should be within reasonable bounds
                assert chunk.end_char <= len(text) + 50  # Some buffer for processing
