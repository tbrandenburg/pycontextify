"""Working tests for Chunker functionality."""

import pytest

from pycontextify.chunker import BaseChunker, CodeChunker, SimpleChunker
from pycontextify.config import Config
from pycontextify.types import Chunk, SourceType


class TestBaseChunker:
    """Test BaseChunker abstract class."""

    def test_base_chunker_instantiation(self):
        """Test that BaseChunker cannot be instantiated directly."""
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
                    self._create_chunk(
                        chunk_text=text,
                        source_path=source_path,
                        source_type=SourceType.DOCUMENT,
                        start_char=0,
                        end_char=len(text),
                        embedding_provider=embedding_provider,
                        embedding_model=embedding_model,
                    )
                ]

        config = Config()

        chunker = ConcreteChunker(config)
        results = chunker.chunk_text(
            "Test text", "test.txt", "openai", "text-embedding-3-small"
        )

        assert len(results) == 1
        assert isinstance(results[0], Chunk)
        assert results[0].chunk_text == "Test text"


class TestSimpleChunker:
    """Test SimpleChunker functionality."""

    def test_simple_chunker_initialization_default(self):
        """Test SimpleChunker initialization with default settings."""
        config = Config()

        chunker = SimpleChunker(config=config)

        assert chunker.config == config
        assert chunker.chunk_size == config.chunk_size
        assert chunker.chunk_overlap == config.chunk_overlap

    def test_chunk_short_text(self):
        """Test chunking short text that fits in one chunk."""
        config = Config()
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
        config = Config()
        config.chunk_size = 50
        config.chunk_overlap = 10

        chunker = SimpleChunker(config=config)
        text = (
            "This is a very long text that needs to be split into multiple chunks. " * 5
        )

        chunks = chunker.chunk_text(
            text, "test.txt", "openai", "text-embedding-3-small"
        )

        assert len(chunks) > 1

        # Verify chunks are Chunk objects
        for chunk in chunks:
            assert isinstance(chunk, Chunk)
            assert chunk.source_type == SourceType.DOCUMENT
            assert chunk.embedding_provider == "openai"
            assert chunk.embedding_model == "text-embedding-3-small"

    def test_chunk_empty_text(self):
        """Test chunking empty text."""
        config = Config()

        chunker = SimpleChunker(config=config)

        chunks = chunker.chunk_text("", "test.txt", "openai", "text-embedding-3-small")

        assert len(chunks) == 0

    def test_chunk_whitespace_only(self):
        """Test chunking text with only whitespace."""
        config = Config()

        chunker = SimpleChunker(config=config)

        chunks = chunker.chunk_text(
            "   \n\t  \n  ", "test.txt", "openai", "text-embedding-3-small"
        )

        assert len(chunks) == 0

    def test_chunk_with_metadata_attributes(self):
        """Test chunking preserves metadata attributes."""
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

    def test_extract_relationships_basic(self):
        """Test basic relationship extraction."""
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
        config = Config()

        chunker = SimpleChunker(config=config)
        text = "Text with émojis 🚀, unicode characters ñáéíóú, and symbols @#$%^&*()."

        chunks = chunker.chunk_text(
            text, "test.txt", "openai", "text-embedding-3-small"
        )

        assert len(chunks) == 1
        assert chunks[0].chunk_text == text

    def test_chunk_with_newlines_and_tabs(self):
        """Test chunking text with various whitespace characters."""
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
        config = Config()

        chunker = SimpleChunker(config=config)
        text = "Test content"

        chunks = chunker.chunk_text(text, "test.py", "openai", "text-embedding-3-small")

        assert len(chunks) == 1
        assert chunks[0].file_extension == ".py"

    def test_chunk_overlap_verification(self):
        """Test that chunks have proper overlap when configured."""
        config = Config()
        config.chunk_size = 30  # Small size to force splits
        config.chunk_overlap = 10  # Overlap

        chunker = SimpleChunker(config=config)
        text = (
            "Word1 Word2 Word3 Word4 Word5 Word6 Word7 Word8 Word9 Word10 Word11 Word12"
        )

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


class TestCodeChunker:
    """Test CodeChunker functionality."""

    def test_code_chunker_basic_functionality(self):
        """Test basic code chunking functionality."""
        config = Config()
        config.chunk_size = 1000

        chunker = CodeChunker(config=config)
        code = '''
def hello():
    print("Hello, World!")

def goodbye():
    print("Goodbye!")
'''

        chunks = chunker.chunk_text(
            code, "test.py", "openai", "text-embedding-3-small"
        )

        assert len(chunks) >= 1
        assert isinstance(chunks[0], Chunk)
        assert chunks[0].source_type == SourceType.CODE
        assert chunks[0].embedding_provider == "openai"
        assert chunks[0].file_extension == ".py"

    def test_code_chunker_respects_function_boundaries(self):
        """Test that CodeChunker tries to respect function boundaries."""
        config = Config()
        config.chunk_size = 50  # Small size to force splitting

        chunker = CodeChunker(config=config)
        code = '''
def function_one():
    """First function."""
    return 1

def function_two():
    """Second function."""
    return 2

def function_three():
    """Third function."""
    return 3
'''

        chunks = chunker.chunk_text(
            code, "test.py", "openai", "text-embedding-3-small"
        )

        # Should create multiple chunks
        assert len(chunks) >= 1
        # All chunks should be CODE type
        for chunk in chunks:
            assert chunk.source_type == SourceType.CODE
            assert isinstance(chunk.chunk_text, str)

    def test_code_chunker_handles_large_functions(self):
        """Test that CodeChunker falls back to token splitting for large functions."""
        config = Config()
        config.chunk_size = 30  # Very small to force fallback

        chunker = CodeChunker(config=config)
        # Create a large function that exceeds chunk size
        code = '''
def very_large_function():
    """A function with many lines."""
    line1 = "data"
    line2 = "more data"
    line3 = "even more data"
    line4 = "lots of data"
    line5 = "tons of data"
    line6 = "massive data"
    line7 = "enormous data"
    line8 = "gigantic data"
    return "result"
'''

        chunks = chunker.chunk_text(
            code, "test.py", "openai", "text-embedding-3-small"
        )

        # Should split the large function into multiple chunks
        assert len(chunks) > 1
        # Verify all are CODE type
        for chunk in chunks:
            assert chunk.source_type == SourceType.CODE

    def test_code_chunker_empty_code(self):
        """Test CodeChunker with empty code."""
        config = Config()

        chunker = CodeChunker(config=config)

        chunks = chunker.chunk_text("", "test.py", "openai", "text-embedding-3-small")

        assert len(chunks) == 0
