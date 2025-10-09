"""Unit tests for postprocess module."""

import pytest

from pycontextify.postprocess import postprocess_file, postprocess_filebase


class TestPostprocessIdentity:
    """Test identity behavior of postprocess functions."""

    def test_postprocess_file_empty_list(self):
        """Test postprocess_file with empty list."""
        result = postprocess_file([])
        assert result == []
        assert isinstance(result, list)

    def test_postprocess_file_single_chunk(self):
        """Test postprocess_file with single chunk."""
        chunk = {"text": "test content", "metadata": {"topic": "test"}}
        result = postprocess_file([chunk])

        assert result == [chunk]
        assert len(result) == 1
        assert result[0] is chunk  # Should be the same object

    def test_postprocess_file_multiple_chunks(self):
        """Test postprocess_file with multiple chunks."""
        chunks = [
            {"text": "chunk1", "metadata": {"topic": "test"}},
            {"text": "chunk2", "metadata": {"topic": "test"}},
            {"text": "chunk3", "metadata": {"topic": "test"}},
        ]
        result = postprocess_file(chunks)

        assert result == chunks
        assert len(result) == 3
        # Should be the same objects
        for i, chunk in enumerate(result):
            assert chunk is chunks[i]

    def test_postprocess_file_preserves_metadata(self):
        """Test that postprocess_file preserves all metadata."""
        chunk = {
            "text": "content",
            "metadata": {
                "topic": "test",
                "full_path": "/path/to/file.txt",
                "chunk_index": 0,
                "language": "python",
            },
        }
        result = postprocess_file([chunk])

        assert result[0]["metadata"] == chunk["metadata"]
        assert result[0]["metadata"]["topic"] == "test"
        assert result[0]["metadata"]["chunk_index"] == 0

    def test_postprocess_filebase_empty_list(self):
        """Test postprocess_filebase with empty list."""
        result = postprocess_filebase([])
        assert result == []
        assert isinstance(result, list)

    def test_postprocess_filebase_single_chunk(self):
        """Test postprocess_filebase with single chunk."""
        chunk = {"text": "test content", "metadata": {"topic": "test"}}
        result = postprocess_filebase([chunk])

        assert result == [chunk]
        assert len(result) == 1
        assert result[0] is chunk

    def test_postprocess_filebase_multiple_chunks(self):
        """Test postprocess_filebase with multiple chunks."""
        chunks = [
            {"text": "chunk1", "metadata": {"topic": "topic1"}},
            {"text": "chunk2", "metadata": {"topic": "topic2"}},
            {"text": "chunk3", "metadata": {"topic": "topic1"}},
        ]
        result = postprocess_filebase(chunks)

        assert result == chunks
        assert len(result) == 3

    def test_postprocess_filebase_preserves_order(self):
        """Test that postprocess_filebase preserves chunk order."""
        chunks = [
            {"text": "first", "metadata": {"chunk_index": 0}},
            {"text": "second", "metadata": {"chunk_index": 1}},
            {"text": "third", "metadata": {"chunk_index": 2}},
        ]
        result = postprocess_filebase(chunks)

        for i, chunk in enumerate(result):
            assert chunk["metadata"]["chunk_index"] == i

    def test_postprocess_file_with_complex_metadata(self):
        """Test postprocess_file with complex metadata structures."""
        chunk = {
            "text": "content",
            "metadata": {
                "topic": "test",
                "links": {"imports": ["module1", "module2"], "exports": ["func1"]},
                "language": "python",
                "chunk_name": "function_name",
            },
        }
        result = postprocess_file([chunk])

        assert result[0]["metadata"]["links"] == chunk["metadata"]["links"]
        assert result[0]["metadata"]["language"] == "python"

    def test_postprocess_filebase_with_mixed_topics(self):
        """Test postprocess_filebase with chunks from different topics."""
        chunks = [
            {"text": "code chunk", "metadata": {"topic": "code"}},
            {"text": "doc chunk", "metadata": {"topic": "docs"}},
            {"text": "data chunk", "metadata": {"topic": "data"}},
        ]
        result = postprocess_filebase(chunks)

        # Should preserve all chunks unchanged
        assert len(result) == 3
        assert result[0]["metadata"]["topic"] == "code"
        assert result[1]["metadata"]["topic"] == "docs"
        assert result[2]["metadata"]["topic"] == "data"

    def test_both_functions_are_identity(self):
        """Test that both functions act as identity functions."""
        chunks = [
            {"text": "chunk1", "metadata": {"topic": "test"}},
            {"text": "chunk2", "metadata": {"topic": "test"}},
        ]

        # Both should return the same result
        file_result = postprocess_file(chunks)
        filebase_result = postprocess_filebase(chunks)

        assert file_result == filebase_result
        assert file_result == chunks

    def test_postprocess_file_does_not_modify_input(self):
        """Test that postprocess_file doesn't modify the input list."""
        original_chunks = [
            {"text": "chunk1", "metadata": {"topic": "test"}},
            {"text": "chunk2", "metadata": {"topic": "test"}},
        ]
        # Make a copy to compare later
        chunks_copy = [chunk.copy() for chunk in original_chunks]

        result = postprocess_file(original_chunks)

        # Original should be unchanged
        for i, chunk in enumerate(original_chunks):
            assert chunk["text"] == chunks_copy[i]["text"]
            assert chunk["metadata"] == chunks_copy[i]["metadata"]

    def test_postprocess_filebase_does_not_modify_input(self):
        """Test that postprocess_filebase doesn't modify the input list."""
        original_chunks = [
            {"text": "chunk1", "metadata": {"topic": "test"}},
            {"text": "chunk2", "metadata": {"topic": "test"}},
        ]
        chunks_copy = [chunk.copy() for chunk in original_chunks]

        result = postprocess_filebase(original_chunks)

        # Original should be unchanged
        for i, chunk in enumerate(original_chunks):
            assert chunk["text"] == chunks_copy[i]["text"]
            assert chunk["metadata"] == chunks_copy[i]["metadata"]

    def test_postprocess_file_return_type(self):
        """Test that postprocess_file returns a list."""
        result = postprocess_file([])
        assert isinstance(result, list)

        result = postprocess_file([{"text": "test", "metadata": {}}])
        assert isinstance(result, list)

    def test_postprocess_filebase_return_type(self):
        """Test that postprocess_filebase returns a list."""
        result = postprocess_filebase([])
        assert isinstance(result, list)

        result = postprocess_filebase([{"text": "test", "metadata": {}}])
        assert isinstance(result, list)
